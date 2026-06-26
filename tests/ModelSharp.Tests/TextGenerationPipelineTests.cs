using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Manifest;
using ModelSharp.Pipeline;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Covers the high-level text-generation wiring — <see cref="TextGenerationProcessor"/> (BPE
/// tokenizer + decoder options from the manifest) and the generation-flavoured <see cref="Pipeline"/>
/// (<see cref="Pipeline.Generate"/> / <see cref="Pipeline.GenerateStream"/>). No real LLM is
/// downloaded: a tiny synthetic byte-level vocab and a scripted <see cref="IExecutionEngine"/> make
/// the whole path deterministic and asset-free.
/// </summary>
public class TextGenerationPipelineTests
{
    // ----------------------------------------------------------------------------------------
    // Tiny synthetic byte-level BPE assets
    // ----------------------------------------------------------------------------------------

    // ASCII letters survive GPT-2 byte-level mapping unchanged (printable range), so a vocab of
    // single-letter tokens decodes id -> letter directly and round-trips through the tokenizer.
    private const string Letters = "abcdefghijklmnopqrstuvwxyz";

    /// <summary>Writes a tiny vocab.json (single-letter tokens) + empty merges.txt and returns the dir.</summary>
    private static (string dir, string vocab, string merges) WriteTinyBpeAssets()
    {
        string dir = Path.Combine(Path.GetTempPath(), "modelsharp-tg-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        var entries = new List<string>();
        for (int i = 0; i < Letters.Length; i++)
            entries.Add($"  \"{Letters[i]}\": {i}");
        string vocabJson = "{\n" + string.Join(",\n", entries) + "\n}\n";

        string vocab = Path.Combine(dir, "vocab.json");
        string merges = Path.Combine(dir, "merges.txt");
        File.WriteAllText(vocab, vocabJson);
        File.WriteAllText(merges, "#version: 0.2\n");   // header only; no merge rules needed for single chars
        return (dir, vocab, merges);
    }

    /// <summary>A scripted no-cache engine emitting one-hot logits whose argmax follows a fixed script.</summary>
    private sealed class ScriptedEngine : IExecutionEngine
    {
        private readonly int[] _peaks;
        private readonly int _vocab;
        private int _call;

        public ScriptedEngine(int vocab, int[] peaks) { _vocab = vocab; _peaks = peaks; }

        public IReadOnlyList<TensorInfo> Inputs { get; } = new[] { new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()) };
        public IReadOnlyList<TensorInfo> Outputs { get; } = new[] { new TensorInfo("logits", ElementType.Float32, Array.Empty<int>()) };

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            int peak = _peaks[Math.Min(_call, _peaks.Length - 1)];
            _call++;
            var row = new float[_vocab];
            row[peak] = 10f;
            var logits = Tensor<float>.FromArray(new TensorShape(1, 1, _vocab), row);
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["logits"] = new NamedTensor("logits", logits),
            };
        }

        public void Dispose() { }
    }

    /// <summary>A uniform-logits engine: the chosen token is driven entirely by the seeded RNG.</summary>
    private sealed class UniformEngine : IExecutionEngine
    {
        private readonly int _vocab;
        public UniformEngine(int vocab) => _vocab = vocab;

        public IReadOnlyList<TensorInfo> Inputs { get; } = new[] { new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()) };
        public IReadOnlyList<TensorInfo> Outputs { get; } = new[] { new TensorInfo("logits", ElementType.Float32, Array.Empty<int>()) };

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
            => new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["logits"] = new NamedTensor("logits", Tensor<float>.FromArray(new TensorShape(1, 1, _vocab), new float[_vocab])),
            };

        public void Dispose() { }
    }

    private static ProcessorContext Ctx(string vocab, string merges, IReadOnlyList<string> inputs)
    {
        var manifest = new ModelManifest
        {
            Task = ModelTask.TextGeneration,
            Extra = new Dictionary<string, string> { ["vocab"] = vocab, ["merges"] = merges },
        };
        return new ProcessorContext(manifest, inputs, new[] { "logits" });
    }

    // ----------------------------------------------------------------------------------------
    // TextGenerationProcessor wiring
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void Processor_Builds_Tokenizer_And_RoundTrips_Letters()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var proc = new TextGenerationProcessor(Ctx(vocab, merges, new[] { "input_ids" }));

            IReadOnlyList<long> ids = proc.Encode("cab");
            Assert.Equal(new long[] { 2, 0, 1 }, ids.ToArray());  // c=2, a=0, b=1
            Assert.Equal("cab", proc.Decode(ids));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Processor_Reads_DecoderOptions_And_Config_From_Manifest()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var manifest = new ModelManifest
            {
                Task = ModelTask.TextGeneration,
                Extra = new Dictionary<string, string>
                {
                    ["vocab"] = vocab,
                    ["merges"] = merges,
                    ["kv_num_heads"] = "4",
                    ["kv_head_dim"] = "8",
                    ["eos_token_id"] = "5",
                    ["max_new_tokens"] = "7",
                },
            };
            var proc = new TextGenerationProcessor(new ProcessorContext(manifest, new[] { "input_ids" }, new[] { "logits" }));

            Assert.Equal(4, proc.DecoderOptions.KvCacheNumHeads);
            Assert.Equal(8, proc.DecoderOptions.KvCacheHeadDim);
            Assert.Equal(7, proc.DefaultConfig.MaxNewTokens);
            Assert.Equal(new[] { 5 }, proc.DefaultConfig.EosTokenIds);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Processor_Throws_When_Tokenizer_Files_Missing()
    {
        var manifest = new ModelManifest { Task = ModelTask.TextGeneration };
        Assert.Throws<ModelSharpException>(
            () => new TextGenerationProcessor(new ProcessorContext(manifest, new[] { "input_ids" }, new[] { "logits" })));
    }

    // ----------------------------------------------------------------------------------------
    // Pipeline.Generate / GenerateStream
    // ----------------------------------------------------------------------------------------

    private static ModelSharp.Pipeline.Pipeline BuildPipeline(string vocab, string merges, IExecutionEngine engine)
    {
        ProcessorContext ctx = Ctx(vocab, merges, engine.Inputs.Select(i => i.Name).ToList());
        var proc = new TextGenerationProcessor(ctx);
        TextGenerator generator = proc.CreateGenerator(engine);
        return new ModelSharp.Pipeline.Pipeline(engine, ctx.Manifest, generator, proc);
    }

    [Fact]
    public void Generate_Decodes_Greedy_Tokens_To_Text()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            // Greedy script: emit "dog" (d=3, o=14, g=6), then never reached (MaxNewTokens caps at 3).
            var engine = new ScriptedEngine(Letters.Length, new[] { 3, 14, 6 });
            using ModelSharp.Pipeline.Pipeline pipeline = BuildPipeline(vocab, merges, engine);

            string text = pipeline.Generate("a", new GenerationConfig { MaxNewTokens = 3, DoSample = false });
            Assert.Equal("dog", text);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void GenerateStream_Yields_Incremental_Fragments_Matching_Generate()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var config = new GenerationConfig { MaxNewTokens = 3, DoSample = false };

            using ModelSharp.Pipeline.Pipeline p1 = BuildPipeline(vocab, merges, new ScriptedEngine(Letters.Length, new[] { 2, 0, 1 }));
            string whole = p1.Generate("a", config);

            using ModelSharp.Pipeline.Pipeline p2 = BuildPipeline(vocab, merges, new ScriptedEngine(Letters.Length, new[] { 2, 0, 1 }));
            string streamed = string.Concat(p2.GenerateStream("a", config));

            Assert.Equal("cab", whole);
            Assert.Equal(whole, streamed);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Generate_Is_Reproducible_With_Fixed_Seed()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var config = new GenerationConfig { MaxNewTokens = 8, DoSample = true, Seed = 1234 };

            using ModelSharp.Pipeline.Pipeline a = BuildPipeline(vocab, merges, new UniformEngine(Letters.Length));
            using ModelSharp.Pipeline.Pipeline b = BuildPipeline(vocab, merges, new UniformEngine(Letters.Length));
            string runA = a.Generate("a", config);
            string runB = b.Generate("a", config);
            Assert.Equal(runA, runB);   // same seed -> identical text

            using ModelSharp.Pipeline.Pipeline c = BuildPipeline(vocab, merges, new UniformEngine(Letters.Length));
            string runDifferent = c.Generate("a", config with { Seed = 9999 });
            Assert.NotEqual(runA, runDifferent);   // different seed -> different text
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Generate_On_NonGeneration_Pipeline_Throws_Clear_Error()
    {
        // A single-shot pipeline (built with pre/post) must reject Generate with a clear message.
        var manifest = new ModelManifest { Task = ModelTask.Embedding };
        using var pipeline = new ModelSharp.Pipeline.Pipeline(new UniformEngine(4), manifest, new ThrowPre(), new ThrowPost());

        ModelSharpException ex = Assert.Throws<ModelSharpException>(() => pipeline.Generate("hi"));
        Assert.Contains("TextGeneration", ex.Message);
    }

    private sealed class ThrowPre : IPreprocessor
    {
        public IReadOnlyDictionary<string, NamedTensor> ToFeeds(object input) => throw new NotSupportedException();
    }

    private sealed class ThrowPost : IPostprocessor
    {
        public object Decode(IReadOnlyDictionary<string, NamedTensor> outputs) => throw new NotSupportedException();
    }
}
