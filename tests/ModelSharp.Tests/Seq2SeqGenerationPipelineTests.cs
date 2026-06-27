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
/// Covers the high-level seq2seq wiring — <see cref="Seq2SeqGenerationProcessor"/> (BPE tokenizer +
/// <see cref="Seq2SeqModelOptions"/> from the manifest) and the seq2seq-flavoured <see cref="Pipeline"/>
/// (<see cref="Pipeline.Generate"/> / <see cref="Pipeline.GenerateStream"/>). No real model is
/// downloaded: a tiny synthetic byte-level vocab and scripted encoder/decoder engines make the path
/// deterministic and asset-free.
/// </summary>
public class Seq2SeqGenerationPipelineTests
{
    private const string Letters = "abcdefghijklmnopqrstuvwxyz";

    /// <summary>Writes a tiny vocab.json (single-letter tokens) + empty merges.txt and returns the dir.</summary>
    private static (string dir, string vocab, string merges) WriteTinyBpeAssets()
    {
        string dir = Path.Combine(Path.GetTempPath(), "modelsharp-s2s-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        var entries = new List<string>();
        for (int i = 0; i < Letters.Length; i++)
            entries.Add($"  \"{Letters[i]}\": {i}");
        string vocabJson = "{\n" + string.Join(",\n", entries) + "\n}\n";

        string vocab = Path.Combine(dir, "vocab.json");
        string merges = Path.Combine(dir, "merges.txt");
        File.WriteAllText(vocab, vocabJson);
        File.WriteAllText(merges, "#version: 0.2\n");
        return (dir, vocab, merges);
    }

    /// <summary>A trivial encoder returning [1, srcLen, hidden] zeros; records that it ran.</summary>
    private sealed class StubEncoder : IExecutionEngine
    {
        public int Calls { get; private set; }
        public IReadOnlyList<TensorInfo> Inputs { get; } = new[]
        {
            new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()),
            new TensorInfo("attention_mask", ElementType.Int64, Array.Empty<int>()),
        };
        public IReadOnlyList<TensorInfo> Outputs { get; } = new[]
        {
            new TensorInfo("last_hidden_state", ElementType.Float32, Array.Empty<int>()),
        };

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            Calls++;
            int srcLen = feeds["input_ids"].Tensor.Shape.Dimensions[1];
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["last_hidden_state"] = new NamedTensor("last_hidden_state", new Tensor<float>(new TensorShape(1, srcLen, 4))),
            };
        }

        public void Dispose() { }
    }

    /// <summary>A scripted no-cache decoder emitting one-hot logits whose argmax follows a fixed script.</summary>
    private sealed class ScriptedDecoder : IExecutionEngine
    {
        private readonly int[] _peaks;
        private readonly int _vocab;
        private int _call;

        public ScriptedDecoder(int vocab, int[] peaks) { _vocab = vocab; _peaks = peaks; }

        public IReadOnlyList<TensorInfo> Inputs { get; } = new[]
        {
            new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()),
            new TensorInfo("encoder_hidden_states", ElementType.Float32, Array.Empty<int>()),
            new TensorInfo("encoder_attention_mask", ElementType.Int64, Array.Empty<int>()),
        };
        public IReadOnlyList<TensorInfo> Outputs { get; } = new[]
        {
            new TensorInfo("logits", ElementType.Float32, Array.Empty<int>()),
        };

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            int peak = _peaks[Math.Min(_call, _peaks.Length - 1)];
            _call++;
            var row = new float[_vocab];
            row[peak] = 10f;
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["logits"] = new NamedTensor("logits", Tensor<float>.FromArray(new TensorShape(1, 1, _vocab), row)),
            };
        }

        public void Dispose() { }
    }

    private static ProcessorContext Ctx(string vocab, string merges, IExecutionEngine decoder)
    {
        var manifest = new ModelManifest
        {
            Task = ModelTask.Seq2SeqGeneration,
            Extra = new Dictionary<string, string>
            {
                ["vocab"] = vocab,
                ["merges"] = merges,
                ["add_special_tokens"] = "false",   // keep the synthetic source ids = letters exactly
            },
        };
        return new ProcessorContext(manifest, decoder.Inputs.Select(i => i.Name).ToList(),
            decoder.Outputs.Select(o => o.Name).ToList());
    }

    // ----------------------------------------------------------------------------------------
    // Processor wiring
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void Processor_Builds_Tokenizer_And_RoundTrips_Letters()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var proc = new Seq2SeqGenerationProcessor(Ctx(vocab, merges, new ScriptedDecoder(26, new[] { 0 })));

            IReadOnlyList<long> ids = proc.Encode("cab");
            Assert.Equal(new long[] { 2, 0, 1 }, ids.ToArray());
            Assert.Equal("cab", proc.Decode(ids));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Processor_Reads_Options_And_Config_From_Manifest()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var manifest = new ModelManifest
            {
                Task = ModelTask.Seq2SeqGeneration,
                Extra = new Dictionary<string, string>
                {
                    ["vocab"] = vocab,
                    ["merges"] = merges,
                    ["kv_num_heads"] = "4",
                    ["kv_head_dim"] = "8",
                    ["decoder_start_token_id"] = "3",
                    ["eos_token_id"] = "5",
                    ["max_new_tokens"] = "7",
                },
            };
            var proc = new Seq2SeqGenerationProcessor(
                new ProcessorContext(manifest, new[] { "input_ids" }, new[] { "logits" }));

            Assert.Equal(4, proc.Options.KvCacheNumHeads);
            Assert.Equal(8, proc.Options.KvCacheHeadDim);
            Assert.Equal(3L, proc.Options.DecoderStartTokenId);
            Assert.Equal(7, proc.DefaultConfig.MaxNewTokens);
            Assert.Equal(new[] { 5 }, proc.DefaultConfig.EosTokenIds);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Processor_Throws_When_Tokenizer_Files_Missing()
    {
        var manifest = new ModelManifest { Task = ModelTask.Seq2SeqGeneration };
        Assert.Throws<ModelSharpException>(() => new Seq2SeqGenerationProcessor(
            new ProcessorContext(manifest, new[] { "input_ids" }, new[] { "logits" })));
    }

    // ----------------------------------------------------------------------------------------
    // Pipeline.Generate / GenerateStream
    // ----------------------------------------------------------------------------------------

    private static ModelSharp.Pipeline.Pipeline BuildPipeline(
        string vocab, string merges, IExecutionEngine encoder, IExecutionEngine decoder)
    {
        ProcessorContext ctx = Ctx(vocab, merges, decoder);
        return ModelSharpPipeline.BuildSeq2Seq(encoder, decoder, ctx.Manifest);
    }

    [Fact]
    public void Generate_Decodes_Greedy_Targets_To_Text_And_Runs_Encoder_Once()
    {
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var encoder = new StubEncoder();
            // Greedy target script: "dog" (d=3, o=14, g=6).
            var decoder = new ScriptedDecoder(Letters.Length, new[] { 3, 14, 6 });
            using ModelSharp.Pipeline.Pipeline pipeline = BuildPipeline(vocab, merges, encoder, decoder);

            string text = pipeline.Generate("cat", new GenerationConfig { MaxNewTokens = 3, DoSample = false });

            Assert.Equal("dog", text);
            Assert.Equal(1, encoder.Calls);   // encoder ran exactly once for the whole generation
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

            using ModelSharp.Pipeline.Pipeline p1 = BuildPipeline(
                vocab, merges, new StubEncoder(), new ScriptedDecoder(Letters.Length, new[] { 2, 0, 1 }));
            string whole = p1.Generate("a", config);

            using ModelSharp.Pipeline.Pipeline p2 = BuildPipeline(
                vocab, merges, new StubEncoder(), new ScriptedDecoder(Letters.Length, new[] { 2, 0, 1 }));
            string streamed = string.Concat(p2.GenerateStream("a", config));

            Assert.Equal("cab", whole);
            Assert.Equal(whole, streamed);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Build_From_SingleGraph_Wires_Seq2Seq_Pipeline()
    {
        // When a manifest's task is Seq2SeqGeneration, BuildSeq2Seq accepts the same engine for both roles
        // (the merged single-graph export). Here we just confirm the wiring routes Generate to the seq2seq
        // path using a decoder that also satisfies the encoder's input/output contract.
        (string dir, string vocab, string merges) = WriteTinyBpeAssets();
        try
        {
            var shared = new MergedEngine(Letters.Length, new[] { 3, 14, 6 });
            ProcessorContext ctx = Ctx(vocab, merges, shared);
            using ModelSharp.Pipeline.Pipeline pipeline = ModelSharpPipeline.BuildSeq2Seq(shared, shared, ctx.Manifest);

            string text = pipeline.Generate("a", new GenerationConfig { MaxNewTokens = 3, DoSample = false });
            Assert.Equal("dog", text);
        }
        finally { Directory.Delete(dir, true); }
    }

    /// <summary>A single engine that serves both encoder and decoder roles (merged-graph stand-in).</summary>
    private sealed class MergedEngine : IExecutionEngine
    {
        private readonly int[] _peaks;
        private readonly int _vocab;
        private int _decCall;

        public MergedEngine(int vocab, int[] peaks) { _vocab = vocab; _peaks = peaks; }

        public IReadOnlyList<TensorInfo> Inputs { get; } = new[]
        {
            new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()),
            new TensorInfo("attention_mask", ElementType.Int64, Array.Empty<int>()),
            new TensorInfo("encoder_hidden_states", ElementType.Float32, Array.Empty<int>()),
            new TensorInfo("encoder_attention_mask", ElementType.Int64, Array.Empty<int>()),
        };
        public IReadOnlyList<TensorInfo> Outputs { get; } = new[]
        {
            new TensorInfo("logits", ElementType.Float32, Array.Empty<int>()),
            new TensorInfo("last_hidden_state", ElementType.Float32, Array.Empty<int>()),
        };

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            // Encoder role: no encoder_hidden_states fed -> return last_hidden_state.
            if (!feeds.ContainsKey("encoder_hidden_states"))
            {
                int srcLen = feeds["input_ids"].Tensor.Shape.Dimensions[1];
                return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
                {
                    ["last_hidden_state"] = new NamedTensor("last_hidden_state", new Tensor<float>(new TensorShape(1, srcLen, 4))),
                };
            }

            int peak = _peaks[Math.Min(_decCall, _peaks.Length - 1)];
            _decCall++;
            var row = new float[_vocab];
            row[peak] = 10f;
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["logits"] = new NamedTensor("logits", Tensor<float>.FromArray(new TensorShape(1, 1, _vocab), row)),
            };
        }

        public void Dispose() { }
    }
}
