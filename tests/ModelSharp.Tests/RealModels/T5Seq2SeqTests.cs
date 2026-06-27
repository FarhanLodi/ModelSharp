using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Generation;
using ModelSharp.Graph;
using ModelSharp.Manifest;
using ModelSharp.Onnx;
using ModelSharp.Pipeline;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Opt-in integration test against a real text seq2seq (encoder-decoder) ONNX export — e.g. a small
/// <c>t5-small</c> / <c>flan-t5-small</c> or a distilled BART/Marian. No-ops (green) unless the model +
/// tokenizer files are present in the resolved models dir; becomes live when a user drops them in.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>seq2seq-encoder.onnx</c>, <c>seq2seq-decoder.onnx</c> (a decoder-with-past or merged export),
/// <c>seq2seq-vocab.json</c>, <c>seq2seq-merges.txt</c>. The decoder-start-token / EOS / KV head dims
/// are model-specific; defaults below match the T5 family (start = pad id 0).</para>
/// </summary>
public class T5Seq2SeqTests
{
    private readonly ITestOutputHelper _out;
    public T5Seq2SeqTests(ITestOutputHelper output) => _out = output;

    private const string EncoderFile = "seq2seq-encoder.onnx";
    private const string DecoderFile = "seq2seq-decoder.onnx";
    private const string VocabFile = "seq2seq-vocab.json";
    private const string MergesFile = "seq2seq-merges.txt";

    // t5-small config: num_heads = 8, d_kv = 64, decoder_start_token_id = 0 (pad), eos = 1.
    private const int NumHeads = 8;
    private const int HeadDim = 64;
    private const int DecoderStartTokenId = 0;
    private const int EosTokenId = 1;

    [Fact]
    public void Seq2Seq_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(EncoderFile, out string encPath)
            || !RealModelAssets.TryPath(DecoderFile, out string decPath))
        {
            _out.WriteLine("seq2seq model not present; skipping.");
            return;
        }

        KernelRegistry registry = KernelRegistry.CreateDefault();
        foreach (string path in new[] { encPath, decPath })
        {
            ModelGraph g = OnnxModelLoader.LoadModel(path);
            var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
            var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();
            _out.WriteLine($"{path}: nodes={g.Nodes.Count} distinctOps={distinct.Count}");
            _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));
            Assert.True(missing.Count == 0, $"Unsupported ops in {path}: " + string.Join(", ", missing));
        }
    }

    [Fact]
    public void Seq2Seq_Greedy_Decode_Is_Deterministic()
    {
        if (!RealModelAssets.TryPath(EncoderFile, out string encPath)
            || !RealModelAssets.TryPath(DecoderFile, out string decPath)
            || !RealModelAssets.TryPath(VocabFile, out string vocabPath)
            || !RealModelAssets.TryPath(MergesFile, out string mergesPath))
        {
            _out.WriteLine("seq2seq assets not present; skipping.");
            return;
        }

        var manifest = new ModelManifest
        {
            Task = ModelTask.Seq2SeqGeneration,
            Extra = new Dictionary<string, string>
            {
                ["vocab"] = vocabPath,
                ["merges"] = mergesPath,
                ["kv_num_heads"] = NumHeads.ToString(),
                ["kv_head_dim"] = HeadDim.ToString(),
                ["decoder_start_token_id"] = DecoderStartTokenId.ToString(),
                ["eos_token_id"] = EosTokenId.ToString(),
                ["max_new_tokens"] = "16",
            },
        };

        using ModelSharp.Pipeline.Pipeline pipeline = ModelSharpPipeline.LoadSeq2Seq(encPath, decPath, manifest);

        const string source = "translate English to German: The house is wonderful.";
        var greedy = GenerationConfig.Greedy(maxNewTokens: 16, eosTokenIds: new[] { EosTokenId });

        string first = pipeline.Generate(source, greedy);
        _out.WriteLine($"source: {source}");
        _out.WriteLine($"greedy output: {first}");

        Assert.False(string.IsNullOrEmpty(first), "Greedy seq2seq generation produced no text.");

        // Greedy argmax decoding is deterministic: the same source + config must reproduce exactly.
        using ModelSharp.Pipeline.Pipeline pipeline2 = ModelSharpPipeline.LoadSeq2Seq(encPath, decPath, manifest);
        string second = pipeline2.Generate(source, greedy);
        Assert.Equal(first, second);
    }
}
