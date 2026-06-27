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
/// Opt-in integration test against a real BART-family encoder/decoder ONNX export
/// (<c>sshleifer/distilbart-cnn-6-6</c>, Xenova/optimum form) — a summarization seq2seq that uses a
/// byte-level BPE tokenizer (vocab.json + merges.txt), which is exactly what
/// <see cref="ModelSharpPipeline.LoadSeq2Seq"/> expects (unlike T5, which is SentencePiece and would
/// need a separate tokenizer). No-ops (green) unless the model + tokenizer files are present.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>bart-encoder.onnx</c>, <c>bart-decoder.onnx</c>, <c>bart-vocab.json</c>, <c>bart-merges.txt</c>.
/// Source: https://huggingface.co/Xenova/distilbart-cnn-6-6 (onnx/encoder_model.onnx,
/// onnx/decoder_model.onnx, vocab.json, merges.txt). BART config: 16 decoder heads, d_model 1024
/// (head_dim 64), decoder_start_token_id = 2 (&lt;/s&gt;), eos = 2, forced_bos = 0 (&lt;s&gt;).</para>
///
/// <para>NOTE — known gap: BART forces <c>&lt;s&gt;</c> (id 0) as the first <i>generated</i> token via
/// <c>forced_bos_token_id</c>; the current <see cref="Seq2SeqGenerationProcessor"/> exposes
/// <c>decoder_start_token_id</c> but no <c>forced_bos_token_id</c>, so the first step is pure greedy.
/// In practice distilbart still emits <c>&lt;s&gt;</c> first, so greedy output remains sensible; this test
/// asserts non-empty + deterministic text rather than an exact reference string.</para>
/// </summary>
public class BartSeq2SeqTests
{
    private readonly ITestOutputHelper _out;
    public BartSeq2SeqTests(ITestOutputHelper output) => _out = output;

    private const string EncoderFile = "bart-encoder.onnx";
    private const string DecoderFile = "bart-decoder.onnx";
    private const string VocabFile = "bart-vocab.json";
    private const string MergesFile = "bart-merges.txt";

    // distilbart-cnn-6-6: decoder_attention_heads = 16, d_model = 1024 -> head_dim = 64.
    private const int NumHeads = 16;
    private const int HeadDim = 64;
    private const int DecoderStartTokenId = 2; // </s>
    private const int EosTokenId = 2;          // </s>

    [Fact]
    public void Bart_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(EncoderFile, out string encPath)
            || !RealModelAssets.TryPath(DecoderFile, out string decPath))
        {
            _out.WriteLine("BART model not present; skipping.");
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
    public void Bart_Greedy_Summary_Is_NonEmpty_AndDeterministic()
    {
        if (!RealModelAssets.TryPath(EncoderFile, out string encPath)
            || !RealModelAssets.TryPath(DecoderFile, out string decPath)
            || !RealModelAssets.TryPath(VocabFile, out string vocabPath)
            || !RealModelAssets.TryPath(MergesFile, out string mergesPath))
        {
            _out.WriteLine("BART assets not present; skipping.");
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
                ["max_new_tokens"] = "24",
            },
        };

        const string source =
            "The James Webb Space Telescope is the largest optical telescope in space. " +
            "It was launched in December 2021 and observes the universe in infrared light, " +
            "letting astronomers see some of the earliest galaxies that formed after the Big Bang.";

        var greedy = GenerationConfig.Greedy(maxNewTokens: 24, eosTokenIds: new[] { EosTokenId });

        using ModelSharp.Pipeline.Pipeline pipeline = ModelSharpPipeline.LoadSeq2Seq(encPath, decPath, manifest);
        string first = pipeline.Generate(source, greedy);
        _out.WriteLine($"source: {source}");
        _out.WriteLine($"summary: '{first}'");

        Assert.False(string.IsNullOrWhiteSpace(first), "Greedy BART generation produced no text.");
        Assert.Contains(first, c => char.IsLetter(c));

        // Greedy argmax decoding is deterministic: same source + config reproduces exactly.
        using ModelSharp.Pipeline.Pipeline pipeline2 = ModelSharpPipeline.LoadSeq2Seq(encPath, decPath, manifest);
        string second = pipeline2.Generate(source, greedy);
        Assert.Equal(first, second);
    }
}
