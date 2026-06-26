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
/// A1 — Opt-in integration test against a real <c>distilgpt2</c> ONNX decoder-with-past export.
/// No-ops (green) unless the model + tokenizer files are present in the resolved models dir; becomes
/// live when a user drops them in. See <c>docs/REAL_MODELS.md</c> for the export recipe.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>distilgpt2.onnx</c> (a NON-merged <c>text-generation-with-past</c> export),
/// <c>distilgpt2-vocab.json</c>, <c>distilgpt2-merges.txt</c>.</para>
/// </summary>
public class DistilGpt2GenerationTests
{
    private readonly ITestOutputHelper _out;
    public DistilGpt2GenerationTests(ITestOutputHelper output) => _out = output;

    private const string ModelFile = "distilgpt2.onnx";
    private const string VocabFile = "distilgpt2-vocab.json";
    private const string MergesFile = "distilgpt2-merges.txt";

    // distilgpt2 config.json: n_head = 12, n_embd = 768 -> head_dim = 64; eos/bos = 50256.
    private const int NumHeads = 12;
    private const int HeadDim = 64;
    private const int EosTokenId = 50256;

    [Fact]
    public void DistilGpt2_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath))
        {
            _out.WriteLine($"distilgpt2 model not present ({modelPath}); skipping.");
            return;
        }

        ModelGraph g = OnnxModelLoader.LoadModel(modelPath);
        KernelRegistry registry = KernelRegistry.CreateDefault();
        var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
        var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();

        _out.WriteLine($"nodes={g.Nodes.Count}  distinctOps={distinct.Count}  initializers={g.Initializers.Count}");
        _out.WriteLine("ALL OPS: " + string.Join(", ", distinct));
        _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));
        Assert.True(missing.Count == 0, "Unsupported ops: " + string.Join(", ", missing));
    }

    [Fact]
    public void DistilGpt2_Greedy_Decode_Is_Deterministic()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath)
            || !RealModelAssets.TryPath(VocabFile, out string vocabPath)
            || !RealModelAssets.TryPath(MergesFile, out string mergesPath))
        {
            _out.WriteLine("distilgpt2 assets not present; skipping.");
            return;
        }

        // Drive the headline one-line text-generation API: a TextGeneration manifest carrying the
        // tokenizer files and the KV-cache head dims (the CPU engine reports no concrete past-KV
        // shapes, so the canonical [batch, heads, seq, head_dim] layout is sized from these).
        var manifest = new ModelManifest
        {
            Task = ModelTask.TextGeneration,
            Extra = new Dictionary<string, string>
            {
                ["vocab"] = vocabPath,
                ["merges"] = mergesPath,
                ["kv_num_heads"] = NumHeads.ToString(),
                ["kv_head_dim"] = HeadDim.ToString(),
                ["eos_token_id"] = EosTokenId.ToString(),
                ["max_new_tokens"] = "16",
            },
        };

        using ModelSharp.Pipeline.Pipeline pipeline = ModelSharp.Pipeline.Pipeline.Load(modelPath, manifest);

        const string prompt = "The quick brown fox";
        var greedy = GenerationConfig.Greedy(maxNewTokens: 16, eosTokenIds: new[] { EosTokenId });

        string first = pipeline.Generate(prompt, greedy);
        _out.WriteLine($"prompt: {prompt}");
        _out.WriteLine($"greedy continuation: {first}");

        Assert.False(string.IsNullOrEmpty(first), "Greedy generation produced no text.");

        // Greedy argmax decoding is deterministic: the same prompt + config must reproduce exactly.
        using ModelSharp.Pipeline.Pipeline pipeline2 = ModelSharp.Pipeline.Pipeline.Load(modelPath, manifest);
        string second = pipeline2.Generate(prompt, greedy);
        Assert.Equal(first, second);
    }
}
