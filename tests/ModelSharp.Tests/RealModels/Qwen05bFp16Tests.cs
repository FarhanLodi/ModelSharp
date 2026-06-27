using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Qwen2.5-0.5B-Instruct, full <b>fp16</b> ONNX export. Proves the fp16 storage path: the half-precision
/// weights are kept as compact <see cref="System.Half"/> tensors (half the memory of float32) and widened
/// to float on demand at the compute boundary — so a large fp16 (non-quantized) model loads without the
/// 2× memory blow-up of decoding every weight to float32 at load. Asset-gated; skips when absent.
/// </summary>
public class Qwen05bFp16Tests
{
    private readonly ITestOutputHelper _out;
    public Qwen05bFp16Tests(ITestOutputHelper output) => _out = output;

    private const string ModelRel = "qwen05b-q4/model_fp16.onnx";
    private const int KvHeads = 2, HeadDim = 64, Vocab = 151936;

    [Fact]
    public void Qwen05b_Fp16_StoredCompactly_And_RunsCoherently()
    {
        if (!RealModelAssets.TryPath(ModelRel, out string path)) { _out.WriteLine("Qwen fp16 not found; skipping."); return; }

        ModelGraph g = OnnxModelLoader.LoadModel(path);

        // Memory win: the fp16 weights must be stored as Half, not upcast to float32 at load.
        long fp16Tensors = g.Initializers.Values.Count(t => t.Dtype == ElementType.Float16);
        long fp16Elems = g.Initializers.Values.Where(t => t.Dtype == ElementType.Float16).Sum(t => t.Length);
        _out.WriteLine($"Loaded Qwen-0.5B fp16: {g.Nodes.Count} nodes, {g.Initializers.Count} initializers, " +
                       $"{fp16Tensors} stored as fp16 ({fp16Elems * 2 / (1024 * 1024)} MB at fp16, " +
                       $"vs {fp16Elems * 4 / (1024 * 1024)} MB if upcast to fp32).");
        Assert.True(fp16Tensors > 0, "expected fp16 weights to be stored as Half (the memory win)");

        // And it must actually run end-to-end (fp16 weights widened to float at the compute boundary).
        long[] tokenIds = { 9707, 11, 1879, 0 };
        Dictionary<string, NamedTensor> feeds = PrefillFeeds(g.Inputs, tokenIds);
        using var eng = new ManagedCpuEngine(g);
        Tensor<float> logits = eng.Run(feeds)["logits"].Data;

        int[] dims = logits.Shape.Dimensions.ToArray();
        int vocab = dims[^1], seq = dims[^2];
        Assert.Equal(Vocab, vocab);
        float[] all = logits.Buffer.ToArray();
        Assert.All(all, v => Assert.True(float.IsFinite(v)));

        int baseOff = (seq - 1) * vocab;
        int best = 0; float bestVal = all[baseOff];
        for (int v = 1; v < vocab; v++) if (all[baseOff + v] > bestVal) { bestVal = all[baseOff + v]; best = v; }
        float mean = all.Skip(baseOff).Take(vocab).Average();
        _out.WriteLine($"fp16 next-token argmax id={best} logit={bestVal:F3} (mean {mean:F3})");
        Assert.True(bestVal > mean + 1f, "coherent (non-degenerate) next-token distribution");
    }

    private static Dictionary<string, NamedTensor> PrefillFeeds(IReadOnlyList<string> inputs, long[] tokenIds)
    {
        int seq = tokenIds.Length;
        var feeds = new Dictionary<string, NamedTensor>();
        foreach (string name in inputs)
        {
            if (name == "input_ids")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq), tokenIds));
            else if (name == "attention_mask")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq), Enumerable.Repeat(1L, seq).ToArray()));
            else if (name == "position_ids")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq),
                    Enumerable.Range(0, seq).Select(i => (long)i).ToArray()));
            else if (name.StartsWith("past_key_values", StringComparison.Ordinal))
                feeds[name] = new NamedTensor(name, new Tensor<float>(new TensorShape(1, KvHeads, 0, HeadDim)));
            else
                throw new InvalidOperationException($"Unexpected Qwen fp16 input '{name}'.");
        }
        return feeds;
    }
}
