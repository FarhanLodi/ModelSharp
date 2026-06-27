using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Engine;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Qwen2.5-0.5B-Instruct, INT4 (transformers.js / Optimum q4 export). A fast, real INT4 LLM: 24 layers,
/// 14 query / 2 KV heads, head_dim 64, vocab 151936. Uses <c>MatMulNBits</c> (168×) with DECOMPOSED standard
/// attention (Softmax/Concat/Expand/Where/Range/Trilu) — so it exercises the INT4 weight path end-to-end on
/// the ops ModelSharp already has (it does NOT use the GroupQueryAttention contrib op; Mistral-7B does).
/// Asset-gated: skips when the model isn't present.
/// </summary>
public class Qwen05bInt4Tests
{
    private readonly ITestOutputHelper _out;
    public Qwen05bInt4Tests(ITestOutputHelper output) => _out = output;

    private const string ModelRel = "qwen05b-q4/model_q4.onnx";
    private const string HubSpec = "onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx";
    private const int KvHeads = 2, HeadDim = 64, Vocab = 151936;

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using ILGPU.Context ctx = ILGPU.Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != ILGPU.Runtime.AcceleratorType.CPU);
        }
        catch { return false; }
    }

    [Fact]
    public void Qwen05b_Int4_Loads_And_FullOpCoverage()
    {
        if (!RealModelAssets.TryResolveOrDownload(ModelRel, HubSpec, out string path, log: _out.WriteLine)) { _out.WriteLine("Qwen-0.5B not found; skipping."); return; }

        ModelGraph g = OnnxModelLoader.LoadModel(path);
        _out.WriteLine($"Loaded Qwen-0.5B q4: {g.Nodes.Count} nodes, {g.Initializers.Count} initializers.");

        KernelRegistry reg = KernelRegistry.CreateDefault();
        List<string> distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
        List<string> missing = distinct.Where(op => !reg.TryGet(op, out _)).ToList();
        _out.WriteLine($"distinct op types ({distinct.Count}): {string.Join(", ", distinct)}");
        if (missing.Count > 0) _out.WriteLine($"MISSING: {string.Join(", ", missing)}");

        Assert.Empty(missing);
        Assert.True(g.Nodes.Count(n => n.OpType == "MatMulNBits") > 0, "expected MatMulNBits (INT4) nodes");
    }

    [Fact]
    public void Qwen05b_Int4_ForwardPass_ProducesCoherentLogits()
    {
        if (!RealModelAssets.TryResolveOrDownload(ModelRel, HubSpec, out string path, log: _out.WriteLine)) { _out.WriteLine("Qwen-0.5B not found; skipping."); return; }

        ModelGraph g = OnnxModelLoader.LoadModel(path);
        long[] tokenIds = { 9707, 11, 1879, 0 }; // arbitrary valid ids
        Dictionary<string, NamedTensor> feeds = PrefillFeeds(g.Inputs, tokenIds);

        bool useGpu = HardwareGpuAvailable();
        _out.WriteLine($"Running Qwen-0.5B INT4 forward pass on {(useGpu ? "GPU" : "CPU")} ...");
        try
        {
            using IExecutionEngine eng = useGpu ? new IlgpuEngine(g, preferCpu: false) : new ManagedCpuEngine(g);
            IReadOnlyDictionary<string, NamedTensor> outp = eng.Run(feeds);
            Tensor<float> logits = outp["logits"].Data;

            int[] dims = logits.Shape.Dimensions.ToArray();
            int vocab = dims[^1], seq = dims[^2];
            _out.WriteLine($"logits shape: [{string.Join(",", dims)}]");
            Assert.Equal(Vocab, vocab);
            Assert.Equal(tokenIds.Length, seq);

            float[] all = logits.Buffer.ToArray();
            Assert.All(all, v => Assert.True(float.IsFinite(v)));

            int baseOff = (seq - 1) * vocab;
            int best = 0; float bestVal = all[baseOff];
            for (int v = 1; v < vocab; v++)
                if (all[baseOff + v] > bestVal) { bestVal = all[baseOff + v]; best = v; }
            float mean = all.Skip(baseOff).Take(vocab).Average();
            _out.WriteLine($"next-token argmax id={best} logit={bestVal:F3} (mean {mean:F3})");

            Assert.InRange(best, 0, Vocab - 1);
            Assert.True(bestVal > mean + 1f, $"top logit {bestVal} should exceed mean {mean} (degenerate output?)");
        }
        catch (Exception ex) when (ex.Message.Contains("out of memory"))
        {
            _out.WriteLine($"GPU out of memory (co-tenant); skipping. [{ex.Message}]");
        }
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
                throw new InvalidOperationException($"Unexpected Qwen input '{name}'.");
        }
        return feeds;
    }
}
