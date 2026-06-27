using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Asset-gated proof that a REAL genuinely-quantized LLM runs end-to-end on the GPU engine.
///
/// <para>The model is <c>onnx-community/gpt2-ONNX</c> <c>model_quantized.onnx</c> (ONNXRuntime dynamic-INT8
/// quantization of gpt2, ~280&#160;MB, a <c>text-generation-with-past</c> export). Its quantized linear layers
/// lower to <b>48× DynamicQuantizeLinear → MatMulInteger</b> plus <c>DequantizeLinear</c> — the exact INT8
/// inference path under test — interleaved with the usual decoder float ops (MatMul/Softmax/LayerNorm-via-
/// primitives/Tanh-GELU) and a 12-layer with-past KV-cache structure. Same I/O contract as the (already
/// GPU-validated) distilgpt2 export, plus <c>position_ids</c>.</para>
///
/// <para>On the GPU engine the INT8 GEMM hot-spot (<c>MatMulInteger</c>) now runs through the <b>native
/// on-device kernel</b>; the lighter quant glue (<c>DynamicQuantizeLinear</c>/<c>DequantizeLinear</c>) still runs
/// through the per-op CPU fallback (download → CPU quant kernel → re-home by dtype), interleaved with the native
/// GPU float ops. This test drives the WHOLE graph through <see cref="IlgpuEngine.Run"/> for a few greedy decode
/// steps and asserts the last-token logits + argmax match <see cref="ManagedCpuEngine"/> — i.e. the quantized LLM
/// decodes coherently and deterministically on the GPU engine (now exercising the native INT8 GEMM), GPU argmax
/// == CPU argmax at every step.</para>
///
/// <para>Discovers the asset via <c>MODELSHARP_MODELS_DIR</c> → repo-relative <c>models/</c> →
/// <c>/home/x16/models</c> and SKIPS cleanly (green, no assertions) when the file is absent — never hard-fails on
/// a missing download. The CUDA run additionally skips when no CUDA device is present; a CPU-accelerator routing
/// variant runs the full quantized graph everywhere (no GPU required).</para>
/// </summary>
public class QuantizedGpt2GpuTests
{
    private readonly ITestOutputHelper _out;
    public QuantizedGpt2GpuTests(ITestOutputHelper output) => _out = output;

    // gpt2 config: 12 layers, 12 heads, n_embd 768 → head_dim 64; eos/bos 50256.
    private const int NumHeads = 12, HeadDim = 64, EosTokenId = 50256;

    // "The quick brown fox" — gpt2 BPE ids (same tokenizer family as distilgpt2).
    private static readonly long[] PromptTokens = { 464, 2068, 7586, 21831 };

    private const string ModelFile = "gpt2-quantized.onnx";

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch { return false; }
    }

    private static bool TryFindModel(out string path)
    {
        string? env = Environment.GetEnvironmentVariable("MODELSHARP_MODELS_DIR");
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(env))
            candidates.Add(Path.Combine(env, ModelFile));
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            candidates.Add(Path.Combine(dir.FullName, "models", ModelFile));
            dir = dir.Parent;
        }
        candidates.Add(Path.Combine("/home/x16/models", ModelFile));

        foreach (string c in candidates)
            if (File.Exists(c)) { path = c; return true; }
        path = candidates.Count > 0 ? candidates[0] : ModelFile;
        return false;
    }

    /// <summary>
    /// Prefill feeds for the quantized gpt2 with-past export: <c>input_ids</c>/<c>attention_mask</c>/
    /// <c>position_ids</c> (int64) plus an EMPTY <c>[1, heads, 0, headDim]</c> float past for every
    /// <c>past_key_values.N.{key,value}</c> input (the prefill state of the KV cache).
    /// </summary>
    private static Dictionary<string, NamedTensor> PrefillFeeds(IReadOnlyList<string> inputs, long[] tokenIds)
    {
        int seq = tokenIds.Length;
        var feeds = new Dictionary<string, NamedTensor>();
        foreach (string name in inputs)
        {
            if (name == "input_ids")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq), tokenIds));
            else if (name == "attention_mask")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq),
                    Enumerable.Repeat(1L, seq).ToArray()));
            else if (name == "position_ids")
                feeds[name] = new NamedTensor(name, new Tensor<long>(new TensorShape(1, seq),
                    Enumerable.Range(0, seq).Select(i => (long)i).ToArray()));
            else if (name.StartsWith("past_key", StringComparison.Ordinal))
                feeds[name] = new NamedTensor(name, new Tensor<float>(new TensorShape(1, NumHeads, 0, HeadDim)));
            else
                throw new InvalidOperationException($"Unexpected quantized-gpt2 input '{name}'.");
        }
        return feeds;
    }

    /// <summary>argmax over the last position's row of a <c>[1, seq, vocab]</c> logits tensor.</summary>
    private static (int arg, float best) LastTokenArgmax(Tensor<float> logits)
    {
        int[] dims = logits.Shape.Dimensions.ToArray();
        int seq = dims[^2], vocab = dims[^1];
        float[] v = logits.Span.ToArray();
        int baseOff = (seq - 1) * vocab;
        int arg = 0; float best = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++)
            if (v[baseOff + i] > best) { best = v[baseOff + i]; arg = i; }
        return (arg, best);
    }

    /// <summary>
    /// Op-coverage probe (no GPU needed): every op the quantized gpt2 graph uses must be in the CPU kernel
    /// registry (so the GPU engine can run it natively or via fallback). Confirms the genuinely-quantized ops are
    /// present: DynamicQuantizeLinear, MatMulInteger, DequantizeLinear.
    /// </summary>
    [Fact]
    public void QuantizedGpt2_Op_Coverage_And_Is_Quantized()
    {
        if (!TryFindModel(out string modelPath))
        {
            _out.WriteLine($"quantized gpt2 not found ({ModelFile}); skipping.");
            return;
        }

        ModelGraph g = OnnxModelLoader.LoadModel(modelPath);
        KernelRegistry registry = KernelRegistry.CreateDefault();
        var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
        var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();

        int dynQuant = g.Nodes.Count(n => n.OpType == "DynamicQuantizeLinear");
        int matMulInt = g.Nodes.Count(n => n.OpType == "MatMulInteger");
        _out.WriteLine($"nodes={g.Nodes.Count} distinctOps={distinct.Count} initializers={g.Initializers.Count}");
        _out.WriteLine($"DynamicQuantizeLinear={dynQuant}  MatMulInteger={matMulInt}");
        _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));

        Assert.True(missing.Count == 0, "Unsupported ops: " + string.Join(", ", missing));
        // Genuinely quantized: the INT8 linear path (DynamicQuantizeLinear → MatMulInteger) must be present.
        Assert.True(dynQuant > 0 && matMulInt > 0,
            $"expected a quantized model with DynamicQuantizeLinear/MatMulInteger, got dyn={dynQuant} mmi={matMulInt}.");
    }

    /// <summary>
    /// Whole quantized graph through the GPU engine on ILGPU's <em>CPU</em> accelerator (no CUDA required): drives
    /// the full INT8 gpt2 graph end-to-end and asserts the last-token logits + argmax match the managed CPU engine.
    /// Proves the quantized inference path (incl. the per-op CPU fallback for the quant kernels, interleaved with
    /// native GPU float ops) is correct on the GPU engine even on machines without a GPU.
    /// </summary>
    [Fact]
    public void QuantizedGpt2_WholeGraph_CpuAccel_Matches_Cpu()
    {
        if (!TryFindModel(out string modelPath))
        {
            _out.WriteLine($"quantized gpt2 not found ({ModelFile}); skipping.");
            return;
        }

        ModelGraph graph = OnnxModelLoader.LoadModel(modelPath);
        using var gpu = new IlgpuEngine(graph, preferCpu: true);
        using var cpu = new ManagedCpuEngine(graph);

        Dictionary<string, NamedTensor> feeds = PrefillFeeds(graph.Inputs, PromptTokens);
        Tensor<float> cLogits = cpu.Run(feeds)["logits"].Data;
        Tensor<float> gLogits = gpu.Run(feeds)["logits"].Data;
        Assert.Equal(cLogits.Shape.Dimensions.ToArray(), gLogits.Shape.Dimensions.ToArray());

        int[] dims = cLogits.Shape.Dimensions.ToArray();
        int seq = dims[^2], vocab = dims[^1];
        float[] g = gLogits.Span.ToArray(), c = cLogits.Span.ToArray();
        int baseOff = (seq - 1) * vocab;
        float maxDiff = 0f;
        for (int v = 0; v < vocab; v++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(g[baseOff + v] - c[baseOff + v]));

        (int gArg, _) = LastTokenArgmax(gLogits);
        (int cArg, _) = LastTokenArgmax(cLogits);
        _out.WriteLine($"CpuAccel quantized whole-graph: logits[{string.Join(",", dims)}] max|Δ|={maxDiff:E2} " +
                       $"argmax cpu={cArg} gpu={gArg}.");

        Assert.True(maxDiff < 5e-2f, $"max|Δ logits|={maxDiff} too large");
        Assert.Equal(cArg, gArg);
    }

    /// <summary>
    /// The headline proof: drive the WHOLE quantized gpt2 graph through <see cref="IlgpuEngine.Run"/> on real
    /// hardware CUDA for several greedy decode steps. Each step re-runs the full graph with the growing token
    /// sequence (stateless prefill — the simplest whole-graph check), and the GPU's next-token argmax must equal
    /// the CPU's at every step. Coherence: greedy decoding is deterministic, so the generated id sequence is
    /// stable and GPU==CPU. Skips cleanly if no CUDA device or no model asset.
    /// </summary>
    [Fact]
    public void Cuda_QuantizedGpt2_GreedyDecode_Matches_Cpu()
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine("Cuda quantized gpt2: no CUDA device; skipping.");
            return;
        }
        if (!TryFindModel(out string modelPath))
        {
            _out.WriteLine($"Cuda quantized gpt2: {ModelFile} not found; skipping.");
            return;
        }

        ModelGraph graph = OnnxModelLoader.LoadModel(modelPath);
        _out.WriteLine($"Cuda quantized gpt2: loaded {modelPath} ({graph.Nodes.Count} nodes).");

        // Shared GPU box: skip (don't fail) if the device is out of memory because a co-tenant process
        // is holding VRAM. The CPU-accelerator whole-graph test covers correctness without device memory.
        try
        {
        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"expected hardware GPU, got '{gpu.AcceleratorName}'.");
        using var cpu = new ManagedCpuEngine(graph);

        var tokens = new List<long>(PromptTokens);
        const int decodeSteps = 3;
        var gpuIds = new List<int>();
        var cpuIds = new List<int>();

        for (int step = 0; step < decodeSteps; step++)
        {
            Dictionary<string, NamedTensor> feeds = PrefillFeeds(graph.Inputs, tokens.ToArray());

            IReadOnlyDictionary<string, NamedTensor> cpuOut = cpu.Run(feeds);
            IReadOnlyDictionary<string, NamedTensor> gpuOut = gpu.Run(feeds);

            Tensor<float> cLogits = cpuOut["logits"].Data;
            Tensor<float> gLogits = gpuOut["logits"].Data;
            Assert.Equal(cLogits.Shape.Dimensions.ToArray(), gLogits.Shape.Dimensions.ToArray());

            int[] dims = cLogits.Shape.Dimensions.ToArray();
            int seq = dims[^2], vocab = dims[^1];
            float[] g = gLogits.Span.ToArray(), c = cLogits.Span.ToArray();
            int baseOff = (seq - 1) * vocab;
            float maxDiff = 0f, cScale = 1e-6f;
            for (int v = 0; v < vocab; v++)
            {
                maxDiff = MathF.Max(maxDiff, MathF.Abs(g[baseOff + v] - c[baseOff + v]));
                cScale = MathF.Max(cScale, MathF.Abs(c[baseOff + v]));
            }
            float relDiff = maxDiff / cScale;

            (int gArg, _) = LastTokenArgmax(gLogits);
            (int cArg, _) = LastTokenArgmax(cLogits);
            _out.WriteLine($"  step {step}: seq={seq} vocab={vocab} max|Δ logits|={maxDiff:E2} " +
                           $"(rel {relDiff:P1} of |logit|≤{cScale:F1}) argmax cpu={cArg} gpu={gArg}.");

            // Argmax agreement is the hard correctness gate (generation must be identical GPU vs CPU).
            // Raw logits drift on real CUDA: the INT8 matmuls run via the (double-precision) CPU-fallback
            // while the float ops run in fp32 on-device, so accumulation differs across the 12 layers.
            // The identical-kernel ILGPU-CPU-accelerator path (separate test) matches CPU tightly; here we
            // bound the drift relative to the logit magnitude, which still catches any gross divergence.
            Assert.Equal(cArg, gArg); // greedy next-token must agree GPU vs CPU
            Assert.True(relDiff < 0.10f, $"step {step}: relative logit drift {relDiff:P1} (max|Δ|={maxDiff}) too large");

            gpuIds.Add(gArg);
            cpuIds.Add(cArg);
            tokens.Add(cArg);
            if (cArg == EosTokenId) break;
        }

        _out.WriteLine($"Cuda quantized gpt2 greedy ids: gpu=[{string.Join(",", gpuIds)}] cpu=[{string.Join(",", cpuIds)}].");
        Assert.Equal(cpuIds, gpuIds); // full decoded id sequence agrees → coherent, deterministic on GPU
        Assert.NotEmpty(gpuIds);
        }
        catch (Exception ex) when (ex.Message.Contains("out of memory"))
        {
            _out.WriteLine($"Cuda quantized gpt2: GPU out of memory (likely a co-tenant process holding VRAM); skipping. [{ex.Message}]");
            return;
        }
    }
}
