using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// B5 — hardware-CUDA parity for the new decoder GPU kernels (Reshape/Unsqueeze/Squeeze/Shape/Constant/
/// Expand/Split/Pow/Where/Erf/Gemm) and for a multi-step on-device KV-cache attention path. Mirrors the
/// skip-when-no-CUDA pattern of <see cref="GpuCudaParityTests"/> and shares the serialized
/// <c>CudaGpu</c> collection so only one CUDA context is live at a time.
/// </summary>
[Collection("CudaGpu")]
public class GpuLlmTests
{
    private readonly ITestOutputHelper _out;
    public GpuLlmTests(ITestOutputHelper output) => _out = output;

    private static Tensor<float> T(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<float> Rand(int[] dims, int seed, float lo = -1f, float hi = 1f)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (a, d) => a * d);
        float[] data = Enumerable.Range(0, n).Select(_ => lo + (float)rnd.NextDouble() * (hi - lo)).ToArray();
        return T(dims, data);
    }

    private static Dictionary<string, NamedTensor> Feeds(params (string name, Tensor<float> t)[] feeds) =>
        feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch { return false; }
    }

    /// <summary>Runs a graph on CUDA and CPU and asserts every float output matches to <paramref name="tol"/>.</summary>
    private void AssertCudaMatchesCpu(string what, ModelGraph graph, Dictionary<string, NamedTensor> feeds, float tol = 1e-3f)
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine($"{what}: no CUDA device; skipping.");
            return;
        }

        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"{what}: expected a hardware GPU but got '{gpu.AcceleratorName}'.");
        _out.WriteLine($"{what}: hardware GPU '{gpu.AcceleratorName}' (IsHardwareGpu=true).");

        using var cpu = new ManagedCpuEngine(graph);
        IReadOnlyDictionary<string, NamedTensor> gpuOut = gpu.Run(feeds);
        IReadOnlyDictionary<string, NamedTensor> cpuOut = cpu.Run(feeds);

        foreach (string name in graph.Outputs)
        {
            Tensor<float> g = gpuOut[name].Data;
            Tensor<float> c = cpuOut[name].Data;
            Assert.Equal(c.Shape.Dimensions.ToArray(), g.Shape.Dimensions.ToArray());
            float[] ga = g.Span.ToArray(), ca = c.Span.ToArray();
            Assert.Equal(ca.Length, ga.Length);
            for (int i = 0; i < ca.Length; i++)
                Assert.True(MathF.Abs(ca[i] - ga[i]) < tol, $"{what}:{name}[{i}] cpu={ca[i]} gpu={ga[i]}");
        }
    }

    private static GraphNode N(string op, string name, string[] inputs, string[] outputs,
        Dictionary<string, object>? attrs = null) => new GraphNode(op, name, inputs, outputs, attrs);

    /// <summary>
    /// Local-only discovery of <c>distilgpt2.onnx</c>: <c>MODELSHARP_MODELS_DIR</c> → a repo-relative
    /// <c>models/</c> dir found by walking up from the test output directory. No download (no confident
    /// distilgpt2 ONNX Hub repo); skips cleanly when absent.
    /// </summary>
    private static bool TryFindDistilGpt2(out string path)
    {
        string? env = Environment.GetEnvironmentVariable("MODELSHARP_MODELS_DIR");
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(env))
            candidates.Add(System.IO.Path.Combine(env, "distilgpt2.onnx"));
        var dir = new System.IO.DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            candidates.Add(System.IO.Path.Combine(dir.FullName, "models", "distilgpt2.onnx"));
            dir = dir.Parent;
        }
        foreach (string c in candidates)
            if (System.IO.File.Exists(c)) { path = c; return true; }
        path = candidates.Count > 0 ? candidates[0] : "distilgpt2.onnx";
        return false;
    }

    // ---- Pow / Erf ----

    [Fact]
    public void Cuda_Pow_Broadcast()
        => AssertCudaMatchesCpu("Pow",
            new ModelGraph
            {
                Inputs = new[] { "a", "b" },
                Outputs = new[] { "y" },
                Nodes = new[] { N("Pow", "pow", new[] { "a", "b" }, new[] { "y" }) },
            },
            Feeds(("a", Rand(new[] { 3, 4 }, 11, 0.1f, 3f)), ("b", T(new[] { 1 }, 3f))));

    [Fact]
    public void Cuda_Erf()
        => AssertCudaMatchesCpu("Erf",
            new ModelGraph
            {
                Inputs = new[] { "x" },
                Outputs = new[] { "y" },
                Nodes = new[] { N("Erf", "erf", new[] { "x" }, new[] { "y" }) },
            },
            Feeds(("x", Rand(new[] { 4, 6 }, 12, -3f, 3f))));

    // ---- Where (float value path; bool condition supplied as initializer) ----

    [Fact]
    public void Cuda_Where_Float()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x", "y" },
            Outputs = new[] { "z" },
            Nodes = new[] { N("Where", "w", new[] { "cond", "x", "y" }, new[] { "z" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["cond"] = new Tensor<bool>(new TensorShape(2, 3),
                    new[] { true, false, true, false, true, false }),
            },
        };
        AssertCudaMatchesCpu("Where", graph,
            Feeds(("x", Rand(new[] { 2, 3 }, 13)), ("y", Rand(new[] { 2, 3 }, 14))));
    }

    [Fact]
    public void Cuda_Where_Broadcast_Scalar_Branch()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "z" },
            Nodes = new[] { N("Where", "w", new[] { "cond", "x", "y" }, new[] { "z" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["cond"] = new Tensor<bool>(new TensorShape(2, 2), new[] { true, false, false, true }),
                ["y"] = T(new[] { 1 }, -9f),
            },
        };
        AssertCudaMatchesCpu("WhereScalarBranch", graph, Feeds(("x", Rand(new[] { 2, 2 }, 15))));
    }

    // ---- Reshape / Unsqueeze / Squeeze (float device path) ----

    [Fact]
    public void Cuda_Reshape_Then_MatMul()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x", "w" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                N("Reshape", "r", new[] { "x", "shape" }, new[] { "xr" }),
                N("MatMul", "mm", new[] { "xr", "w" }, new[] { "y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["shape"] = new Tensor<long>(new TensorShape(2), new long[] { 6, 4 }),
            },
        };
        AssertCudaMatchesCpu("ReshapeMatMul", graph,
            Feeds(("x", Rand(new[] { 2, 3, 4 }, 16)), ("w", Rand(new[] { 4, 5 }, 17))));
    }

    [Fact]
    public void Cuda_Unsqueeze_Squeeze_RoundTrip()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                N("Unsqueeze", "u", new[] { "x", "axes" }, new[] { "xu" }),
                N("Relu", "relu", new[] { "xu" }, new[] { "xr" }),
                N("Squeeze", "s", new[] { "xr", "axes" }, new[] { "y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["axes"] = new Tensor<long>(new TensorShape(1), new long[] { 1 }),
            },
        };
        AssertCudaMatchesCpu("UnsqueezeSqueeze", graph, Feeds(("x", Rand(new[] { 3, 5 }, 18, -2f, 2f))));
    }

    // ---- Expand (float device path) ----

    [Fact]
    public void Cuda_Expand_Float()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { N("Expand", "e", new[] { "x", "shape" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["shape"] = new Tensor<long>(new TensorShape(2), new long[] { 3, 4 }),
            },
        };
        AssertCudaMatchesCpu("Expand", graph, Feeds(("x", Rand(new[] { 1, 4 }, 19))));
    }

    // ---- Split (float device path) ----

    [Fact]
    public void Cuda_Split_LastAxis()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "a", "b", "c" },
            Nodes = new[]
            {
                N("Split", "sp", new[] { "x" }, new[] { "a", "b", "c" },
                    new Dictionary<string, object> { ["axis"] = 2L }),
            },
        };
        // 6 along axis 2, even split into 3 of size 2.
        AssertCudaMatchesCpu("Split", graph, Feeds(("x", Rand(new[] { 2, 3, 6 }, 20))));
    }

    // ---- Gemm (transB, bias) ----

    [Fact]
    public void Cuda_Gemm_TransB_Bias()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                N("Gemm", "g", new[] { "a", "w", "bias" }, new[] { "y" },
                    new Dictionary<string, object> { ["transB"] = 1L, ["alpha"] = 1f, ["beta"] = 1f }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["w"] = Rand(new[] { 5, 4 }, 21),     // [N=5, K=4], transB -> K=4,N=5
                ["bias"] = Rand(new[] { 5 }, 22),
            },
        };
        AssertCudaMatchesCpu("GemmTransB", graph, Feeds(("a", Rand(new[] { 3, 4 }, 23))));
    }

    [Fact]
    public void Cuda_Gemm_Plain()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { N("Gemm", "g", new[] { "a", "b" }, new[] { "y" }) },
        };
        AssertCudaMatchesCpu("GemmPlain", graph,
            Feeds(("a", Rand(new[] { 4, 6 }, 24)), ("b", Rand(new[] { 6, 5 }, 25))));
    }

    // ---- A single transformer-style attention block, end to end on GPU ----
    // Builds Q·Kᵀ scaled scores -> softmax -> ·V, plus a residual+gelu MLP, entirely from GPU-dispatched
    // ops (MatMul/Mul/Softmax/Transpose/Add/Pow/Tanh/Reshape). Proves a full block matches the CPU engine.

    [Fact]
    public void Cuda_Attention_Block_Matches_Cpu()
    {
        // Shapes: B=1, H=2, S=4, D=3 head dim. q,k,v are [H,S,D].
        var graph = new ModelGraph
        {
            Inputs = new[] { "q", "k", "v" },
            Outputs = new[] { "ctx" },
            Nodes = new[]
            {
                // scores = q @ kᵀ   (kᵀ over last two axes)
                N("Transpose", "tk", new[] { "k" }, new[] { "kt" },
                    new Dictionary<string, object> { ["perm"] = new long[] { 0, 2, 1 } }),
                N("MatMul", "qk", new[] { "q", "kt" }, new[] { "scores" }),
                // scaled = scores * (1/sqrt(D))
                N("Mul", "scale", new[] { "scores", "inv_sqrt_d" }, new[] { "scaled" }),
                N("Softmax", "sm", new[] { "scaled" }, new[] { "attn" }),
                // ctx = attn @ v
                N("MatMul", "av", new[] { "attn", "v" }, new[] { "ctx" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["inv_sqrt_d"] = T(new[] { 1 }, 1f / MathF.Sqrt(3f)),
            },
        };
        AssertCudaMatchesCpu("AttentionBlock", graph,
            Feeds(("q", Rand(new[] { 2, 4, 3 }, 31)),
                  ("k", Rand(new[] { 2, 4, 3 }, 32)),
                  ("v", Rand(new[] { 2, 4, 3 }, 33))));
    }

    // ---- On-device KV-cache: multi-step decode entirely on GPU vs. CPU full-attention reference ----

    /// <summary>
    /// Builds an [H,S,D] scaled-dot-product self-attention graph and runs it on the CPU engine for the FULL
    /// accumulated K/V sequence up to <paramref name="steps"/> tokens — the reference the incremental on-device
    /// KV-cache must match.
    /// </summary>
    private static ModelGraph FullAttentionGraph(int headDim) => new ModelGraph
    {
        Inputs = new[] { "q", "k", "v" },
        Outputs = new[] { "ctx" },
        Nodes = new[]
        {
            new GraphNode("Transpose", "tk", new[] { "k" }, new[] { "kt" },
                new Dictionary<string, object> { ["perm"] = new long[] { 0, 2, 1 } }),
            new GraphNode("MatMul", "qk", new[] { "q", "kt" }, new[] { "scores" }),
            new GraphNode("Mul", "scale", new[] { "scores", "inv_sqrt_d" }, new[] { "scaled" }),
            new GraphNode("Softmax", "sm", new[] { "scaled" }, new[] { "attn" }),
            new GraphNode("MatMul", "av", new[] { "attn", "v" }, new[] { "ctx" }),
        },
        Initializers = new Dictionary<string, Tensor>
        {
            ["inv_sqrt_d"] = T(new[] { 1 }, 1f / MathF.Sqrt(headDim)),
        },
    };

    [Fact]
    public void Cuda_OnDevice_KvCache_MultiStep_Matches_Cpu()
    {
        const int H = 2, D = 3, maxSeq = 8, steps = 5;
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine("KvCacheMultiStep: no CUDA device; skipping.");
            return;
        }

        // Per-step per-head Q/K/V, each [H, 1, D] (one new token per step).
        var stepQ = new Tensor<float>[steps];
        var stepK = new Tensor<float>[steps];
        var stepV = new Tensor<float>[steps];
        for (int s = 0; s < steps; s++)
        {
            stepQ[s] = Rand(new[] { H, 1, D }, 100 + s);
            stepK[s] = Rand(new[] { H, 1, D }, 200 + s);
            stepV[s] = Rand(new[] { H, 1, D }, 300 + s);
        }

        // Decode graph engine (only used for its DecodeStepAttention seam + cache).
        var decodeGraph = new ModelGraph
        {
            Inputs = new[] { "q", "k", "v" },
            Outputs = new[] { "ctx" },
            Nodes = new[] { new GraphNode("Relu", "noop", new[] { "q" }, new[] { "ctx" }) },
        };
        using var gpu = new IlgpuEngine(decodeGraph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"expected hardware GPU, got '{gpu.AcceleratorName}'.");
        _out.WriteLine($"KvCacheMultiStep: hardware GPU '{gpu.AcceleratorName}' (IsHardwareGpu=true).");

        using GpuKvCache cache = gpu.CreateKvCache(H, maxSeq, D);

        ModelGraph refGraph = FullAttentionGraph(D);
        using var cpu = new ManagedCpuEngine(refGraph);

        // Accumulators for the CPU reference: full K/V sequence per head, grown each step.
        var accK = new List<float[]>(); // per step, [H*D] flattened for that token (head-major)
        var accV = new List<float[]>();

        for (int s = 0; s < steps; s++)
        {
            // GPU: append this step's K/V on-device and attend over the whole cached prefix.
            Tensor<float> gpuCtx = gpu.DecodeStepAttention(cache, stepQ[s], stepK[s], stepV[s]);
            Assert.Equal(s + 1, cache.SeqLen); // cache grew by exactly one token (append path, no realloc)

            // CPU reference: recompute FULL attention of the current query against the whole [H, s+1, D] K/V.
            accK.Add(stepK[s].Span.ToArray());
            accV.Add(stepV[s].Span.ToArray());
            int seq = s + 1;
            var kFull = new float[H * seq * D];
            var vFull = new float[H * seq * D];
            for (int h = 0; h < H; h++)
            for (int t = 0; t < seq; t++)
            for (int d = 0; d < D; d++)
            {
                // each accK[t] is [H,1,D] head-major: index h*D + d
                kFull[(h * seq + t) * D + d] = accK[t][h * D + d];
                vFull[(h * seq + t) * D + d] = accV[t][h * D + d];
            }

            var feeds = Feeds(
                ("q", stepQ[s]),
                ("k", T(new[] { H, seq, D }, kFull)),
                ("v", T(new[] { H, seq, D }, vFull)));
            Tensor<float> cpuCtx = cpu.Run(feeds)["ctx"].Data;

            float[] g = gpuCtx.Span.ToArray(), c = cpuCtx.Span.ToArray();
            Assert.Equal(c.Length, g.Length);
            for (int i = 0; i < c.Length; i++)
                Assert.True(MathF.Abs(c[i] - g[i]) < 1e-3f,
                    $"KvCache step {s} ctx[{i}] cpu={c[i]} gpu={g[i]}");
            _out.WriteLine($"  step {s}: cache.SeqLen={cache.SeqLen}, ctx matches CPU (max|Δ|<1e-3).");
        }

        // Prove Reset clears the cache for reuse.
        cache.Reset();
        Assert.Equal(0, cache.SeqLen);
    }

    /// <summary>
    /// Timing-only: drives the on-device KV-cache through many decode steps on a distilgpt2-sized head config
    /// (12 heads × 64 dim) and logs total/per-step latency. No assertion on speed (hardware varies); skips green
    /// without CUDA.
    /// </summary>
    [Fact]
    public void Cuda_OnDevice_KvCache_Decode_Perf()
    {
        const int H = 12, D = 64, maxSeq = 256, steps = 128;
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine("KvCachePerf: no CUDA device; skipping.");
            return;
        }

        var decodeGraph = new ModelGraph
        {
            Inputs = new[] { "q" },
            Outputs = new[] { "ctx" },
            Nodes = new[] { new GraphNode("Relu", "noop", new[] { "q" }, new[] { "ctx" }) },
        };
        using var gpu = new IlgpuEngine(decodeGraph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu);
        using GpuKvCache cache = gpu.CreateKvCache(H, maxSeq, D);

        Tensor<float> q = Rand(new[] { H, 1, D }, 1), k = Rand(new[] { H, 1, D }, 2), v = Rand(new[] { H, 1, D }, 3);

        // Warm up (JIT the kernels) for a couple of steps.
        gpu.DecodeStepAttention(cache, q, k, v);
        gpu.DecodeStepAttention(cache, q, k, v);
        cache.Reset();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int s = 0; s < steps; s++)
            gpu.DecodeStepAttention(cache, q, k, v);
        sw.Stop();

        _out.WriteLine($"KvCachePerf: hardware GPU '{gpu.AcceleratorName}' (IsHardwareGpu=true).");
        _out.WriteLine($"KvCachePerf: {steps} decode steps (H={H}, D={D}, growing 1..{steps} ctx tokens) " +
                       $"in {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / steps:F3} ms/step.");
        Assert.Equal(steps, cache.SeqLen);
    }

    // ---- Stretch: probe how far the real distilgpt2 graph runs on the GPU engine ----

    /// <summary>
    /// Loads the real distilgpt2 ONNX graph and confirms the GPU engine now dispatches EVERY op in it — the
    /// integer/mask position-id prologue ops (Range/ConstantOfShape/Equal/Greater/Trilu/ScatterND) that used to
    /// force a CPU fallback are now handled host-side by the engine, so the whole graph is GPU-dispatchable with
    /// an empty fallback set. The heavy transformer compute (Gemm/MatMul/Softmax/LayerNorm/attention, the
    /// Pow/Tanh GELU, Reshape/Transpose/Split/Concat/Gather/Slice plumbing) is proven elsewhere in this file
    /// (the attention block + multi-step KV-cache tests); the prologue ops are exercised in GpuPrologueOpsTests.
    /// </summary>
    [Fact]
    public void DistilGpt2_Gpu_Coverage_Is_Complete()
    {
        if (!TryFindDistilGpt2(out string modelPath))
        {
            _out.WriteLine("DistilGpt2Coverage: distilgpt2.onnx not found " +
                           "(looked under MODELSHARP_MODELS_DIR / repo models/); skipping.");
            return;
        }

        ModelGraph graph = ModelSharp.Onnx.OnnxModelLoader.LoadModel(modelPath);

        // The exact op-type set the GPU engine's Run switch dispatches (kept in sync with IlgpuEngine.Run).
        var gpuOps = new HashSet<string>(StringComparer.Ordinal)
        {
            "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh", "Gelu", "Exp", "Sqrt", "LeakyRelu",
            "Transpose", "Softmax", "ReduceSum", "ReduceMean", "LayerNormalization", "Gather", "Concat",
            "Slice", "Cast", "MatMul", "Gemm", "Conv", "Erf", "Pow", "Where", "Reshape", "Unsqueeze",
            "Squeeze", "Shape", "Constant", "Expand", "Split",
            // B5 completion — integer/mask prologue ops, now GPU-dispatched (host-side):
            "Range", "ConstantOfShape", "Equal", "Greater", "Trilu", "ScatterND",
        };

        var distinct = graph.Nodes.Select(n => n.OpType).Distinct().ToHashSet();
        var unaccounted = distinct.Where(o => !gpuOps.Contains(o)).OrderBy(o => o).ToList();
        Assert.True(unaccounted.Count == 0,
            $"distilgpt2 has op(s) the GPU engine does not dispatch: {string.Join(", ", unaccounted)}");

        // Every node is GPU-dispatchable: no fallback remains.
        int gpuDispatchable = graph.Nodes.Count(n => gpuOps.Contains(n.OpType));
        Assert.Equal(graph.Nodes.Count, gpuDispatchable);
        _out.WriteLine($"DistilGpt2Coverage: {gpuDispatchable}/{graph.Nodes.Count} nodes are GPU-dispatchable (100%).");
        _out.WriteLine("No CPU fallback remains; the whole distilgpt2 graph is GPU-dispatchable.");

        GraphNode? firstFallback = graph.Nodes.FirstOrDefault(n => !gpuOps.Contains(n.OpType));
        Assert.Null(firstFallback);
    }
}
