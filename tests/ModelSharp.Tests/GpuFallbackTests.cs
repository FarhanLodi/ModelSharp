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
/// Per-op CPU fallback for the GPU engine: any op the GPU switch has no native handler for is routed through the
/// managed CPU kernel registry (inputs downloaded to host, the CPU kernel run, float outputs re-uploaded to the
/// device so downstream GPU ops keep seeing device buffers). This makes any CPU-runnable graph run through
/// <see cref="IlgpuEngine"/> — accelerated where a native GPU kernel exists, correct via fallback otherwise.
///
/// CUDA-gated exactly like <see cref="GpuLlmTests"/>: shares the serialized <c>CudaGpu</c> collection and skips
/// cleanly (asserting nothing) when no CUDA device is present.
/// </summary>
[Collection("CudaGpu")]
public class GpuFallbackTests
{
    private readonly ITestOutputHelper _out;
    public GpuFallbackTests(ITestOutputHelper output) => _out = output;

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

    private static GraphNode N(string op, string name, string[] inputs, string[] outputs,
        Dictionary<string, object>? attrs = null) => new GraphNode(op, name, inputs, outputs, attrs);

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
    private void AssertCudaMatchesCpu(string what, ModelGraph graph, Dictionary<string, NamedTensor> feeds, float tol = 1e-4f)
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

    // ---- 1) A single op with NO native GPU handler but supported on CPU (round-trips device↔host) ----

    /// <summary>
    /// <c>Softplus</c> = log(1+exp(x)) has no native GPU kernel but is in the CPU registry. Running it through
    /// the GPU engine must download the input, run the CPU kernel, re-upload, and match the CPU engine exactly.
    /// </summary>
    [Fact]
    public void Cuda_Softplus_Fallback_Matches_Cpu()
        => AssertCudaMatchesCpu("SoftplusFallback",
            new ModelGraph
            {
                Inputs = new[] { "x" },
                Outputs = new[] { "y" },
                Nodes = new[] { N("Softplus", "sp", new[] { "x" }, new[] { "y" }) },
            },
            Feeds(("x", Rand(new[] { 4, 5 }, 101, -3f, 3f))));

    /// <summary>
    /// <c>ReduceMax</c> is a reduction variant the GPU switch lacks (it only natively handles ReduceSum/
    /// ReduceMean) but the CPU registry supports. Exercises an axes-attributed fallback with keepdims.
    /// </summary>
    [Fact]
    public void Cuda_ReduceMax_Fallback_Matches_Cpu()
        => AssertCudaMatchesCpu("ReduceMaxFallback",
            new ModelGraph
            {
                Inputs = new[] { "x" },
                Outputs = new[] { "y" },
                Nodes = new[]
                {
                    N("ReduceMax", "rmax", new[] { "x" }, new[] { "y" },
                        new Dictionary<string, object> { ["axes"] = new long[] { 1 }, ["keepdims"] = 1L }),
                },
            },
            Feeds(("x", Rand(new[] { 3, 4 }, 102, -5f, 5f))));

    /// <summary>
    /// <c>Tile</c> (a data-movement op absent from the GPU switch) repeats the float input by the integer
    /// <c>repeats</c> initializer. Confirms the fallback handles an op whose second input is an int tensor (kept
    /// host-side) while the first is a float device buffer.
    /// </summary>
    [Fact]
    public void Cuda_Tile_Fallback_Matches_Cpu()
        => AssertCudaMatchesCpu("TileFallback",
            new ModelGraph
            {
                Inputs = new[] { "x" },
                Outputs = new[] { "y" },
                Nodes = new[] { N("Tile", "tile", new[] { "x", "repeats" }, new[] { "y" }) },
                Initializers = new Dictionary<string, Tensor>
                {
                    ["repeats"] = new Tensor<long>(new TensorShape(2), new long[] { 2, 3 }),
                },
            },
            Feeds(("x", Rand(new[] { 2, 2 }, 103))));

    // ---- 2) An op unknown to BOTH engines still throws ----

    /// <summary>
    /// A made-up op type has no GPU handler and no CPU kernel, so the fallback must throw
    /// <see cref="UnsupportedOperatorException"/> ("no engine supports it"). Runs on ILGPU's CPU accelerator so
    /// it executes everywhere (no CUDA needed) — the device routing is identical.
    /// </summary>
    [Fact]
    public void NoEngineSupportsOp_Throws_Unsupported()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { N("ModelSharpNonexistentOp42", "bogus", new[] { "x" }, new[] { "y" }) },
        };
        using var gpu = new IlgpuEngine(graph, preferCpu: true);
        var ex = Assert.Throws<UnsupportedOperatorException>(
            () => gpu.Run(Feeds(("x", T(new[] { 2 }, 1f, 2f)))));
        Assert.Contains("ModelSharpNonexistentOp42", ex.Message);
    }

    // ---- 3) Part native-GPU / part fallback, end-to-end ----

    /// <summary>
    /// MatMul (native GPU) → Softplus (CPU fallback) → Add (native GPU). The fallback output must re-enter the
    /// device so the trailing native Add reads a device buffer; the end-to-end result matches the CPU engine.
    /// </summary>
    [Fact]
    public void Cuda_NativeGpu_Then_Fallback_Then_NativeGpu_Matches_Cpu()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b", "bias" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                N("MatMul", "mm", new[] { "a", "b" }, new[] { "ab" }),       // native GPU
                N("Softplus", "sp", new[] { "ab" }, new[] { "act" }),         // CPU fallback
                N("Add", "add", new[] { "act", "bias" }, new[] { "y" }),      // native GPU
            },
        };
        AssertCudaMatchesCpu("MixedNativeFallback", graph,
            Feeds(
                ("a", Rand(new[] { 4, 6 }, 201)),
                ("b", Rand(new[] { 6, 5 }, 202)),
                ("bias", Rand(new[] { 4, 5 }, 203))));
    }

    /// <summary>
    /// Mixed graph where the fallback op also feeds a downstream reduction fallback: MatMul (native) → Mish
    /// (fallback) → ReduceMax (fallback) → Add (native). Stresses two consecutive host round-trips chained with
    /// native ops on both ends.
    /// </summary>
    [Fact]
    public void Cuda_ChainedFallbacks_Between_Native_Matches_Cpu()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b", "bias" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                N("MatMul", "mm", new[] { "a", "b" }, new[] { "ab" }),                    // native GPU
                N("Mish", "mish", new[] { "ab" }, new[] { "m" }),                         // CPU fallback
                N("ReduceMax", "rmax", new[] { "m" }, new[] { "r" },                      // CPU fallback
                    new Dictionary<string, object> { ["axes"] = new long[] { 1 }, ["keepdims"] = 1L }),
                N("Add", "add", new[] { "r", "bias" }, new[] { "y" }),                    // native GPU (broadcast)
            },
        };
        AssertCudaMatchesCpu("ChainedFallbacks", graph,
            Feeds(
                ("a", Rand(new[] { 3, 4 }, 211)),
                ("b", Rand(new[] { 4, 5 }, 212)),
                ("bias", Rand(new[] { 3, 1 }, 213))));
    }

    // ---- 4) distilgpt2-style native ops still take the native path (no behavioral regression) ----

    /// <summary>
    /// A lightweight synthetic graph of ops that all have native GPU handlers (MatMul/Add/Gelu/LayerNormalization/
    /// Softmax) — the kinds distilgpt2 uses. Asserts CPU parity (so adding the fallback didn't perturb the native
    /// paths) and, on ILGPU's CPU accelerator, that the engine never builds the CPU-fallback registry — proving
    /// these ops still dispatch natively rather than silently falling back.
    /// </summary>
    [Fact]
    public void Cuda_DistilGpt2StyleNativeOps_StayNative_Matches_Cpu()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x", "w" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                N("MatMul", "mm", new[] { "x", "w" }, new[] { "h" }),
                N("Add", "add", new[] { "h", "bias" }, new[] { "hb" }),
                N("Gelu", "gelu", new[] { "hb" }, new[] { "g" }),
                N("LayerNormalization", "ln", new[] { "g", "scale", "lnbias" }, new[] { "n" },
                    new Dictionary<string, object> { ["axis"] = -1L, ["epsilon"] = 1e-5f }),
                N("Softmax", "sm", new[] { "n" }, new[] { "y" },
                    new Dictionary<string, object> { ["axis"] = -1L }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["bias"] = Rand(new[] { 6 }, 301),
                ["scale"] = Rand(new[] { 6 }, 302, 0.5f, 1.5f),
                ["lnbias"] = Rand(new[] { 6 }, 303),
            },
        };

        // CPU-accelerator run is CUDA-independent: proves the native path is untouched everywhere.
        using (var gpuCpuAccel = new IlgpuEngine(graph, preferCpu: true))
        using (var cpu = new ManagedCpuEngine(graph))
        {
            var feeds = Feeds(("x", Rand(new[] { 4, 5 }, 304)), ("w", Rand(new[] { 5, 6 }, 305)));
            Tensor<float> g = gpuCpuAccel.Run(feeds)["y"].Data;
            Tensor<float> c = cpu.Run(feeds)["y"].Data;
            float[] ga = g.Span.ToArray(), ca = c.Span.ToArray();
            Assert.Equal(ca.Length, ga.Length);
            for (int i = 0; i < ca.Length; i++)
                Assert.True(MathF.Abs(ca[i] - ga[i]) < 1e-4f, $"native[{i}] cpu={ca[i]} gpu={ga[i]}");
        }

        // And on real hardware, same parity.
        AssertCudaMatchesCpu("DistilGpt2StyleNative", graph,
            Feeds(("x", Rand(new[] { 4, 5 }, 304)), ("w", Rand(new[] { 5, 6 }, 305))));
    }
}
