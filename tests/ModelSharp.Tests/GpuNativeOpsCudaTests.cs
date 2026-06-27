using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ModelSharp.Cpu;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// Hardware-CUDA parity for the native GPU kernels added in the B-GPU-2 pass — the extra unary/activation
/// ops (Sign/Floor/Ceil/Round/Softplus/Mish/HardSwish/HardSigmoid/Elu/Selu/Clip), the
/// ReduceMax/ReduceMin/ReduceProd reductions, the variadic Min/Max/Sum/Mean folds, and Pad/Tile. The same
/// graphs run on the ILGPU CPU accelerator in <see cref="GpuOpsExtraTests"/> (covering parity on any machine);
/// here they re-run on real CUDA (<c>preferCpu:false</c>), asserting <see cref="IlgpuEngine.IsHardwareGpu"/>.
///
/// <para>CUDA-gated: when no hardware GPU is present every test logs and returns green. Shared-box robust: a
/// device <c>out of memory</c> (a co-tenant process holding VRAM) is caught and skipped (logged, not failed),
/// the same treatment the whole-graph tests use.</para>
/// </summary>
[Collection("CudaGpu")]
public class GpuNativeOpsCudaTests
{
    private readonly ITestOutputHelper _out;
    public GpuNativeOpsCudaTests(ITestOutputHelper output) => _out = output;

    private static Tensor<float> T(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Dictionary<string, NamedTensor> Feeds(params (string name, Tensor<float> t)[] feeds) =>
        feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));

    private static Tensor<float> Rand(int[] dims, int seed, float lo = -1f, float hi = 1f)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (a, d) => a * d);
        float[] data = Enumerable.Range(0, n).Select(_ => lo + (float)rnd.NextDouble() * (hi - lo)).ToArray();
        return T(dims, data);
    }

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch { return false; }
    }

    /// <summary>
    /// Runs <paramref name="graph"/> on hardware CUDA and the managed CPU engine and asserts every float output
    /// matches to <paramref name="tol"/>. Skips (green) with no CUDA device, and skips on a device out-of-memory.
    /// </summary>
    private void AssertCudaMatchesCpu(string what, ModelGraph graph, Dictionary<string, NamedTensor> feeds, float tol = 1e-3f)
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine($"{what}: no CUDA device; skipping.");
            return;
        }

        try
        {
            using var gpu = new IlgpuEngine(graph, preferCpu: false);
            Assert.True(gpu.IsHardwareGpu, $"{what}: expected a hardware GPU but got '{gpu.AcceleratorName}'.");
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
            _out.WriteLine($"{what}: parity OK on '{gpu.AcceleratorName}'.");
        }
        catch (Exception ex) when (ex.Message.Contains("out of memory"))
        {
            _out.WriteLine($"{what}: GPU out of memory (co-tenant holding VRAM); skipping. [{ex.Message}]");
        }
    }

    private static ModelGraph Unary(string op, Dictionary<string, object>? attrs = null) => new ModelGraph
    {
        Inputs = new[] { "x" },
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode(op, op.ToLowerInvariant(), new[] { "x" }, new[] { "y" }, attrs) },
    };

    [Theory]
    [InlineData("Sign")]
    [InlineData("Floor")]
    [InlineData("Ceil")]
    [InlineData("Round")]
    [InlineData("Softplus")]
    [InlineData("Mish")]
    [InlineData("HardSwish")]
    [InlineData("HardSigmoid")]
    [InlineData("Elu")]
    [InlineData("Selu")]
    public void Cuda_NativeUnary(string op)
        => AssertCudaMatchesCpu(op, Unary(op), Feeds(("x", Rand(new[] { 4, 6 }, 100 + op.Length, -4f, 4f))));

    [Fact]
    public void Cuda_Clip()
        => AssertCudaMatchesCpu("Clip",
            Unary("Clip", new Dictionary<string, object> { ["min"] = -0.5f, ["max"] = 0.5f }),
            Feeds(("x", Rand(new[] { 4, 6 }, 110, -2f, 2f))));

    [Theory]
    [InlineData("ReduceMax")]
    [InlineData("ReduceMin")]
    [InlineData("ReduceProd")]
    public void Cuda_ReduceMaxMinProd(string op)
        => AssertCudaMatchesCpu(op,
            Unary(op, new Dictionary<string, object> { ["axes"] = new long[] { 1 }, ["keepdims"] = 1L }),
            Feeds(("x", Rand(new[] { 3, 4, 5 }, 120, 0.5f, 1.5f))));

    [Theory]
    [InlineData("Min")]
    [InlineData("Max")]
    [InlineData("Sum")]
    [InlineData("Mean")]
    public void Cuda_Variadic(string op)
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x0", "x1", "x2" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode(op, op.ToLowerInvariant(), new[] { "x0", "x1", "x2" }, new[] { "y" }) },
        };
        AssertCudaMatchesCpu(op, graph, Feeds(
            ("x0", Rand(new[] { 2, 3 }, 130)),
            ("x1", Rand(new[] { 1, 3 }, 131)),
            ("x2", Rand(new[] { 2, 1 }, 132))));
    }

    [Theory]
    [InlineData("constant")]
    [InlineData("edge")]
    [InlineData("reflect")]
    public void Cuda_Pad(string mode)
    {
        var attrs = new Dictionary<string, object> { ["mode"] = mode };
        if (mode == "constant") attrs["value"] = 0.25f;
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Pad", "pad", new[] { "x", "pads" }, new[] { "y" }, attrs) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["pads"] = new Tensor<long>(new TensorShape(4), new long[] { 2, 2, 2, 2 }),
            },
        };
        AssertCudaMatchesCpu($"Pad_{mode}", graph, Feeds(("x", Rand(new[] { 4, 5 }, 140))));
    }

    [Fact]
    public void Cuda_Tile()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Tile", "tile", new[] { "x", "reps" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["reps"] = new Tensor<long>(new TensorShape(2), new long[] { 2, 3 }),
            },
        };
        AssertCudaMatchesCpu("Tile", graph, Feeds(("x", Rand(new[] { 2, 3 }, 150))));
    }
}
