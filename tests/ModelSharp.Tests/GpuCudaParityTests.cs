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
/// Phase 0 / B1 — real-CUDA parity. These tests exercise the <see cref="IlgpuEngine"/> on the
/// actual hardware GPU (<c>preferCpu:false</c>), running the same op graphs covered by
/// <see cref="GpuEngineTests"/> and <see cref="GpuOpsExtraTests"/> (which only use the ILGPU CPU
/// accelerator) and comparing every output against the managed CPU engine.
///
/// When no hardware CUDA/OpenCL device is present (e.g. CI on a CPU-only box) every test writes a
/// "no CUDA device; skipping." line via <see cref="ITestOutputHelper"/> and returns green, so the
/// suite still passes everywhere. On the RTX 4090 box these tests actually drive the GPU; the
/// tolerance is ~1e-3 because CUDA fp math diverges slightly from the CPU.
/// </summary>
[Collection("CudaGpu")]
public class GpuCudaParityTests
{
    private readonly ITestOutputHelper _out;

    public GpuCudaParityTests(ITestOutputHelper output) => _out = output;

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

    /// <summary>
    /// True when ILGPU can see at least one non-CPU (Cuda/OpenCL) device on this machine. Probed
    /// directly against <see cref="Context"/> so we can decide to skip without constructing an engine.
    /// </summary>
    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Runs <paramref name="graph"/> + <paramref name="feeds"/> through the CUDA engine
    /// (<c>preferCpu:false</c>) and the managed CPU engine and asserts every output matches in shape,
    /// length and value to <paramref name="tol"/>. Confirms via output that the GPU was actually hardware.
    /// If no hardware GPU is present, logs and returns (green).
    /// </summary>
    private void AssertCudaMatchesCpu(string what, ModelGraph graph, Dictionary<string, NamedTensor> feeds, float tol = 1e-3f)
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine($"{what}: no CUDA device; skipping.");
            return;
        }

        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"{what}: expected a hardware GPU but got '{gpu.AcceleratorName}'.");
        _out.WriteLine($"{what}: running on hardware GPU '{gpu.AcceleratorName}' (IsHardwareGpu=true).");

        using var cpu = new ManagedCpuEngine(graph);

        IReadOnlyDictionary<string, NamedTensor> gpuOut = gpu.Run(feeds);
        IReadOnlyDictionary<string, NamedTensor> cpuOut = cpu.Run(feeds);

        foreach (string name in graph.Outputs)
        {
            Tensor<float> g = gpuOut[name].Data;
            Tensor<float> c = cpuOut[name].Data;
            Assert.Equal(c.Shape.Dimensions.ToArray(), g.Shape.Dimensions.ToArray());

            float[] ga = g.Span.ToArray();
            float[] ca = c.Span.ToArray();
            Assert.Equal(ca.Length, ga.Length);
            for (int i = 0; i < ca.Length; i++)
                Assert.True(MathF.Abs(ca[i] - ga[i]) < tol, $"{what}:{name}[{i}] cpu={ca[i]} gpu={ga[i]}");
        }
    }

    private static ModelGraph Binary(string op) => new ModelGraph
    {
        Inputs = new[] { "a", "b" },
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode(op, op.ToLowerInvariant(), new[] { "a", "b" }, new[] { "y" }) },
    };

    private static ModelGraph Unary(string op, Dictionary<string, object>? attrs = null) => new ModelGraph
    {
        Inputs = new[] { "x" },
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode(op, op.ToLowerInvariant(), new[] { "x" }, new[] { "y" }, attrs) },
    };

    // ---- Device-presence assertion ----

    [Fact]
    public void Cuda_Device_Is_Hardware_Gpu_When_Present()
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine("Cuda_Device_Is_Hardware_Gpu: no CUDA device; skipping.");
            return;
        }

        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Add", "add0", new[] { "a", "b" }, new[] { "y" }) },
        };
        using var gpu = new IlgpuEngine(graph, preferCpu: false);

        Assert.True(gpu.IsHardwareGpu);
        Assert.NotNull(gpu.AcceleratorName);
        _out.WriteLine($"Hardware GPU selected: '{gpu.AcceleratorName}' (IsHardwareGpu={gpu.IsHardwareGpu}).");
        // The accelerator name should not be the CPU fallback.
        Assert.DoesNotContain("CPU", gpu.AcceleratorName, StringComparison.OrdinalIgnoreCase);
    }

    // ---- Elementwise + broadcast ----

    [Fact]
    public void Cuda_Add_Equal_Shape()
        => AssertCudaMatchesCpu("Add", Binary("Add"),
            Feeds(("a", Rand(new[] { 4, 5 }, 1)), ("b", Rand(new[] { 4, 5 }, 2))));

    [Fact]
    public void Cuda_Sub_Equal_Shape()
        => AssertCudaMatchesCpu("Sub", Binary("Sub"),
            Feeds(("a", Rand(new[] { 4, 5 }, 3)), ("b", Rand(new[] { 4, 5 }, 4))));

    [Fact]
    public void Cuda_Mul_Equal_Shape()
        => AssertCudaMatchesCpu("Mul", Binary("Mul"),
            Feeds(("a", Rand(new[] { 4, 5 }, 5)), ("b", Rand(new[] { 4, 5 }, 6))));

    [Fact]
    public void Cuda_Div_Equal_Shape()
        => AssertCudaMatchesCpu("Div", Binary("Div"),
            Feeds(("a", Rand(new[] { 4, 5 }, 7)), ("b", Rand(new[] { 4, 5 }, 8, 1f, 3f))));

    [Fact]
    public void Cuda_Add_Broadcast_Row()
        => AssertCudaMatchesCpu("AddBroadcastRow", Binary("Add"),
            Feeds(("a", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)), ("b", T(new[] { 3 }, 10, 20, 30))));

    [Fact]
    public void Cuda_Div_Broadcast_Column()
        => AssertCudaMatchesCpu("DivBroadcastCol", Binary("Div"),
            Feeds(("a", T(new[] { 2, 3 }, 2, 4, 6, 9, 12, 15)), ("b", T(new[] { 2, 1 }, 2, 3))));

    [Fact]
    public void Cuda_Mul_Broadcast_3D()
        => AssertCudaMatchesCpu("MulBroadcast3D", Binary("Mul"),
            Feeds(("a", Rand(new[] { 2, 3, 4 }, 21)), ("b", Rand(new[] { 1, 3, 1 }, 22))));

    // ---- Activations ----

    [Fact]
    public void Cuda_Relu()
        => AssertCudaMatchesCpu("Relu", Unary("Relu"), Feeds(("x", Rand(new[] { 4, 6 }, 31, -5f, 5f))));

    [Fact]
    public void Cuda_Sigmoid()
        => AssertCudaMatchesCpu("Sigmoid", Unary("Sigmoid"), Feeds(("x", Rand(new[] { 4, 6 }, 32, -6f, 6f))));

    [Fact]
    public void Cuda_Tanh()
        => AssertCudaMatchesCpu("Tanh", Unary("Tanh"), Feeds(("x", Rand(new[] { 4, 6 }, 33, -4f, 4f))));

    [Fact]
    public void Cuda_Gelu()
        => AssertCudaMatchesCpu("Gelu", Unary("Gelu"), Feeds(("x", Rand(new[] { 4, 6 }, 34, -4f, 4f))));

    [Fact]
    public void Cuda_Exp()
        => AssertCudaMatchesCpu("Exp", Unary("Exp"), Feeds(("x", Rand(new[] { 4, 6 }, 35, -2f, 2f))));

    [Fact]
    public void Cuda_Sqrt()
        => AssertCudaMatchesCpu("Sqrt", Unary("Sqrt"), Feeds(("x", Rand(new[] { 4, 6 }, 36, 0.1f, 9f))));

    [Fact]
    public void Cuda_LeakyRelu()
        => AssertCudaMatchesCpu("LeakyRelu",
            Unary("LeakyRelu", new Dictionary<string, object> { ["alpha"] = 0.1f }),
            Feeds(("x", Rand(new[] { 4, 6 }, 37, -5f, 5f))));

    // ---- Transpose ----

    [Fact]
    public void Cuda_Transpose_2D()
        => AssertCudaMatchesCpu("Transpose2D",
            Unary("Transpose", new Dictionary<string, object> { ["perm"] = new long[] { 1, 0 } }),
            Feeds(("x", Rand(new[] { 3, 5 }, 41))));

    [Fact]
    public void Cuda_Transpose_3D()
        => AssertCudaMatchesCpu("Transpose3D",
            Unary("Transpose", new Dictionary<string, object> { ["perm"] = new long[] { 0, 2, 1 } }),
            Feeds(("x", Rand(new[] { 2, 3, 4 }, 42))));

    // ---- Softmax ----

    [Fact]
    public void Cuda_Softmax_LastAxis()
        => AssertCudaMatchesCpu("SoftmaxLast", Unary("Softmax"), Feeds(("x", Rand(new[] { 4, 7 }, 51, -3f, 3f))));

    [Fact]
    public void Cuda_Softmax_Axis0()
        => AssertCudaMatchesCpu("SoftmaxAxis0",
            Unary("Softmax", new Dictionary<string, object> { ["axis"] = 0L }),
            Feeds(("x", Rand(new[] { 4, 7 }, 52, -3f, 3f))));

    // ---- Reductions ----

    [Theory]
    [InlineData(1L)]
    [InlineData(0L)]
    public void Cuda_ReduceSum_Axis1(long keepdims)
        => AssertCudaMatchesCpu($"ReduceSum_kd{keepdims}",
            Unary("ReduceSum", new Dictionary<string, object> { ["axes"] = new long[] { 1 }, ["keepdims"] = keepdims }),
            Feeds(("x", Rand(new[] { 3, 4, 5 }, 61))));

    [Theory]
    [InlineData(1L)]
    [InlineData(0L)]
    public void Cuda_ReduceMean_Axis1(long keepdims)
        => AssertCudaMatchesCpu($"ReduceMean_kd{keepdims}",
            Unary("ReduceMean", new Dictionary<string, object> { ["axes"] = new long[] { 1 }, ["keepdims"] = keepdims }),
            Feeds(("x", Rand(new[] { 3, 4, 5 }, 62))));

    // ---- MatMul ----

    [Fact]
    public void Cuda_MatMul_2D()
        => AssertCudaMatchesCpu("MatMul2D",
            new ModelGraph
            {
                Inputs = new[] { "a", "b" },
                Outputs = new[] { "y" },
                Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "y" }) },
            },
            Feeds(("a", Rand(new[] { 6, 7 }, 71)), ("b", Rand(new[] { 7, 5 }, 72))));

    [Fact]
    public void Cuda_MatMul_Batched_3D()
        => AssertCudaMatchesCpu("MatMulBatched3D",
            new ModelGraph
            {
                Inputs = new[] { "a", "b" },
                Outputs = new[] { "y" },
                Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "y" }) },
            },
            Feeds(("a", Rand(new[] { 2, 3, 4 }, 73)), ("b", Rand(new[] { 2, 4, 3 }, 74))));

    [Fact]
    public void Cuda_MatMul_Batch_Broadcast()
        => AssertCudaMatchesCpu("MatMulBatchBroadcast",
            new ModelGraph
            {
                Inputs = new[] { "a", "b" },
                Outputs = new[] { "y" },
                Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "y" }) },
            },
            Feeds(("a", Rand(new[] { 3, 2, 4 }, 75)), ("b", Rand(new[] { 4, 5 }, 76))));

    // ---- Conv2D ----

    [Fact]
    public void Cuda_Conv2D_Stride_Pad_Bias()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Conv", "c", new[] { "x", "w", "bias" }, new[] { "y" },
                    new Dictionary<string, object>
                    {
                        ["strides"] = new long[] { 2, 2 },
                        ["pads"] = new long[] { 1, 1, 1, 1 },
                        ["dilations"] = new long[] { 1, 1 },
                    }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["w"] = T(new[] { 1, 1, 3, 3 }, 1, 0, -1, 1, 0, -1, 1, 0, -1),
                ["bias"] = T(new[] { 1 }, 0.5f),
            },
        };
        float[] xData = Enumerable.Range(1, 25).Select(i => (float)i).ToArray();
        AssertCudaMatchesCpu("Conv2D", graph, Feeds(("x", T(new[] { 1, 1, 5, 5 }, xData))));
    }

    [Fact]
    public void Cuda_Conv2D_Grouped()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Conv", "c", new[] { "x", "w" }, new[] { "y" },
                    new Dictionary<string, object> { ["group"] = 2L, ["pads"] = new long[] { 1, 1, 1, 1 } }),
            },
            Initializers = new Dictionary<string, Tensor> { ["w"] = Rand(new[] { 4, 2, 3, 3 }, 81) },
        };
        AssertCudaMatchesCpu("Conv2DGrouped", graph, Feeds(("x", Rand(new[] { 1, 4, 4, 4 }, 82))));
    }

    // ---- Multi-op chain (keeps several intermediates on-device) ----

    [Fact]
    public void Cuda_MatMul_Add_Relu_Chain()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b", "bias" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "mmout" }),
                new GraphNode("Add", "add", new[] { "mmout", "bias" }, new[] { "added" }),
                new GraphNode("Relu", "relu", new[] { "added" }, new[] { "y" }),
            },
        };
        AssertCudaMatchesCpu("MatMulAddReluChain", graph,
            Feeds(("a", Rand(new[] { 8, 16 }, 91)), ("b", Rand(new[] { 16, 8 }, 92)), ("bias", Rand(new[] { 8 }, 93))));
    }
}
