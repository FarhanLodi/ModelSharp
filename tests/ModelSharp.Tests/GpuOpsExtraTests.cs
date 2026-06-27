using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Coverage for the GPU ops added beyond the original elementwise/MatMul/Conv set:
/// the activations (Sigmoid/Tanh/Gelu/LeakyRelu/Exp/Sqrt), Transpose, Softmax, and
/// ReduceSum/ReduceMean. Each runs the same graph through the ILGPU engine (on the CPU
/// accelerator, so it works on any machine) and the managed CPU engine and asserts they agree.
/// </summary>
public class GpuOpsExtraTests
{
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

    private static void AssertGpuMatchesCpu(ModelGraph graph, Dictionary<string, NamedTensor> feeds, float tol = 1e-4f)
    {
        using var gpu = new IlgpuEngine(graph, preferCpu: true);
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
            for (int i = 0; i < ca.Length; i++)
                Assert.True(MathF.Abs(ca[i] - ga[i]) < tol, $"{name}[{i}] cpu={ca[i]} gpu={ga[i]}");
        }
    }

    private static ModelGraph Unary(string op, Dictionary<string, object>? attrs = null) => new ModelGraph
    {
        Inputs = new[] { "x" },
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode(op, op.ToLowerInvariant(), new[] { "x" }, new[] { "y" }, attrs) },
    };

    [Fact]
    public void Ilgpu_Sigmoid_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Sigmoid"), Feeds(("x", Rand(new[] { 2, 3 }, 1, -6f, 6f))), 1e-3f);

    [Fact]
    public void Ilgpu_Tanh_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Tanh"), Feeds(("x", Rand(new[] { 2, 3 }, 2, -4f, 4f))), 1e-3f);

    [Fact]
    public void Ilgpu_Gelu_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Gelu"), Feeds(("x", Rand(new[] { 2, 3 }, 3, -4f, 4f))), 1e-3f);

    [Fact]
    public void Ilgpu_LeakyRelu_Default_Alpha_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("LeakyRelu"), Feeds(("x", Rand(new[] { 2, 4 }, 4))));

    [Fact]
    public void Ilgpu_LeakyRelu_Explicit_Alpha_Matches_Cpu() =>
        AssertGpuMatchesCpu(
            Unary("LeakyRelu", new Dictionary<string, object> { ["alpha"] = 0.1f }),
            Feeds(("x", Rand(new[] { 2, 4 }, 5))));

    [Fact]
    public void Ilgpu_Exp_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Exp"), Feeds(("x", Rand(new[] { 2, 3 }, 6, -2f, 2f))), 1e-3f);

    [Fact]
    public void Ilgpu_Sqrt_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Sqrt"), Feeds(("x", Rand(new[] { 2, 3 }, 7, 0.1f, 9f))), 1e-3f);

    [Fact]
    public void Ilgpu_Transpose_2D_Matches_Cpu()
    {
        var g = Unary("Transpose", new Dictionary<string, object> { ["perm"] = new long[] { 1, 0 } });
        AssertGpuMatchesCpu(g, Feeds(("x", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6))));
    }

    [Fact]
    public void Ilgpu_Transpose_3D_Matches_Cpu()
    {
        var g = Unary("Transpose", new Dictionary<string, object> { ["perm"] = new long[] { 0, 2, 1 } });
        AssertGpuMatchesCpu(g, Feeds(("x", Rand(new[] { 2, 3, 4 }, 8))));
    }

    [Fact]
    public void Ilgpu_Softmax_LastAxis_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Softmax"), Feeds(("x", Rand(new[] { 2, 5 }, 9, -3f, 3f))), 1e-3f);

    [Theory]
    [InlineData(1L)]
    [InlineData(0L)]
    public void Ilgpu_ReduceSum_Axis1_Matches_Cpu(long keepdims)
    {
        var g = Unary("ReduceSum", new Dictionary<string, object>
        {
            ["axes"] = new long[] { 1 },
            ["keepdims"] = keepdims,
        });
        AssertGpuMatchesCpu(g, Feeds(("x", Rand(new[] { 2, 3, 4 }, 10))), 1e-3f);
    }

    [Theory]
    [InlineData(1L)]
    [InlineData(0L)]
    public void Ilgpu_ReduceMean_Axis1_Matches_Cpu(long keepdims)
    {
        var g = Unary("ReduceMean", new Dictionary<string, object>
        {
            ["axes"] = new long[] { 1 },
            ["keepdims"] = keepdims,
        });
        AssertGpuMatchesCpu(g, Feeds(("x", Rand(new[] { 2, 3, 4 }, 11))), 1e-3f);
    }

    // ===========================================================================================
    //  New native float kernels (B-GPU-2): unary/activation, ReduceMax/Min/Prod, variadic
    //  Min/Max/Sum/Mean, Clip, Pad, Tile. Each runs the ILGPU CPU accelerator vs the managed CPU
    //  engine (so it covers parity on any machine, no CUDA needed) — the device-routing/native-kernel
    //  logic is identical on CUDA. A few are re-run on hardware CUDA in GpuCudaParityTests.
    // ===========================================================================================

    // ---- extra native unary float ops / activations ----

    [Fact] public void Ilgpu_Sign_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Sign"), Feeds(("x", T(new[] { 2, 3 }, -2f, -0.5f, 0f, 0.5f, 3f, -7f))));

    [Fact] public void Ilgpu_Floor_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Floor"), Feeds(("x", T(new[] { 2, 3 }, -2.3f, -0.5f, 0.4f, 1.5f, 2.9f, -7.1f))));

    [Fact] public void Ilgpu_Ceil_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Ceil"), Feeds(("x", T(new[] { 2, 3 }, -2.3f, -0.5f, 0.4f, 1.5f, 2.9f, -7.1f))));

    [Fact] public void Ilgpu_Round_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Round"), Feeds(("x", T(new[] { 2, 4 }, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 0.49f, -0.51f))));

    [Fact] public void Ilgpu_Softplus_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Softplus"), Feeds(("x", Rand(new[] { 2, 4 }, 20, -4f, 4f))), 1e-3f);

    [Fact] public void Ilgpu_Mish_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Mish"), Feeds(("x", Rand(new[] { 2, 4 }, 21, -4f, 4f))), 1e-3f);

    [Fact] public void Ilgpu_HardSwish_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("HardSwish"), Feeds(("x", Rand(new[] { 2, 4 }, 22, -5f, 5f))), 1e-3f);

    [Fact] public void Ilgpu_HardSigmoid_Default_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("HardSigmoid"), Feeds(("x", Rand(new[] { 2, 4 }, 23, -8f, 8f))), 1e-3f);

    [Fact] public void Ilgpu_HardSigmoid_Explicit_Matches_Cpu() =>
        AssertGpuMatchesCpu(
            Unary("HardSigmoid", new Dictionary<string, object> { ["alpha"] = 0.1f, ["beta"] = 0.6f }),
            Feeds(("x", Rand(new[] { 2, 4 }, 24, -8f, 8f))), 1e-3f);

    [Fact] public void Ilgpu_Elu_Default_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Elu"), Feeds(("x", Rand(new[] { 2, 4 }, 25, -4f, 4f))), 1e-3f);

    [Fact] public void Ilgpu_Elu_Alpha_Matches_Cpu() =>
        AssertGpuMatchesCpu(
            Unary("Elu", new Dictionary<string, object> { ["alpha"] = 1.5f }),
            Feeds(("x", Rand(new[] { 2, 4 }, 26, -4f, 4f))), 1e-3f);

    [Fact] public void Ilgpu_Selu_Default_Matches_Cpu() =>
        AssertGpuMatchesCpu(Unary("Selu"), Feeds(("x", Rand(new[] { 2, 4 }, 27, -4f, 4f))), 1e-3f);

    // ---- ReduceMax / ReduceMin / ReduceProd ----

    [Theory]
    [InlineData("ReduceMax", 1L)]
    [InlineData("ReduceMax", 0L)]
    [InlineData("ReduceMin", 1L)]
    [InlineData("ReduceMin", 0L)]
    [InlineData("ReduceProd", 1L)]
    [InlineData("ReduceProd", 0L)]
    public void Ilgpu_ReduceMaxMinProd_Axis1_Matches_Cpu(string op, long keepdims)
    {
        var g = Unary(op, new Dictionary<string, object>
        {
            ["axes"] = new long[] { 1 },
            ["keepdims"] = keepdims,
        });
        // Keep ReduceProd values near 1 so the running product stays well-conditioned.
        AssertGpuMatchesCpu(g, Feeds(("x", Rand(new[] { 2, 3, 4 }, 30, 0.5f, 1.5f))), 1e-3f);
    }

    [Fact]
    public void Ilgpu_ReduceMax_AllAxes_Matches_Cpu()
    {
        var g = Unary("ReduceMax", new Dictionary<string, object> { ["keepdims"] = 0L });
        AssertGpuMatchesCpu(g, Feeds(("x", Rand(new[] { 3, 5 }, 31, -3f, 3f))), 1e-3f);
    }

    // ---- variadic Min / Max / Sum / Mean (with broadcasting) ----

    private static ModelGraph Variadic(string op, int inputs) => new ModelGraph
    {
        Inputs = Enumerable.Range(0, inputs).Select(i => $"x{i}").ToArray(),
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode(op, op.ToLowerInvariant(),
            Enumerable.Range(0, inputs).Select(i => $"x{i}").ToArray(), new[] { "y" }) },
    };

    [Theory]
    [InlineData("Min")]
    [InlineData("Max")]
    [InlineData("Sum")]
    [InlineData("Mean")]
    public void Ilgpu_Variadic_ThreeInputs_Matches_Cpu(string op)
    {
        var g = Variadic(op, 3);
        AssertGpuMatchesCpu(g, Feeds(
            ("x0", Rand(new[] { 2, 3 }, 40)),
            ("x1", Rand(new[] { 2, 3 }, 41)),
            ("x2", Rand(new[] { 2, 3 }, 42))), 1e-4f);
    }

    [Theory]
    [InlineData("Min")]
    [InlineData("Max")]
    [InlineData("Sum")]
    [InlineData("Mean")]
    public void Ilgpu_Variadic_Broadcast_Matches_Cpu(string op)
    {
        var g = Variadic(op, 3);
        AssertGpuMatchesCpu(g, Feeds(
            ("x0", Rand(new[] { 2, 3 }, 43)),       // [2,3]
            ("x1", Rand(new[] { 1, 3 }, 44)),       // broadcast row
            ("x2", Rand(new[] { 2, 1 }, 45))), 1e-4f); // broadcast col
    }

    // ---- Clip ----

    [Fact]
    public void Ilgpu_Clip_Attrs_Matches_Cpu()
    {
        var g = Unary("Clip", new Dictionary<string, object> { ["min"] = -0.5f, ["max"] = 0.5f });
        AssertGpuMatchesCpu(g, Feeds(("x", Rand(new[] { 3, 4 }, 50, -2f, 2f))));
    }

    [Fact]
    public void Ilgpu_Clip_MinMaxInputs_Matches_Cpu()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Clip", "clip", new[] { "x", "lo", "hi" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["lo"] = new Tensor<float>(new TensorShape(), new[] { -0.3f }),
                ["hi"] = new Tensor<float>(new TensorShape(), new[] { 0.7f }),
            },
        };
        AssertGpuMatchesCpu(g, Feeds(("x", Rand(new[] { 3, 4 }, 51, -2f, 2f))));
    }

    // ---- Pad (constant / edge / reflect) ----

    private static ModelGraph PadGraph(long[] pads, string mode, float value)
    {
        var attrs = new Dictionary<string, object> { ["mode"] = mode };
        if (mode == "constant") attrs["value"] = value;
        return new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Pad", "pad", new[] { "x", "pads" }, new[] { "y" }, attrs) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["pads"] = new Tensor<long>(new TensorShape(pads.Length), pads),
            },
        };
    }

    [Fact]
    public void Ilgpu_Pad_Constant_Matches_Cpu() =>
        AssertGpuMatchesCpu(PadGraph(new long[] { 1, 2, 1, 0 }, "constant", 0.5f),
            Feeds(("x", Rand(new[] { 2, 3 }, 60))));

    [Fact]
    public void Ilgpu_Pad_Edge_Matches_Cpu() =>
        AssertGpuMatchesCpu(PadGraph(new long[] { 2, 1, 1, 2 }, "edge", 0f),
            Feeds(("x", Rand(new[] { 3, 4 }, 61))));

    [Fact]
    public void Ilgpu_Pad_Reflect_Matches_Cpu() =>
        AssertGpuMatchesCpu(PadGraph(new long[] { 2, 2, 2, 2 }, "reflect", 0f),
            Feeds(("x", Rand(new[] { 4, 5 }, 62))));

    [Fact]
    public void Ilgpu_Pad_Negative_Crop_Matches_Cpu() =>
        AssertGpuMatchesCpu(PadGraph(new long[] { -1, 0, 0, -1 }, "constant", 0f),
            Feeds(("x", Rand(new[] { 3, 4 }, 63))));

    // ---- Tile ----

    private static ModelGraph TileGraph(long[] reps) => new ModelGraph
    {
        Inputs = new[] { "x" },
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode("Tile", "tile", new[] { "x", "reps" }, new[] { "y" }) },
        Initializers = new Dictionary<string, Tensor>
        {
            ["reps"] = new Tensor<long>(new TensorShape(reps.Length), reps),
        },
    };

    [Fact]
    public void Ilgpu_Tile_2D_Matches_Cpu() =>
        AssertGpuMatchesCpu(TileGraph(new long[] { 2, 3 }), Feeds(("x", Rand(new[] { 2, 3 }, 70))));

    [Fact]
    public void Ilgpu_Tile_3D_Matches_Cpu() =>
        AssertGpuMatchesCpu(TileGraph(new long[] { 1, 2, 2 }), Feeds(("x", Rand(new[] { 2, 2, 3 }, 71))));
}
