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
}
