using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class GpuEngineTests
{
    private static Tensor<float> T(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Dictionary<string, NamedTensor> Feeds(params (string name, Tensor<float> t)[] feeds) =>
        feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));

    /// <summary>
    /// Runs the same graph + feeds through the GPU engine (on ILGPU's CPU accelerator, so it works on CI)
    /// and the managed CPU engine, asserting every output matches in shape and value to ~1e-4.
    /// </summary>
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
            Assert.Equal(ca.Length, ga.Length);
            for (int i = 0; i < ca.Length; i++)
                Assert.True(MathF.Abs(ca[i] - ga[i]) < tol, $"{name}[{i}] cpu={ca[i]} gpu={ga[i]}");
        }
    }

    [Fact]
    public void Ilgpu_Runs_Elementwise_Graph_Through_Same_Seam()
    {
        // y = Relu(a + b) — the same graph the managed CPU engine runs, now on the GPU path.
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Add", "add0", new[] { "a", "b" }, new[] { "sum" }),
                new GraphNode("Relu", "relu0", new[] { "sum" }, new[] { "y" }),
            },
        };

        // preferCpu: deterministic + runs on any machine (ILGPU's CPU accelerator);
        // the identical code path targets CUDA/OpenCL when a GPU is present.
        using var engine = new IlgpuEngine(graph, preferCpu: true);

        var shape = new TensorShape(2, 2);
        Tensor<float> a = Tensor<float>.FromArray(shape, new[] { 1f, -5f, 3f, -2f });
        Tensor<float> b = Tensor<float>.FromArray(shape, new[] { 0.5f, 1f, -10f, 4f });

        IReadOnlyDictionary<string, NamedTensor> outputs = engine.Run(new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", a),
            ["b"] = new NamedTensor("b", b),
        });

        Assert.Equal(new[] { 1.5f, 0f, 0f, 2f }, outputs["y"].Data.Span.ToArray());
    }

    [Fact]
    public void Ilgpu_Add_Broadcasts_Row_Vector()
    {
        // (2,3) + (3,) -> (2,3). Trailing dims align; the row vector is broadcast over both rows.
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Add", "add0", new[] { "a", "b" }, new[] { "y" }) },
        };

        using var engine = new IlgpuEngine(graph, preferCpu: true);
        IReadOnlyDictionary<string, NamedTensor> outputs = engine.Run(Feeds(
            ("a", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)),
            ("b", T(new[] { 3 }, 10, 20, 30))));

        Assert.Equal(new[] { 2, 3 }, outputs["y"].Data.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 11f, 22f, 33f, 14f, 25f, 36f }, outputs["y"].Data.Span.ToArray());
    }

    [Fact]
    public void Ilgpu_Div_Broadcasts_Column_Vector_Matches_Cpu()
    {
        // (2,3) / (2,1) -> (2,3): each row divided by its own scalar. Compared against the CPU engine.
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Div", "div0", new[] { "a", "b" }, new[] { "y" }) },
        };

        AssertGpuMatchesCpu(graph, Feeds(
            ("a", T(new[] { 2, 3 }, 2, 4, 6, 9, 12, 15)),
            ("b", T(new[] { 2, 1 }, 2, 3))));
    }

    [Fact]
    public void Ilgpu_MatMul_2D_Matches_Cpu()
    {
        // (2,3) x (3,2) -> (2,2).
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "y" }) },
        };

        var feeds = Feeds(
            ("a", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)),
            ("b", T(new[] { 3, 2 }, 1, 0, 0, 1, 1, 1)));

        AssertGpuMatchesCpu(graph, feeds);

        using var engine = new IlgpuEngine(graph, preferCpu: true);
        Tensor<float> y = engine.Run(feeds)["y"].Data;
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 4f, 5f, 10f, 11f }, y.Span.ToArray());
    }

    [Fact]
    public void Ilgpu_MatMul_Batched_3D_Matches_Cpu()
    {
        // (2,2,3) x (2,3,2) -> (2,2,2): per-batch matmul, no broadcasting needed.
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "y" }) },
        };

        var a = T(new[] { 2, 2, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        var b = T(new[] { 2, 3, 2 }, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1);
        AssertGpuMatchesCpu(graph, Feeds(("a", a), ("b", b)));
    }

    [Fact]
    public void Ilgpu_MatMul_Batch_Broadcast_Matches_Cpu()
    {
        // (3,2,4) x (4,5) -> (3,2,5): the 2-D operand is broadcast across the leading batch dimension.
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "y" }) },
        };

        var rnd = new Random(7);
        float[] aData = Enumerable.Range(0, 3 * 2 * 4).Select(_ => (float)rnd.NextDouble()).ToArray();
        float[] bData = Enumerable.Range(0, 4 * 5).Select(_ => (float)rnd.NextDouble()).ToArray();
        AssertGpuMatchesCpu(graph, Feeds(
            ("a", T(new[] { 3, 2, 4 }, aData)),
            ("b", T(new[] { 4, 5 }, bData))));
    }

    [Fact]
    public void Ilgpu_Conv2D_Stride_Pad_Bias_Matches_Cpu()
    {
        // Single 3x3 kernel over a 1x1x5x5 input, stride 2, pad 1, with bias. Compared against the CPU engine.
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
        AssertGpuMatchesCpu(graph, Feeds(("x", T(new[] { 1, 1, 5, 5 }, xData))));
    }

    [Fact]
    public void Ilgpu_Conv2D_Grouped_Matches_Cpu()
    {
        // 2 groups: input 1x4x4x4, weight (4,2,3,3) -> 4 output channels, group=2. Compared against the CPU engine.
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Conv", "c", new[] { "x", "w" }, new[] { "y" },
                    new Dictionary<string, object> { ["group"] = 2L, ["pads"] = new long[] { 1, 1, 1, 1 } }),
            },
            Initializers = new Dictionary<string, Tensor> { ["w"] = RandTensor(new[] { 4, 2, 3, 3 }, 11) },
        };

        AssertGpuMatchesCpu(graph, Feeds(("x", RandTensor(new[] { 1, 4, 4, 4 }, 12))));
    }

    private static Tensor<float> RandTensor(int[] dims, int seed)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (acc, d) => acc * d);
        float[] data = Enumerable.Range(0, n).Select(_ => (float)(rnd.NextDouble() * 2 - 1)).ToArray();
        return T(dims, data);
    }
}
