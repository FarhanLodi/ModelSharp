using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class KernelTests
{
    private static Tensor<float> Run(ModelGraph g, params (string name, Tensor<float> t)[] feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        Dictionary<string, NamedTensor> dict = feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));
        return engine.Run(dict).Values.First().Data;
    }

    private static Tensor<float> T(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    [Fact]
    public void Conv_2x2_Valid()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Conv", "c", new[] { "x", "w" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["w"] = T(new[] { 1, 1, 2, 2 }, 1, 0, 0, 0) },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 1, 1, 3, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
        Assert.Equal(new[] { 1f, 2f, 4f, 5f }, y.Span.ToArray());
    }

    [Fact]
    public void MaxPool_2x2_Stride2()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("MaxPool", "p", new[] { "x" }, new[] { "y" },
                    new Dictionary<string, object> { ["kernel_shape"] = new long[] { 2, 2 }, ["strides"] = new long[] { 2, 2 } }),
            },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 1, 1, 4, 4 }, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)));
        Assert.Equal(new[] { 5f, 7f, 13f, 15f }, y.Span.ToArray());
    }

    [Fact]
    public void MatMul_2x3_Times_3x2()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("MatMul", "m", new[] { "a", "b" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["b"] = T(new[] { 3, 2 }, 1, 0, 0, 1, 1, 1) },
        };
        Tensor<float> y = Run(g, ("a", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)));
        Assert.Equal(new[] { 4f, 5f, 10f, 11f }, y.Span.ToArray());
    }

    [Fact]
    public void Gemm_With_Broadcast_Bias()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Gemm", "g", new[] { "a", "b", "c" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["b"] = T(new[] { 2, 2 }, 1, 2, 3, 4),
                ["c"] = T(new[] { 2 }, 10, 20),
            },
        };
        Tensor<float> y = Run(g, ("a", T(new[] { 2, 2 }, 1, 0, 0, 1)));   // A = identity
        Assert.Equal(new[] { 11f, 22f, 13f, 24f }, y.Span.ToArray());
    }

    [Fact]
    public void Softmax_Of_Equal_Logits_Is_Uniform()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Softmax", "s", new[] { "x" }, new[] { "y" }) },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 1, 3 }, 0, 0, 0)));
        Assert.All(y.Span.ToArray(), v => Assert.True(MathF.Abs(v - 1f / 3f) < 1e-6f));
    }

    [Fact]
    public void Add_Broadcasts_Per_Column_Bias()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Add", "a0", new[] { "a", "b" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["b"] = T(new[] { 2 }, 10, 20) },
        };
        Tensor<float> y = Run(g, ("a", T(new[] { 2, 2 }, 1, 2, 3, 4)));
        Assert.Equal(new[] { 11f, 22f, 13f, 24f }, y.Span.ToArray());
    }

    [Fact]
    public void Reshape_To_Explicit_Shape_Preserves_Data()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Reshape", "r", new[] { "x", "shape" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["shape"] = T(new[] { 2 }, 3, 2) },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 1, 6 }, 0, 1, 2, 3, 4, 5)));
        Assert.Equal(new[] { 3, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 0f, 1f, 2f, 3f, 4f, 5f }, y.Span.ToArray());
    }

    [Fact]
    public void GlobalAveragePool_Computes_Mean()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("GlobalAveragePool", "p", new[] { "x" }, new[] { "y" }) },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4)));
        Assert.Equal(new[] { 2.5f }, y.Span.ToArray());
    }
}
