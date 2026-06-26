using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class TransformerKernelTests
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
    public void LayerNorm_Normalizes_Last_Axis()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("LayerNormalization", "ln", new[] { "x", "scale", "bias" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["scale"] = T(new[] { 4 }, 1, 1, 1, 1),
                ["bias"] = T(new[] { 4 }, 0, 0, 0, 0),
            },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 1, 4 }, 1, 2, 3, 4)));
        float[] a = y.Span.ToArray();
        // mean 2.5, population var 1.25 -> normalized values:
        float inv = 1f / MathF.Sqrt(1.25f + 1e-5f);
        float[] expected = { -1.5f * inv, -0.5f * inv, 0.5f * inv, 1.5f * inv };
        for (int i = 0; i < 4; i++) Assert.True(MathF.Abs(expected[i] - a[i]) < 1e-4f, $"[{i}] {expected[i]} vs {a[i]}");
        Assert.True(MathF.Abs(a.Sum()) < 1e-4f);   // zero mean
    }

    [Fact]
    public void Transpose_2D()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Transpose", "t", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object> { ["perm"] = new long[] { 1, 0 } }) },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 2, 3 }, 0, 1, 2, 3, 4, 5)));
        Assert.Equal(new[] { 3, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 0f, 3f, 1f, 4f, 2f, 5f }, y.Span.ToArray());
    }

    [Fact]
    public void Gather_Rows()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "data" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Gather", "g", new[] { "data", "idx" }, new[] { "y" },
                new Dictionary<string, object> { ["axis"] = 0L }) },
            Initializers = new Dictionary<string, Tensor> { ["idx"] = T(new[] { 2 }, 2, 0) },
        };
        Tensor<float> y = Run(g, ("data", T(new[] { 3, 2 }, 0, 1, 2, 3, 4, 5)));
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 4f, 5f, 0f, 1f }, y.Span.ToArray());
    }

    [Fact]
    public void Gelu_Matches_Reference()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Gelu", "ge", new[] { "x" }, new[] { "y" }) },
        };
        Tensor<float> y = Run(g, ("x", T(new[] { 3 }, -1f, 0f, 1f)));
        float[] a = y.Span.ToArray();
        Assert.Equal(0f, a[1], 5);                 // gelu(0) = 0
        Assert.True(a[2] is > 0.84f and < 0.85f);  // gelu(1) ≈ 0.8413
        Assert.True(a[0] is > -0.16f and < -0.15f); // gelu(-1) ≈ -0.1587
    }

    [Fact]
    public void Sub_Broadcasts()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Sub", "s", new[] { "a", "b" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["b"] = T(new[] { 1 }, 1) },
        };
        Tensor<float> y = Run(g, ("a", T(new[] { 3 }, 5, 6, 7)));
        Assert.Equal(new[] { 4f, 5f, 6f }, y.Span.ToArray());
    }
}
