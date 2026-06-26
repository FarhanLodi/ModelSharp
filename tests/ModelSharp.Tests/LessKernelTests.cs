using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class LessKernelTests
{
    private static Tensor RunLess(ModelGraph g, params (string name, Tensor t)[] feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        Dictionary<string, NamedTensor> dict = feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));
        return engine.Run(dict).Values.First().Tensor;
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> L(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    private static GraphNode LessNode() =>
        new GraphNode("Less", "lt", new[] { "a", "b" }, new[] { "y" });

    [Fact]
    public void Less_Float_EqualShapes_Elementwise()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { LessNode() },
        };
        // a:   1  5  3  4
        // b:   2  2  3  9
        // a<b: T  F  F  T
        Tensor y = RunLess(g,
            ("a", F(new[] { 2, 2 }, 1, 5, 3, 4)),
            ("b", F(new[] { 2, 2 }, 2, 2, 3, 9)));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { true, false, false, true }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Less_Float_Broadcasts_RowVector()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { LessNode() },
        };
        // a is [2,3], b is [3] broadcast across rows; threshold per column = (3, 3, 3)
        Tensor y = RunLess(g,
            ("a", F(new[] { 2, 3 }, 0, 1, 2, 3, 4, 5)),
            ("b", F(new[] { 3 }, 3, 3, 3)));

        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { true, true, true, false, false, false }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Less_Int64_EqualShapes_Elementwise()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { LessNode() },
        };
        // a:   1  2  3
        // b:   3  2  1
        // a<b: T  F  F
        Tensor y = RunLess(g,
            ("a", L(new[] { 3 }, 1, 2, 3)),
            ("b", L(new[] { 3 }, 3, 2, 1)));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { true, false, false }, y.AsBool().Span.ToArray());
    }
}
