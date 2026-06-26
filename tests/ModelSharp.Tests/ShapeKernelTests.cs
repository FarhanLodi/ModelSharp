using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class ShapeKernelTests
{
    private static Tensor RunShape(ModelGraph g, params (string name, Tensor t)[] feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        Dictionary<string, NamedTensor> dict = feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));
        return engine.Run(dict).Values.First().Tensor;
    }

    private static Tensor<float> T(int[] dims) =>
        Tensor<float>.FromArray(new TensorShape(dims), new float[new TensorShape(dims).Length]);

    [Fact]
    public void Shape_Returns_Full_Shape_As_1D_Int64()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Shape", "s", new[] { "x" }, new[] { "y" }) },
        };
        Tensor y = RunShape(g, ("x", T(new[] { 2, 3, 4 })));
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 3 }, y.Shape.Dimensions.ToArray());   // 1-D, length == rank
        Assert.Equal(new long[] { 2, 3, 4 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Shape_With_Start_And_End_Slices_The_Shape()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Shape", "s", new[] { "x" }, new[] { "y" },
                    new Dictionary<string, object> { ["start"] = 1L, ["end"] = 3L }),
            },
        };
        Tensor y = RunShape(g, ("x", T(new[] { 2, 3, 4, 5 })));
        Assert.Equal(new[] { 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 3, 4 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Shape_With_Negative_Start_Counts_From_The_End()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Shape", "s", new[] { "x" }, new[] { "y" },
                    new Dictionary<string, object> { ["start"] = -2L }),
            },
        };
        Tensor y = RunShape(g, ("x", T(new[] { 2, 3, 4, 5 })));
        Assert.Equal(new long[] { 4, 5 }, y.AsInt64().Span.ToArray());
    }
}
