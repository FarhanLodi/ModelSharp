using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class GreaterKernelTests
{
    private static Tensor Run(ModelGraph g, params (string name, Tensor t)[] feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        Dictionary<string, NamedTensor> dict = feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));
        return engine.Run(dict).Values.First().Tensor;
    }

    private static ModelGraph GreaterGraph() => new ModelGraph
    {
        Inputs = new[] { "a", "b" },
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode("Greater", "g", new[] { "a", "b" }, new[] { "y" }) },
        Initializers = new Dictionary<string, Tensor>(),
    };

    private static Tensor<float> Tf(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> Tl(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    [Fact]
    public void Greater_Float_SameShape_Elementwise()
    {
        ModelGraph g = GreaterGraph();
        Tensor y = Run(g,
            ("a", Tf(new[] { 4 }, 1f, 5f, 3f, 2f)),
            ("b", Tf(new[] { 4 }, 2f, 2f, 3f, 4f)));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { false, true, false, false }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Greater_Float_Broadcasts_RowVector()
    {
        // a is [2,2]; b is [2] broadcast across the last axis: y[i,j] = a[i,j] > b[j].
        ModelGraph g = GreaterGraph();
        Tensor y = Run(g,
            ("a", Tf(new[] { 2, 2 }, 1f, 2f, 3f, 4f)),
            ("b", Tf(new[] { 2 }, 2f, 3f)));

        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { false, false, true, true }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Greater_Int64_Elementwise()
    {
        ModelGraph g = GreaterGraph();
        Tensor y = Run(g,
            ("a", Tl(new[] { 3 }, 3L, 1L, 4L)),
            ("b", Tl(new[] { 3 }, 1L, 5L, 4L)));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { true, false, false }, y.AsBool().Span.ToArray());
    }
}
