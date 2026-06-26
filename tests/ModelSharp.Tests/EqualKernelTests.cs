using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Logical;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class EqualKernelTests
{
    // Equal isn't registered in CreateDefault() yet (the integration phase does that),
    // so register it explicitly on a default registry for these focused tests.
    private static KernelRegistry Registry() =>
        KernelRegistry.CreateDefault().Register(new EqualKernel());

    private static Tensor RunEqual(Tensor a, Tensor b)
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Equal", "eq", new[] { "a", "b" }, new[] { "y" }) },
        };
        using var engine = new ManagedCpuEngine(g, Registry());
        var feeds = new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", a),
            ["b"] = new NamedTensor("b", b),
        };
        return engine.Run(feeds).Values.First().Tensor;
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> L(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    [Fact]
    public void Equal_SameShape_Float_ProducesBool()
    {
        Tensor y = RunEqual(F(new[] { 2, 2 }, 1, 2, 3, 4), F(new[] { 2, 2 }, 1, 0, 3, 9));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { true, false, true, false }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Equal_SameShape_Int64()
    {
        Tensor y = RunEqual(L(new[] { 3 }, 1, 2, 3), L(new[] { 3 }, 1, 5, 3));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { true, false, true }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Equal_BroadcastsRowVector()
    {
        // a = [[1,2],[3,4]] compared against b = [1,4] broadcast over both rows.
        Tensor y = RunEqual(F(new[] { 2, 2 }, 1, 2, 3, 4), F(new[] { 2 }, 1, 4));

        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        // row0: [1==1, 2==4] -> [T,F]; row1: [3==1, 4==4] -> [F,T]
        Assert.Equal(new[] { true, false, false, true }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Equal_BroadcastsScalar_Int64()
    {
        Tensor y = RunEqual(L(new[] { 4 }, 7, 2, 7, 3), L(System.Array.Empty<int>(), 7));

        Assert.Equal(new[] { true, false, true, false }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Equal_Boolean_Inputs()
    {
        Tensor a = Tensor<bool>.FromArray(new TensorShape(new[] { 3 }), new[] { true, false, true });
        Tensor b = Tensor<bool>.FromArray(new TensorShape(new[] { 3 }), new[] { true, true, false });

        Tensor y = RunEqual(a, b);

        Assert.Equal(new[] { true, false, false }, y.AsBool().Span.ToArray());
    }
}
