using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Shape;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class ConstantKernelTests
{
    // Constant has no graph inputs, so feeds are empty. We register the kernel
    // explicitly (the integration phase wires it into CreateDefault separately)
    // and return the dtype-carrying payload so non-float outputs can be read.
    private static Tensor Run(GraphNode node)
    {
        var g = new ModelGraph
        {
            Inputs = System.Array.Empty<string>(),
            Outputs = node.Outputs.ToArray(),
            Nodes = new[] { node },
        };
        KernelRegistry registry = KernelRegistry.CreateDefault().Register(new ConstantKernel());
        using var engine = new ManagedCpuEngine(g, registry);
        return engine.Run(new Dictionary<string, NamedTensor>()).Values.First().Tensor;
    }

    private static GraphNode ConstNode(string attrName, object attrValue) =>
        new GraphNode("Constant", "k", System.Array.Empty<string>(), new[] { "y" },
            new Dictionary<string, object> { [attrName] = attrValue });

    [Fact]
    public void Constant_Value_FloatTensor_PreservesDataAndShape()
    {
        Tensor<float> value = Tensor<float>.FromArray(new TensorShape(2, 2), new[] { 1f, 2f, 3f, 4f });
        Tensor y = Run(ConstNode("value", value));

        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Constant_Value_Int64Tensor_PreservesDtype()
    {
        Tensor<long> value = Tensor<long>.FromArray(new TensorShape(3), new long[] { 10, 20, 30 });
        Tensor y = Run(ConstNode("value", value));

        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 10, 20, 30 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Constant_ValueInt_ProducesScalarInt64()
    {
        Tensor y = Run(ConstNode("value_int", 7L));

        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(0, y.Shape.Rank);            // scalar (rank-0)
        Assert.Equal(1, y.Length);
        Assert.Equal(new long[] { 7 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Constant_ValueFloat_ProducesScalarFloat32()
    {
        Tensor y = Run(ConstNode("value_float", 2.5f));

        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(0, y.Shape.Rank);
        Assert.Equal(new[] { 2.5f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Constant_ValueInts_Produces1DInt64()
    {
        Tensor y = Run(ConstNode("value_ints", new long[] { 4, 5, 6 }));

        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 4, 5, 6 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Constant_ValueFloats_Produces1DFloat32()
    {
        Tensor y = Run(ConstNode("value_floats", new[] { 1.5f, -2f }));

        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1.5f, -2f }, y.AsFloat().Span.ToArray());
    }
}
