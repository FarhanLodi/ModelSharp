using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Shape;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class ConstantOfShapeKernelTests
{
    // ConstantOfShape isn't in the default registry yet (the integration phase adds it),
    // so register it explicitly on top of the default set for these tests.
    private static readonly KernelRegistry Registry =
        KernelRegistry.CreateDefault().Register(new ConstantOfShapeKernel());

    // Runs a graph whose shape input is a baked-in int64 initializer; returns the raw output tensor.
    private static Tensor Run(ModelGraph g)
    {
        using var engine = new ManagedCpuEngine(g, Registry);
        return engine.Run(new Dictionary<string, NamedTensor>()).Values.First().Tensor;
    }

    private static ModelGraph Graph(long[] shape, IReadOnlyDictionary<string, object>? attrs) =>
        new()
        {
            Inputs = Array.Empty<string>(),
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("ConstantOfShape", "cos", new[] { "shape" }, new[] { "y" }, attrs) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["shape"] = Tensor<long>.FromArray(new TensorShape(shape.Length), shape),
            },
        };

    private static Dictionary<string, object> ValueAttr(Tensor value) =>
        new() { ["value"] = value };

    [Fact]
    public void Default_Value_Is_Float32_Zeros()
    {
        Tensor y = Run(Graph(new long[] { 2, 3 }, attrs: null));

        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        Tensor<float> f = y.AsFloat();
        Assert.Equal(6, f.Span.Length);
        Assert.All(f.Span.ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Fills_With_Float_Value()
    {
        var attrs = ValueAttr(Tensor<float>.FromArray(new TensorShape(1), new[] { 7.5f }));

        Tensor y = Run(Graph(new long[] { 2, 2 }, attrs));

        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 7.5f, 7.5f, 7.5f, 7.5f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Fills_With_Int64_Value_And_Preserves_Dtype()
    {
        var attrs = ValueAttr(Tensor<long>.FromArray(new TensorShape(1), new[] { 5L }));

        Tensor y = Run(Graph(new long[] { 3 }, attrs));

        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 5L, 5L, 5L }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Fills_With_Bool_Value()
    {
        var attrs = ValueAttr(Tensor<bool>.FromArray(new TensorShape(1), new[] { true }));

        Tensor y = Run(Graph(new long[] { 2 }, attrs));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { true, true }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Empty_Shape_Produces_Scalar()
    {
        var attrs = ValueAttr(Tensor<float>.FromArray(new TensorShape(1), new[] { 3f }));

        Tensor y = Run(Graph(Array.Empty<long>(), attrs));

        Assert.Equal(0, y.Shape.Rank);
        Assert.Equal(1, y.AsFloat().Span.Length);
        Assert.Equal(3f, y.AsFloat().Span[0]);
    }
}
