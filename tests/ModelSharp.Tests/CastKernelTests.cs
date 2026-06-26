using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class CastKernelTests
{
    // Runs a one-node Cast graph and returns the produced (dtype-carrying) tensor.
    private static Tensor Run(long to, Tensor input)
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Cast", "c", new[] { "x" }, new[] { "y" },
                    new Dictionary<string, object> { ["to"] = to }),
            },
        };
        using var engine = new ManagedCpuEngine(g);
        var feeds = new Dictionary<string, NamedTensor> { ["x"] = new NamedTensor("x", input) };
        return engine.Run(feeds).Values.First().Tensor;
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> L(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    [Fact]
    public void Float_To_Int64_Truncates_Toward_Zero()
    {
        Tensor y = Run(7, F(new[] { 4 }, 1.9f, -1.9f, 2.0f, 0.0f));
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 4 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 1, -1, 2, 0 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Float_To_Int32_Truncates_Toward_Zero()
    {
        Tensor y = Run(6, F(new[] { 2, 2 }, 2.7f, -2.7f, 0.4f, -0.4f));
        Assert.Equal(ElementType.Int32, y.Dtype);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 2, -2, 0, 0 }, y.AsInt32().Span.ToArray());
    }

    [Fact]
    public void Float_To_Bool_NonZero_Is_True()
    {
        Tensor y = Run(9, F(new[] { 4 }, 0.0f, 3.5f, -0.0f, -2.0f));
        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { false, true, false, true }, y.AsBool().Span.ToArray());
    }

    [Fact]
    public void Int64_To_Float32_Roundtrips_Values()
    {
        Tensor y = Run(1, L(new[] { 3 }, 1, 2, 3));
        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 1f, 2f, 3f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Same_Dtype_Cast_Is_Passthrough()
    {
        Tensor input = F(new[] { 3 }, 1f, 2f, 3f);
        Tensor y = Run(1, input); // float -> float (to=FLOAT)
        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Same(input, y);
    }
}
