using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class ExpandKernelTests
{
    private static Tensor Run(ModelGraph g, params (string name, Tensor t)[] feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        Dictionary<string, NamedTensor> dict = feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));
        return engine.Run(dict).Values.First().Tensor;
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> L(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    private static ModelGraph Graph(Tensor shapeTensor) => new()
    {
        Inputs = new[] { "x" },
        Outputs = new[] { "y" },
        Nodes = new[] { new GraphNode("Expand", "e", new[] { "x", "shape" }, new[] { "y" }) },
        Initializers = new Dictionary<string, Tensor> { ["shape"] = shapeTensor },
    };

    [Fact]
    public void Expand_Float_Broadcasts_Trailing_Dim()
    {
        // [2,1] -> requested [2,3] = [2,3]; each row value repeated 3 times.
        Tensor y = Run(Graph(L(new[] { 2 }, 2, 3)), ("x", F(new[] { 2, 1 }, 10, 20)));

        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 10f, 10f, 10f, 20f, 20f, 20f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Expand_Float_Bidirectional_Broadcast_Adds_Rank()
    {
        // ONNX example: data [3,1] with requested [2,1,6] broadcasts to [2,3,6].
        Tensor y = Run(Graph(L(new[] { 3 }, 2, 1, 6)), ("x", F(new[] { 3, 1 }, 1, 2, 3)));

        Assert.Equal(new[] { 2, 3, 6 }, y.Shape.Dimensions.ToArray());

        var expected = new float[36];
        int k = 0;
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 6; j++)
                    expected[k++] = i + 1; // data[i][0]
        Assert.Equal(expected, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Expand_Int64_Preserves_Dtype()
    {
        // [1,2] -> requested [3,2] = [3,2]; each row is [7, 8].
        Tensor y = Run(Graph(L(new[] { 2 }, 3, 2)), ("x", L(new[] { 1, 2 }, 7, 8)));

        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 3, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 7, 8, 7, 8, 7, 8 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Expand_Bool_Preserves_Dtype()
    {
        // scalar-ish [1] -> requested [3] = [3]; all true.
        var x = Tensor<bool>.FromArray(new TensorShape(new[] { 1 }), new[] { true });
        Tensor y = Run(Graph(L(new[] { 1 }, 3)), ("x", x));

        Assert.Equal(ElementType.Boolean, y.Dtype);
        Assert.Equal(new[] { 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { true, true, true }, y.AsBool().Span.ToArray());
    }
}
