using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class WhereKernelTests
{
    /// <summary>Runs a single-node Where graph and returns the dtype-preserving output tensor.</summary>
    private static Tensor Run(ModelGraph g, params (string name, Tensor t)[] feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        Dictionary<string, NamedTensor> dict = feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));
        return engine.Run(dict).Values.First().Tensor;
    }

    private static ModelGraph WhereGraph() => new ModelGraph
    {
        Inputs = new[] { "c", "x", "y" },
        Outputs = new[] { "out" },
        Nodes = new[] { new GraphNode("Where", "w", new[] { "c", "x", "y" }, new[] { "out" }) },
    };

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<bool> B(int[] dims, params bool[] data) =>
        Tensor<bool>.FromArray(new TensorShape(dims), data);

    [Fact]
    public void Selects_Elementwise_Float_SameShape()
    {
        Tensor result = Run(WhereGraph(),
            ("c", B(new[] { 3 }, true, false, true)),
            ("x", F(new[] { 3 }, 1, 2, 3)),
            ("y", F(new[] { 3 }, 10, 20, 30)));

        Assert.Equal(new[] { 1f, 20f, 3f }, result.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Broadcasts_Condition_Across_Columns()
    {
        // condition [2,1] broadcasts over the 2 columns of x/y [2,2].
        Tensor result = Run(WhereGraph(),
            ("c", B(new[] { 2, 1 }, true, false)),
            ("x", F(new[] { 2, 2 }, 1, 2, 3, 4)),
            ("y", F(new[] { 2, 2 }, 10, 20, 30, 40)));

        Assert.Equal(new[] { 2, 2 }, result.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1f, 2f, 30f, 40f }, result.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Broadcasts_All_Three_Inputs()
    {
        // c [2,1], x [1,3] scalar-ish rows, y [1] scalar -> output [2,3].
        Tensor result = Run(WhereGraph(),
            ("c", B(new[] { 2, 1 }, true, false)),
            ("x", F(new[] { 1, 3 }, 7, 8, 9)),
            ("y", F(new[] { 1 }, -1)));

        Assert.Equal(new[] { 2, 3 }, result.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 7f, 8f, 9f, -1f, -1f, -1f }, result.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Preserves_Int64_Output_Dtype()
    {
        Tensor result = Run(WhereGraph(),
            ("c", B(new[] { 2 }, true, false)),
            ("x", Tensor<long>.FromArray(new TensorShape(2), new long[] { 5, 6 })),
            ("y", Tensor<long>.FromArray(new TensorShape(2), new long[] { 7, 8 })));

        Assert.Equal(ElementType.Int64, result.Dtype);
        Assert.Equal(new long[] { 5, 8 }, result.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Preserves_Bool_Output_Dtype()
    {
        Tensor result = Run(WhereGraph(),
            ("c", B(new[] { 2 }, true, false)),
            ("x", B(new[] { 2 }, true, true)),
            ("y", B(new[] { 2 }, false, false)));

        Assert.Equal(ElementType.Boolean, result.Dtype);
        Assert.Equal(new[] { true, false }, result.AsBool().Span.ToArray());
    }

    [Fact]
    public void Scalar_Condition_Selects_Whole_Tensor()
    {
        // rank-0 condition broadcasts over everything.
        Tensor result = Run(WhereGraph(),
            ("c", B(System.Array.Empty<int>(), false)),
            ("x", F(new[] { 2 }, 1, 2)),
            ("y", F(new[] { 2 }, 9, 8)));

        Assert.Equal(new[] { 9f, 8f }, result.AsFloat().Span.ToArray());
    }
}
