using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class SliceKernelTests
{
    private static Tensor RunSlice(ModelGraph g, params (string name, Tensor t)[] feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        Dictionary<string, NamedTensor> dict = feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));
        return engine.Run(dict).Values.First().Tensor;
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> I64(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    [Fact]
    public void Slice_2D_PositiveRanges_With_Explicit_Axes_And_Steps()
    {
        // 3x4 matrix, values 0..11 row-major.
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Slice", "s", new[] { "x", "starts", "ends", "axes", "steps" }, new[] { "y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["starts"] = I64(new[] { 2 }, 1, 0),  // rows from 1, cols from 0
                ["ends"] = I64(new[] { 2 }, 3, 2),    // rows to 3 (exclusive), cols to 2 (exclusive)
                ["axes"] = I64(new[] { 2 }, 0, 1),
                ["steps"] = I64(new[] { 2 }, 1, 1),
            },
        };
        Tensor y = RunSlice(g, ("x", F(new[] { 3, 4 }, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)));

        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        // rows 1,2 x cols 0,1 -> [4,5 ; 8,9]
        Assert.Equal(new[] { 4f, 5f, 8f, 9f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Slice_DefaultAxes_And_NegativeIndices()
    {
        // axes input omitted entirely -> defaults to [0,1]; negative end on axis 1.
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Slice", "s", new[] { "x", "starts", "ends" }, new[] { "y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["starts"] = I64(new[] { 2 }, 0, 1),
                ["ends"] = I64(new[] { 2 }, 2, -1),   // axis1: 1 .. (4-1)=3 -> cols 1,2
            },
        };
        Tensor y = RunSlice(g, ("x", F(new[] { 2, 4 }, 0, 1, 2, 3, 4, 5, 6, 7)));

        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        // rows 0,1 x cols 1,2 -> [1,2 ; 5,6]
        Assert.Equal(new[] { 1f, 2f, 5f, 6f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Slice_NegativeStep_Reverses_Axis()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Slice", "s", new[] { "x", "starts", "ends", "axes", "steps" }, new[] { "y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["starts"] = I64(new[] { 1 }, 3),
                ["ends"] = I64(new[] { 1 }, -5),   // < lower for step<0 -> clamps to -1 (include index 0)
                ["axes"] = I64(new[] { 1 }, 1),
                ["steps"] = I64(new[] { 1 }, -1),
            },
        };
        Tensor y = RunSlice(g, ("x", F(new[] { 1, 4 }, 10, 20, 30, 40)));

        Assert.Equal(new[] { 1, 4 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 40f, 30f, 20f, 10f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Slice_Preserves_Int64_Dtype()
    {
        var g = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Slice", "s", new[] { "x", "starts", "ends" }, new[] { "y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["starts"] = I64(new[] { 1 }, 1),
                ["ends"] = I64(new[] { 1 }, 3),
            },
        };
        // int64 data fed as the graph input; output must stay Int64.
        Tensor y = RunSlice(g, ("x", I64(new[] { 4 }, 5, 6, 7, 8)));

        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 6, 7 }, y.AsInt64().Span.ToArray());
    }
}
