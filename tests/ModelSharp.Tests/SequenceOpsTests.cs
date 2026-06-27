using System.Collections.Generic;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Tests for the ONNX <c>Sequence*</c> op family. Sequences are non-tensor values that live only
/// on the wire between nodes, so each graph ends in a node that reduces the sequence back to a
/// tensor (SequenceAt / SequenceLength / ConcatFromSequence) which is then a graph output.
/// </summary>
public class SequenceOpsTests
{
    private static Tensor<float> Vec(params float[] v) => new(new TensorShape(v.Length), v);
    private static Tensor<float> Mat(int r, int c, params float[] v) => new(new TensorShape(r, c), v);
    private static Tensor<long> LongScalar(long v) => new(new TensorShape(), new[] { v });

    private static IReadOnlyDictionary<string, NamedTensor> Run(ModelGraph g, Dictionary<string, NamedTensor> feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        return engine.Run(feeds);
    }

    [Fact]
    public void Construct_Then_At_And_Length_RoundTrip()
    {
        // seq = [a, b, c]; out0 = seq[1]; out1 = len(seq)
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b", "c", "idx" },
            Outputs = new[] { "elem", "len" },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "sc", new[] { "a", "b", "c" }, new[] { "seq" }),
                new GraphNode("SequenceAt", "sa", new[] { "seq", "idx" }, new[] { "elem" }),
                new GraphNode("SequenceLength", "sl", new[] { "seq" }, new[] { "len" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", Vec(1f, 2f)),
            ["b"] = new NamedTensor("b", Vec(3f, 4f)),
            ["c"] = new NamedTensor("c", Vec(5f, 6f)),
            ["idx"] = new NamedTensor("idx", LongScalar(1)),
        });

        Assert.Equal(new[] { 3f, 4f }, outputs["elem"].Data.Span.ToArray());
        Assert.Equal(3L, outputs["len"].Tensor.AsInt64().Span[0]);
    }

    [Fact]
    public void At_Negative_Index_Counts_From_End()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b", "idx" },
            Outputs = new[] { "elem" },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "sc", new[] { "a", "b" }, new[] { "seq" }),
                new GraphNode("SequenceAt", "sa", new[] { "seq", "idx" }, new[] { "elem" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", Vec(1f)),
            ["b"] = new NamedTensor("b", Vec(9f)),
            ["idx"] = new NamedTensor("idx", LongScalar(-1)),
        });
        Assert.Equal(9f, outputs["elem"].Data.Span[0]);
    }

    [Fact]
    public void Empty_Insert_Erase_Then_Length()
    {
        // start empty; insert a (append), insert b (append) -> [a,b]; erase last -> [a]; len = 1
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "len", "first" },
            Nodes = new[]
            {
                new GraphNode("SequenceEmpty", "se", new string[0], new[] { "s0" }),
                new GraphNode("SequenceInsert", "i1", new[] { "s0", "a" }, new[] { "s1" }),
                new GraphNode("SequenceInsert", "i2", new[] { "s1", "b" }, new[] { "s2" }),
                new GraphNode("SequenceErase", "er", new[] { "s2" }, new[] { "s3" }),
                new GraphNode("SequenceLength", "sl", new[] { "s3" }, new[] { "len" }),
                new GraphNode("SequenceAt", "sa", new[] { "s3", "zero" }, new[] { "first" }),
            },
            Initializers = new Dictionary<string, Tensor> { ["zero"] = LongScalar(0) },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", Vec(7f)),
            ["b"] = new NamedTensor("b", Vec(8f)),
        });
        Assert.Equal(1L, outputs["len"].Tensor.AsInt64().Span[0]);
        Assert.Equal(7f, outputs["first"].Data.Span[0]);
    }

    [Fact]
    public void Insert_At_Position_Zero()
    {
        // [a] then insert b at 0 -> [b, a]; SequenceAt 0 -> b
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "first" },
            Initializers = new Dictionary<string, Tensor> { ["zero"] = LongScalar(0) },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "sc", new[] { "a" }, new[] { "s0" }),
                new GraphNode("SequenceInsert", "i", new[] { "s0", "b", "zero" }, new[] { "s1" }),
                new GraphNode("SequenceAt", "sa", new[] { "s1", "zero" }, new[] { "first" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", Vec(1f)),
            ["b"] = new NamedTensor("b", Vec(2f)),
        });
        Assert.Equal(2f, outputs["first"].Data.Span[0]);
    }

    [Fact]
    public void Split_To_Sequence_Then_Concat_Is_Inverse()
    {
        // data is [4,2]; split along axis 0 into 4 length-1 chunks (keepdims), then concat back.
        var data = Mat(4, 2, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
        var graph = new ModelGraph
        {
            Inputs = new[] { "data" },
            Outputs = new[] { "round", "len" },
            Nodes = new[]
            {
                new GraphNode("SplitToSequence", "sts", new[] { "data" }, new[] { "seq" },
                    new Dictionary<string, object> { ["axis"] = (long)0 }),
                new GraphNode("SequenceLength", "sl", new[] { "seq" }, new[] { "len" }),
                new GraphNode("ConcatFromSequence", "cfs", new[] { "seq" }, new[] { "round" },
                    new Dictionary<string, object> { ["axis"] = (long)0 }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["data"] = new NamedTensor("data", data),
        });

        Assert.Equal(4L, outputs["len"].Tensor.AsInt64().Span[0]);
        Tensor<float> round = outputs["round"].Data;
        Assert.Equal(new[] { 4, 2 }, round.Shape.Dimensions.ToArray());
        Assert.Equal(data.Span.ToArray(), round.Span.ToArray());
    }

    [Fact]
    public void Split_With_Explicit_Sizes()
    {
        // data [5,1] split into chunks [2,3] along axis 0; concat back recovers it.
        var data = new Tensor<float>(new TensorShape(5, 1), new[] { 1f, 2f, 3f, 4f, 5f });
        var graph = new ModelGraph
        {
            Inputs = new[] { "data", "split" },
            Outputs = new[] { "round", "len" },
            Nodes = new[]
            {
                new GraphNode("SplitToSequence", "sts", new[] { "data", "split" }, new[] { "seq" },
                    new Dictionary<string, object> { ["axis"] = (long)0 }),
                new GraphNode("SequenceLength", "sl", new[] { "seq" }, new[] { "len" }),
                new GraphNode("ConcatFromSequence", "cfs", new[] { "seq" }, new[] { "round" },
                    new Dictionary<string, object> { ["axis"] = (long)0 }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["data"] = new NamedTensor("data", data),
            ["split"] = new NamedTensor("split", new Tensor<long>(new TensorShape(2), new long[] { 2, 3 })),
        });

        Assert.Equal(2L, outputs["len"].Tensor.AsInt64().Span[0]);
        Assert.Equal(data.Span.ToArray(), outputs["round"].Data.Span.ToArray());
    }

    [Fact]
    public void Concat_From_Sequence_New_Axis_Stacks()
    {
        // Split [3,2] into 3 length-1 chunks WITHOUT keepdims -> three [2] vectors;
        // ConcatFromSequence new_axis stacks them back to [3,2].
        var data = Mat(3, 2, 1f, 2f, 3f, 4f, 5f, 6f);
        var graph = new ModelGraph
        {
            Inputs = new[] { "data" },
            Outputs = new[] { "stacked" },
            Nodes = new[]
            {
                new GraphNode("SplitToSequence", "sts", new[] { "data" }, new[] { "seq" },
                    new Dictionary<string, object> { ["axis"] = (long)0, ["keepdims"] = (long)0 }),
                new GraphNode("ConcatFromSequence", "cfs", new[] { "seq" }, new[] { "stacked" },
                    new Dictionary<string, object> { ["axis"] = (long)0, ["new_axis"] = (long)1 }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["data"] = new NamedTensor("data", data),
        });

        Tensor<float> stacked = outputs["stacked"].Data;
        Assert.Equal(new[] { 3, 2 }, stacked.Shape.Dimensions.ToArray());
        Assert.Equal(data.Span.ToArray(), stacked.Span.ToArray());
    }
}
