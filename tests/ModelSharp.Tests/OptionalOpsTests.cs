using System.Collections.Generic;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Tests for the ONNX <c>Optional</c> / <c>OptionalGetElement</c> / <c>OptionalHasElement</c> ops.
/// Optional values live only between nodes; each graph reduces to a tensor output.
/// </summary>
public class OptionalOpsTests
{
    private static Tensor<float> Vec(params float[] v) => new(new TensorShape(v.Length), v);

    private static IReadOnlyDictionary<string, NamedTensor> Run(ModelGraph g, Dictionary<string, NamedTensor> feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        return engine.Run(feeds);
    }

    [Fact]
    public void Present_Optional_HasElement_True_And_GetElement_RoundTrips()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "has", "elem" },
            Nodes = new[]
            {
                new GraphNode("Optional", "opt", new[] { "x" }, new[] { "o" }),
                new GraphNode("OptionalHasElement", "ohe", new[] { "o" }, new[] { "has" }),
                new GraphNode("OptionalGetElement", "oge", new[] { "o" }, new[] { "elem" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["x"] = new NamedTensor("x", Vec(1f, 2f, 3f)),
        });

        Assert.True(outputs["has"].Tensor.AsBool().Span[0]);
        Assert.Equal(new[] { 1f, 2f, 3f }, outputs["elem"].Data.Span.ToArray());
    }

    [Fact]
    public void Absent_Optional_HasElement_False()
    {
        // Optional with no input -> none; OptionalHasElement -> false.
        var graph = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "has" },
            Nodes = new[]
            {
                new GraphNode("Optional", "opt", new string[0], new[] { "o" },
                    new Dictionary<string, object> { ["type"] = "tensor(float)" }),
                new GraphNode("OptionalHasElement", "ohe", new[] { "o" }, new[] { "has" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>());
        Assert.False(outputs["has"].Tensor.AsBool().Span[0]);
    }

    [Fact]
    public void GetElement_On_Absent_Optional_Throws()
    {
        var graph = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "elem" },
            Nodes = new[]
            {
                new GraphNode("Optional", "opt", new string[0], new[] { "o" }),
                new GraphNode("OptionalGetElement", "oge", new[] { "o" }, new[] { "elem" }),
            },
        };

        Assert.ThrowsAny<ModelSharpException>(() =>
            Run(graph, new Dictionary<string, NamedTensor>()));
    }

    [Fact]
    public void Optional_Wrapping_A_Sequence_HasElement_True()
    {
        // SequenceConstruct -> Optional(seq) -> OptionalHasElement = true,
        // OptionalGetElement -> sequence -> SequenceLength = 2.
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "has", "len" },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "sc", new[] { "a", "b" }, new[] { "seq" }),
                new GraphNode("Optional", "opt", new[] { "seq" }, new[] { "o" }),
                new GraphNode("OptionalHasElement", "ohe", new[] { "o" }, new[] { "has" }),
                new GraphNode("OptionalGetElement", "oge", new[] { "o" }, new[] { "seq2" }),
                new GraphNode("SequenceLength", "sl", new[] { "seq2" }, new[] { "len" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", Vec(1f)),
            ["b"] = new NamedTensor("b", Vec(2f)),
        });

        Assert.True(outputs["has"].Tensor.AsBool().Span[0]);
        Assert.Equal(2L, outputs["len"].Tensor.AsInt64().Span[0]);
    }
}
