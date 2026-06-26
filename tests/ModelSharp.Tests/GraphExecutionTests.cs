using System.Collections.Generic;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class GraphExecutionTests
{
    [Fact]
    public void Runs_Add_Then_Relu()
    {
        // y = Relu(a + b)
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Add", "add0", new[] { "a", "b" }, new[] { "sum" }),
                new GraphNode("Relu", "relu0", new[] { "sum" }, new[] { "y" }),
            },
        };

        using var engine = new ManagedCpuEngine(graph);

        var shape = new TensorShape(2, 2);
        Tensor<float> a = Tensor<float>.FromArray(shape, new[] { 1f, -5f, 3f, -2f });
        Tensor<float> b = Tensor<float>.FromArray(shape, new[] { 0.5f, 1f, -10f, 4f });

        IReadOnlyDictionary<string, NamedTensor> outputs = engine.Run(new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", a),
            ["b"] = new NamedTensor("b", b),
        });

        // a + b = [1.5, -4, -7, 2]  ->  Relu = [1.5, 0, 0, 2]
        Assert.Equal(new[] { 1.5f, 0f, 0f, 2f }, outputs["y"].Data.Span.ToArray());
    }

    [Fact]
    public void Unsupported_Operator_Throws_With_Op_Name()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("NotARealOp", "op0", new[] { "a" }, new[] { "y" }) },
        };

        using var engine = new ManagedCpuEngine(graph);
        var a = new Tensor<float>(new TensorShape(1));

        var ex = Assert.Throws<UnsupportedOperatorException>(() =>
            engine.Run(new Dictionary<string, NamedTensor> { ["a"] = new NamedTensor("a", a) }));
        Assert.Equal("NotARealOp", ex.Operator);
    }
}
