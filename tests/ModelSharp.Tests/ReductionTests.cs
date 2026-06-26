using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class ReductionTests
{
    private static Tensor<float> Run(ModelGraph g, string feedName, Tensor<float> feed)
    {
        using var engine = new ManagedCpuEngine(g);
        return engine.Run(new Dictionary<string, NamedTensor> { [feedName] = new NamedTensor(feedName, feed) }).Values.First().Data;
    }

    private static Tensor<float> T(int[] dims, params float[] data) => Tensor<float>.FromArray(new TensorShape(dims), data);

    private static ModelGraph Graph(long[] axes, long keepdims) => new ModelGraph
    {
        Inputs = new[] { "x" },
        Outputs = new[] { "y" },
        Nodes = new[]
        {
            new GraphNode("ReduceMean", "r", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object> { ["axes"] = axes, ["keepdims"] = keepdims }),
        },
    };

    [Fact]
    public void ReduceMean_Axis1() =>
        Assert.Equal(new[] { 2f, 5f }, Run(Graph(new long[] { 1 }, 0), "x", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)).Span.ToArray());

    [Fact]
    public void ReduceMean_Axis0() =>
        Assert.Equal(new[] { 2.5f, 3.5f, 4.5f }, Run(Graph(new long[] { 0 }, 0), "x", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)).Span.ToArray());

    [Fact]
    public void ReduceMean_All_KeepDims() =>
        Assert.Equal(new[] { 3.5f }, Run(Graph(Array.Empty<long>(), 1), "x", T(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)).Span.ToArray());
}
