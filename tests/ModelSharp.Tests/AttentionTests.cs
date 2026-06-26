using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class AttentionTests
{
    private static Tensor<float> T(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    /// <summary>
    /// Scaled dot-product self-attention assembled purely from existing ops
    /// (MatMul, Transpose, Mul, Softmax) — proving the transformer core composes
    /// on ModelSharp. With identity projections the result is hand-verifiable.
    /// </summary>
    [Fact]
    public void Single_Head_Self_Attention_Composes_Correctly()
    {
        float scale = 1f / MathF.Sqrt(2f);
        var identity = T(new[] { 2, 2 }, 1, 0, 0, 1);

        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "out" },
            Nodes = new[]
            {
                new GraphNode("MatMul", "q", new[] { "x", "Wq" }, new[] { "Q" }),
                new GraphNode("MatMul", "k", new[] { "x", "Wk" }, new[] { "K" }),
                new GraphNode("MatMul", "v", new[] { "x", "Wv" }, new[] { "V" }),
                new GraphNode("Transpose", "kt", new[] { "K" }, new[] { "Kt" }),
                new GraphNode("MatMul", "scores", new[] { "Q", "Kt" }, new[] { "S" }),
                new GraphNode("Mul", "scale", new[] { "S", "scale" }, new[] { "Ss" }),
                new GraphNode("Softmax", "sm", new[] { "Ss" }, new[] { "A" }),
                new GraphNode("MatMul", "ctx", new[] { "A", "V" }, new[] { "out" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["Wq"] = identity,
                ["Wk"] = identity,
                ["Wv"] = identity,
                ["scale"] = T(new[] { 1 }, scale),
            },
        };

        using var engine = new ManagedCpuEngine(graph);
        Tensor<float> x = T(new[] { 2, 2 }, 1, 0, 0, 1);   // two orthonormal tokens
        Tensor<float> outT = engine.Run(new Dictionary<string, NamedTensor>
        {
            ["x"] = new NamedTensor("x", x),
        }).Values.Single().Data;

        // scores = I, scaled = diag(1/sqrt2); softmax over each row of [s,0]:
        float hi = MathF.Exp(scale) / (MathF.Exp(scale) + 1f);
        float lo = 1f - hi;
        float[] expected = { hi, lo, lo, hi };

        Assert.Equal(new[] { 2, 2 }, outT.Shape.Dimensions.ToArray());
        float[] a = outT.Span.ToArray();
        for (int i = 0; i < 4; i++) Assert.True(MathF.Abs(expected[i] - a[i]) < 1e-5f, $"[{i}] {expected[i]} vs {a[i]}");
    }
}
