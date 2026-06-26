using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Llm;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the fused LLM normalization / positional ops
/// (Roadmap C6): RMSNorm, skip+RMSNorm, and rotary position embedding. Each test
/// builds a <see cref="GraphNode"/> plus a <see cref="GraphContext"/> by hand and
/// checks against values computed straight from the op definition.
/// </summary>
public class FusedLlmNormTests
{
    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> I64(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    // ---- SimplifiedLayerNormalization (RMSNorm) ------------------------------------------------

    [Fact]
    public void RmsNorm_LastAxis_HandComputed()
    {
        // x = [[1, 2, 3, 4]], scale = [1, 1, 1, 1], eps = 0.
        // mean(x^2) = (1+4+9+16)/4 = 7.5 ; inv = 1/sqrt(7.5).
        float inv = 1f / MathF.Sqrt(7.5f);
        var ctx = Ctx(
            ("x", F(new[] { 1, 4 }, 1, 2, 3, 4)),
            ("scale", F(new[] { 4 }, 1, 1, 1, 1)));
        new SimplifiedLayerNormalizationKernel().Execute(
            Node("SimplifiedLayerNormalization", new[] { "x", "scale" }, new[] { "y" },
                new Dictionary<string, object> { ["epsilon"] = 0f }), ctx);
        Close(new[] { 1f * inv, 2f * inv, 3f * inv, 4f * inv }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void RmsNorm_Scale_And_InvStdOutput()
    {
        // Two rows, non-trivial scale, eps default-ish small. Verify per-row inv_std_dev.
        float eps = 1e-5f;
        var ctx = Ctx(
            ("x", F(new[] { 2, 2 }, 3, 4, 1, 1)),
            ("scale", F(new[] { 2 }, 2, 0.5f)));
        new SimplifiedLayerNormalizationKernel().Execute(
            Node("SimplifiedLayerNormalization", new[] { "x", "scale" }, new[] { "y", "inv" },
                new Dictionary<string, object> { ["epsilon"] = eps }), ctx);

        float inv0 = 1f / MathF.Sqrt((9f + 16f) / 2f + eps); // row [3,4]
        float inv1 = 1f / MathF.Sqrt((1f + 1f) / 2f + eps);  // row [1,1]
        Close(new[]
        {
            3f * inv0 * 2f, 4f * inv0 * 0.5f,
            1f * inv1 * 2f, 1f * inv1 * 0.5f,
        }, ctx.Get("y").Span.ToArray());
        Close(new[] { inv0, inv1 }, ctx.GetTensor("inv").AsFloat().Span.ToArray());
    }

    // ---- SkipSimplifiedLayerNormalization ------------------------------------------------------

    [Fact]
    public void SkipRmsNorm_SumThenNorm_HandComputed()
    {
        // input + skip = [1+1, 2+2, 3+0, 4+0] = [2, 4, 3, 4].
        // mean(t^2) = (4+16+9+16)/4 = 11.25 ; inv = 1/sqrt(11.25).
        // gamma = [1,1,1,1], no beta/bias, eps = 0.
        float inv = 1f / MathF.Sqrt(11.25f);
        var ctx = Ctx(
            ("input", F(new[] { 1, 4 }, 1, 2, 3, 4)),
            ("skip", F(new[] { 1, 4 }, 1, 2, 0, 0)),
            ("gamma", F(new[] { 4 }, 1, 1, 1, 1)));
        new SkipSimplifiedLayerNormalizationKernel().Execute(
            Node("SkipSimplifiedLayerNormalization",
                new[] { "input", "skip", "gamma" },
                new[] { "out", "", "", "sum" },
                new Dictionary<string, object> { ["epsilon"] = 0f }), ctx);

        Close(new[] { 2f * inv, 4f * inv, 3f * inv, 4f * inv }, ctx.Get("out").Span.ToArray());
        // Output 3 = residual sum (input + skip).
        Close(new[] { 2f, 4f, 3f, 4f }, ctx.GetTensor("sum").AsFloat().Span.ToArray());
    }

    [Fact]
    public void SkipRmsNorm_WithBiasBetaGamma()
    {
        // t = input + skip + bias = [1+0+1, 2+1+1] = [2, 4].
        // mean(t^2) = (4+16)/2 = 10 ; inv = 1/sqrt(10).
        // out = t*inv*gamma + beta, gamma=[2,3], beta=[0.1,-0.2].
        float eps = 0f;
        float inv = 1f / MathF.Sqrt(10f + eps);
        var ctx = Ctx(
            ("input", F(new[] { 1, 2 }, 1, 2)),
            ("skip", F(new[] { 1, 2 }, 0, 1)),
            ("gamma", F(new[] { 2 }, 2, 3)),
            ("beta", F(new[] { 2 }, 0.1f, -0.2f)),
            ("bias", F(new[] { 2 }, 1, 1)));
        new SkipSimplifiedLayerNormalizationKernel().Execute(
            Node("SkipSimplifiedLayerNormalization",
                new[] { "input", "skip", "gamma", "beta", "bias" },
                new[] { "out" },
                new Dictionary<string, object> { ["epsilon"] = eps }), ctx);

        Close(new[]
        {
            2f * inv * 2f + 0.1f,
            4f * inv * 3f - 0.2f,
        }, ctx.Get("out").Span.ToArray());
    }

    // ---- RotaryEmbedding -----------------------------------------------------------------------

    // A tiny [1,1,2,4] case: batch=1, heads=1, seq=2, head_size=4, rotary=4 (rotHalf=2).
    // cos/sin cache is [max_pos=2, rotary/2=2]. Position 0 = identity (cos=1,sin=0),
    // position 1 uses angles (t0, t1).
    private static (float[] cos, float[] sin) Cache()
    {
        float t0 = 0.5f, t1 = 1.1f;
        return (
            new[] { 1f, 1f, MathF.Cos(t0), MathF.Cos(t1) },
            new[] { 0f, 0f, MathF.Sin(t0), MathF.Sin(t1) });
    }

    [Fact]
    public void Rope_HalfSplit_NeoX_HandComputed()
    {
        (float[] cos, float[] sin) = Cache();
        // token0 = [1,2,3,4], token1 = [5,6,7,8].
        var ctx = Ctx(
            ("x", F(new[] { 1, 1, 2, 4 }, 1, 2, 3, 4, 5, 6, 7, 8)),
            ("pos", I64(new[] { 1, 2 }, 0, 1)),
            ("cos", F(new[] { 2, 2 }, cos)),
            ("sin", F(new[] { 2, 2 }, sin)));
        new RotaryEmbeddingKernel().Execute(
            Node("RotaryEmbedding", new[] { "x", "pos", "cos", "sin" }, new[] { "y" },
                new Dictionary<string, object> { ["interleaved"] = 0L }), ctx);

        // half-split pairs: (x[0],x[2]) and (x[1],x[3]).
        // token0 pos0 -> cos=[1,1], sin=[0,0] -> identity.
        // token1 pos1 -> j=0: c=cos(t1col0), j=1: c=cos(t1col1).
        float c0 = cos[2], s0 = sin[2]; // pos1, col0
        float c1 = cos[3], s1 = sin[3]; // pos1, col1
        // token1 = [5,6,7,8]: pairs (5,7) with (c0,s0), (6,8) with (c1,s1).
        var expected = new[]
        {
            1f, 2f, 3f, 4f,                       // token0 identity
            5f * c0 - 7f * s0, 6f * c1 - 8f * s1,
            7f * c0 + 5f * s0, 8f * c1 + 6f * s1,
        };
        Close(expected, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void Rope_Interleaved_GptJ_HandComputed()
    {
        (float[] cos, float[] sin) = Cache();
        var ctx = Ctx(
            ("x", F(new[] { 1, 1, 2, 4 }, 1, 2, 3, 4, 5, 6, 7, 8)),
            ("pos", I64(new[] { 1, 2 }, 0, 1)),
            ("cos", F(new[] { 2, 2 }, cos)),
            ("sin", F(new[] { 2, 2 }, sin)));
        new RotaryEmbeddingKernel().Execute(
            Node("RotaryEmbedding", new[] { "x", "pos", "cos", "sin" }, new[] { "y" },
                new Dictionary<string, object> { ["interleaved"] = 1L }), ctx);

        // interleaved pairs: (x[0],x[1]) with cos col0, (x[2],x[3]) with cos col1.
        float c0 = cos[2], s0 = sin[2]; // pos1 col0
        float c1 = cos[3], s1 = sin[3]; // pos1 col1
        // token1 = [5,6,7,8]: pair (5,6) with (c0,s0), pair (7,8) with (c1,s1).
        var expected = new[]
        {
            1f, 2f, 3f, 4f,                       // token0 identity
            5f * c0 - 6f * s0, 6f * c0 + 5f * s0,
            7f * c1 - 8f * s1, 8f * c1 + 7f * s1,
        };
        Close(expected, ctx.Get("y").Span.ToArray());
    }
}
