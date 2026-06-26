using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Nn;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the normalization-family operators (InstanceNormalization,
/// GroupNormalization, MeanVarianceNormalization, LpNormalization, LRN). Each test builds
/// a <see cref="GraphNode"/> plus a <see cref="GraphContext"/> by hand, runs the kernel in
/// isolation, and checks hand-computed expected values. Mirrors NewOpsTests.cs.
/// </summary>
public class NormalizationOpsTests
{
    // ---- helpers -------------------------------------------------------------------------------

    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    // Per-channel/group standardization used to build expected values (matches the
    // population-variance, epsilon-inside-sqrt convention the kernels use).
    private static float[] Standardize(float[] data, float eps)
    {
        float mean = data.Average();
        float var = data.Select(v => (v - mean) * (v - mean)).Sum() / data.Length;
        float inv = 1f / MathF.Sqrt(var + eps);
        return data.Select(v => (v - mean) * inv).ToArray();
    }

    // ---- InstanceNormalization -----------------------------------------------------------------

    [Fact]
    public void InstanceNormalization_2Channels()
    {
        // X [1,2,2,2]: channel 0 = [1,2,3,4], channel 1 = [5,6,7,8].
        // Normalize each channel over its 4 spatial values, then affine per channel.
        const float eps = 1e-5f;
        float[] ch0 = Standardize(new[] { 1f, 2f, 3f, 4f }, eps);
        float[] ch1 = Standardize(new[] { 5f, 6f, 7f, 8f }, eps);

        // scale = [2, 3], B = [0.5, -1].
        var expected = new float[8];
        for (int i = 0; i < 4; i++) expected[i] = ch0[i] * 2f + 0.5f;
        for (int i = 0; i < 4; i++) expected[4 + i] = ch1[i] * 3f - 1f;

        var ctx = Ctx(
            ("x", F(new[] { 1, 2, 2, 2 }, 1, 2, 3, 4, 5, 6, 7, 8)),
            ("scale", F(new[] { 2 }, 2f, 3f)),
            ("b", F(new[] { 2 }, 0.5f, -1f)));
        new InstanceNormalizationKernel().Execute(
            Node("InstanceNormalization", new[] { "x", "scale", "b" }, new[] { "y" }), ctx);

        Close(expected, ctx.Get("y").Span.ToArray());
    }

    // ---- GroupNormalization --------------------------------------------------------------------

    [Fact]
    public void GroupNormalization_TwoGroups()
    {
        // X [1,4,1,2], num_groups=2 -> group0 = channels {0,1}, group1 = channels {2,3}.
        // channel data (spatial=2 each):
        //   c0=[1,2] c1=[3,4]  -> group0 over [1,2,3,4]
        //   c2=[5,6] c3=[7,8]  -> group1 over [5,6,7,8]
        const float eps = 1e-5f;
        float[] g0 = Standardize(new[] { 1f, 2f, 3f, 4f }, eps);
        float[] g1 = Standardize(new[] { 5f, 6f, 7f, 8f }, eps);

        // scale/bias length C = 4: scale=[1,2,3,4], bias=[0,0,0,0] for simplicity.
        float[] scale = { 1f, 2f, 3f, 4f };
        var expected = new float[8];
        // group0: c0 uses scale[0]=1 -> g0[0..1]; c1 uses scale[1]=2 -> g0[2..3]*2
        expected[0] = g0[0] * scale[0];
        expected[1] = g0[1] * scale[0];
        expected[2] = g0[2] * scale[1];
        expected[3] = g0[3] * scale[1];
        // group1: c2 uses scale[2]=3 -> g1[0..1]*3; c3 uses scale[3]=4 -> g1[2..3]*4
        expected[4] = g1[0] * scale[2];
        expected[5] = g1[1] * scale[2];
        expected[6] = g1[2] * scale[3];
        expected[7] = g1[3] * scale[3];

        var ctx = Ctx(
            ("x", F(new[] { 1, 4, 1, 2 }, 1, 2, 3, 4, 5, 6, 7, 8)),
            ("scale", F(new[] { 4 }, scale)),
            ("bias", F(new[] { 4 }, 0f, 0f, 0f, 0f)));
        new GroupNormalizationKernel().Execute(
            Node("GroupNormalization", new[] { "x", "scale", "bias" }, new[] { "y" },
                new Dictionary<string, object> { ["num_groups"] = 2L }), ctx);

        Close(expected, ctx.Get("y").Span.ToArray());
    }

    // ---- MeanVarianceNormalization -------------------------------------------------------------

    [Fact]
    public void MeanVarianceNormalization_DefaultAxes()
    {
        // X [1,2,2,2]. Default axes = [0,2,3]: reduce over N and spatial, keep channel.
        // channel 0 elements = [1,2,3,4], channel 1 = [5,6,7,8].
        const float eps = 1e-9f;
        float[] ch0 = Standardize(new[] { 1f, 2f, 3f, 4f }, eps);
        float[] ch1 = Standardize(new[] { 5f, 6f, 7f, 8f }, eps);

        // Layout is [n,c,h,w]; with N=1 the channel-0 block is the first 4, channel-1 the next 4.
        var expected = new float[8];
        for (int i = 0; i < 4; i++) expected[i] = ch0[i];
        for (int i = 0; i < 4; i++) expected[4 + i] = ch1[i];

        var ctx = Ctx(("x", F(new[] { 1, 2, 2, 2 }, 1, 2, 3, 4, 5, 6, 7, 8)));
        new MeanVarianceNormalizationKernel().Execute(
            Node("MeanVarianceNormalization", new[] { "x" }, new[] { "y" }), ctx);

        Close(expected, ctx.Get("y").Span.ToArray(), 1e-3f);
    }

    [Fact]
    public void MeanVarianceNormalization_ExplicitAxis()
    {
        // X [2,3], axes=[1]: normalize each row over its 3 columns.
        const float eps = 1e-9f;
        float[] r0 = Standardize(new[] { 1f, 2f, 3f }, eps);
        float[] r1 = Standardize(new[] { 4f, 5f, 6f }, eps);

        var ctx = Ctx(("x", F(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)));
        new MeanVarianceNormalizationKernel().Execute(
            Node("MeanVarianceNormalization", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object> { ["axes"] = new long[] { 1 } }), ctx);

        Close(r0.Concat(r1).ToArray(), ctx.Get("y").Span.ToArray(), 1e-3f);
    }

    // ---- LpNormalization -----------------------------------------------------------------------

    [Fact]
    public void LpNormalization_L2_LastAxis()
    {
        // X [2,2]: rows [3,4] and [0,5]. L2 norms: 5 and 5.
        var ctx = Ctx(("x", F(new[] { 2, 2 }, 3, 4, 0, 5)));
        new LpNormalizationKernel().Execute(
            Node("LpNormalization", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object> { ["axis"] = -1L, ["p"] = 2L }), ctx);
        Close(new[] { 0.6f, 0.8f, 0f, 1f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void LpNormalization_L1_LastAxis()
    {
        // X [2,2]: rows [1,3] and [2,2]. L1 norms: 4 and 4.
        var ctx = Ctx(("x", F(new[] { 2, 2 }, 1, 3, 2, 2)));
        new LpNormalizationKernel().Execute(
            Node("LpNormalization", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object> { ["axis"] = -1L, ["p"] = 1L }), ctx);
        Close(new[] { 0.25f, 0.75f, 0.5f, 0.5f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void LpNormalization_L2_Axis0()
    {
        // X [2,2]: columns are [3,4]^T and [0,5]^T along axis 0. L2 norms: 5 and 5.
        var ctx = Ctx(("x", F(new[] { 2, 2 }, 3, 0, 4, 5)));
        new LpNormalizationKernel().Execute(
            Node("LpNormalization", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object> { ["axis"] = 0L, ["p"] = 2L }), ctx);
        // col0 = [3,4]/5 = [0.6,0.8]; col1 = [0,5]/5 = [0,1].
        // Stored row-major: [c00, c01, c10, c11] = [0.6, 0, 0.8, 1].
        Close(new[] { 0.6f, 0f, 0.8f, 1f }, ctx.Get("y").Span.ToArray());
    }

    // ---- LRN -----------------------------------------------------------------------------------

    [Fact]
    public void LRN_Size3_AcrossChannels()
    {
        // X [1,3,1,1]: channel values x = [1,2,3]. size=3, alpha=1, beta=1, bias=0.
        // Neighborhood for size=3 is [c-1, c+1] clamped. coeff = alpha/size = 1/3.
        //   c0: neighbors {0,1} -> sq = 1+4 = 5;  denom = (0 + 1/3*5)^1 = 5/3; y = 1/(5/3) = 0.6
        //   c1: neighbors {0,1,2} -> sq = 1+4+9 = 14; denom = 14/3; y = 2/(14/3) = 6/14 = 0.428571
        //   c2: neighbors {1,2} -> sq = 4+9 = 13; denom = 13/3; y = 3/(13/3) = 9/13 = 0.692307
        var ctx = Ctx(("x", F(new[] { 1, 3, 1, 1 }, 1, 2, 3)));
        new LRNKernel().Execute(
            Node("LRN", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object>
                {
                    ["size"] = 3L,
                    ["alpha"] = 1f,
                    ["beta"] = 1f,
                    ["bias"] = 0f,
                }), ctx);
        Close(new[] { 0.6f, 6f / 14f, 9f / 13f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void LRN_Defaults_AreNearIdentityForSmallValues()
    {
        // With default alpha=1e-4, beta=0.75, bias=1, small inputs barely change.
        // X [1,2,1,1] = [1,2], size=1 -> neighborhood is just the element itself.
        //   c0: sq=1; denom=(1 + 1e-4/1 * 1)^0.75; y = 1/denom
        //   c1: sq=4; denom=(1 + 1e-4 * 4)^0.75; y = 2/denom
        float d0 = MathF.Pow(1f + 1e-4f * 1f, 0.75f);
        float d1 = MathF.Pow(1f + 1e-4f * 4f, 0.75f);
        var ctx = Ctx(("x", F(new[] { 1, 2, 1, 1 }, 1, 2)));
        new LRNKernel().Execute(
            Node("LRN", new[] { "x" }, new[] { "y" },
                new Dictionary<string, object> { ["size"] = 1L }), ctx);
        Close(new[] { 1f / d0, 2f / d1 }, ctx.Get("y").Span.ToArray());
    }
}
