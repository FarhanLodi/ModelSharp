using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Generators;
using ModelSharp.Cpu.Kernels.MathOps;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the miscellaneous math / generator ops (Mod, BitShift, and the
/// RandomNormal/RandomUniform + *Like family). Mirrors <see cref="NewOpsTests"/>: each test builds
/// a <see cref="GraphNode"/> + <see cref="GraphContext"/> by hand and runs the kernel in isolation.
/// The Random* tests pass a fixed <c>seed</c> and assert reproducibility by recomputing the exact
/// reference sequence with the same RNG/formula, plus shape/range/statistical sanity.
/// </summary>
public class MiscMathOpsTests
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

    private static Tensor<long> I64(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    private static Tensor<int> I32(int[] dims, params int[] data) =>
        Tensor<int>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    // ---- Mod -----------------------------------------------------------------------------------

    [Fact]
    public void Mod_Int_Fmod0_SignOfDivisor()
    {
        // Python-style %: result has the sign of the divisor.
        //   -7 % 3 = 2, 7 % -3 = -2, 7 % 3 = 1, -7 % -3 = -1.
        var ctx = Ctx(("a", I64(new[] { 4 }, -7, 7, 7, -7)), ("b", I64(new[] { 4 }, 3, -3, 3, -3)));
        new ModKernel().Execute(Node("Mod", new[] { "a", "b" }, new[] { "y" }), ctx);
        Assert.Equal(new long[] { 2, -2, 1, -1 }, ctx.GetTensor("y").AsInt64().Span.ToArray());
    }

    [Fact]
    public void Mod_Int32_Fmod0_Broadcasts()
    {
        // a:[2,1], b:[1,3] -> [2,3], Python-style modulo (sign of divisor).
        var ctx = Ctx(("a", I32(new[] { 2, 1 }, -5, 5)), ("b", I32(new[] { 1, 3 }, 2, 3, 4)));
        new ModKernel().Execute(Node("Mod", new[] { "a", "b" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 2, 3 }, ctx.GetTensor("y").Shape.Dimensions.ToArray());
        // row0 (-5): -5%2=1, -5%3=1, -5%4=3 ; row1 (5): 5%2=1, 5%3=2, 5%4=1.
        Assert.Equal(new[] { 1, 1, 3, 1, 2, 1 }, ctx.GetTensor("y").AsInt32().Span.ToArray());
    }

    [Fact]
    public void Mod_Float_Fmod1_SignOfDividend()
    {
        // C fmod: result has the sign of the dividend.
        //   5.3 % 2 = 1.3, -5.3 % 2 = -1.3, 5.3 % -2 = 1.3.
        var ctx = Ctx(("a", F(new[] { 3 }, 5.3f, -5.3f, 5.3f)), ("b", F(new[] { 3 }, 2f, 2f, -2f)));
        new ModKernel().Execute(Node("Mod", new[] { "a", "b" }, new[] { "y" },
            new Dictionary<string, object> { ["fmod"] = 1L }), ctx);
        Close(new[] { 5.3f % 2f, -5.3f % 2f, 5.3f % -2f }, ctx.Get("y").Span.ToArray(), 1e-5f);
    }

    [Fact]
    public void Mod_Float_Fmod0_Throws()
    {
        var ctx = Ctx(("a", F(new[] { 1 }, 5f)), ("b", F(new[] { 1 }, 2f)));
        Assert.Throws<ModelSharpException>(() =>
            new ModKernel().Execute(Node("Mod", new[] { "a", "b" }, new[] { "y" }), ctx));
    }

    // ---- BitShift ------------------------------------------------------------------------------

    [Fact]
    public void BitShift_Left_Int64()
    {
        var ctx = Ctx(("a", I64(new[] { 3 }, 1, 2, 3)), ("b", I64(new[] { 3 }, 1, 2, 3)));
        new BitShiftKernel().Execute(Node("BitShift", new[] { "a", "b" }, new[] { "y" },
            new Dictionary<string, object> { ["direction"] = "LEFT" }), ctx);
        // 1<<1=2, 2<<2=8, 3<<3=24.
        Assert.Equal(new long[] { 2, 8, 24 }, ctx.GetTensor("y").AsInt64().Span.ToArray());
    }

    [Fact]
    public void BitShift_Right_Int32_Broadcasts()
    {
        // a:[4], b: scalar [1] (shift by 1) -> [4]; logical right shift.
        var ctx = Ctx(("a", I32(new[] { 4 }, 16, 9, 1, 255)), ("b", I32(new[] { 1 }, 1)));
        new BitShiftKernel().Execute(Node("BitShift", new[] { "a", "b" }, new[] { "y" },
            new Dictionary<string, object> { ["direction"] = "RIGHT" }), ctx);
        Assert.Equal(new[] { 8, 4, 0, 127 }, ctx.GetTensor("y").AsInt32().Span.ToArray());
    }

    [Fact]
    public void BitShift_BadDirection_Throws()
    {
        var ctx = Ctx(("a", I64(new[] { 1 }, 1)), ("b", I64(new[] { 1 }, 1)));
        Assert.Throws<ModelSharpException>(() =>
            new BitShiftKernel().Execute(Node("BitShift", new[] { "a", "b" }, new[] { "y" },
                new Dictionary<string, object> { ["direction"] = "SIDEWAYS" }), ctx));
    }

    // ---- RandomUniform -------------------------------------------------------------------------

    [Fact]
    public void RandomUniform_Shape_Range_And_Determinism()
    {
        const float seed = 42f, low = -2f, high = 5f;
        var attrs = new Dictionary<string, object>
        {
            ["shape"] = new long[] { 2, 3 },
            ["low"] = low,
            ["high"] = high,
            ["seed"] = seed,
        };

        var ctx = Ctx();
        new RandomUniformKernel().Execute(Node("RandomUniform", Array.Empty<string>(), new[] { "y" }, attrs), ctx);
        Tensor y = ctx.GetTensor("y");

        // Shape + dtype.
        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(ElementType.Float32, y.Dtype);

        // Range: every sample in [low, high).
        float[] got = y.AsFloat().Span.ToArray();
        foreach (float v in got)
            Assert.True(v >= low && v < high, $"sample {v} out of [{low}, {high})");

        // Determinism: recompute the exact reference with the same RNG/formula.
        var rng = new Random((int)seed);
        var expected = new float[6];
        for (int i = 0; i < expected.Length; i++) expected[i] = (float)(low + (high - low) * rng.NextDouble());
        Assert.Equal(expected, got);

        // Same seed -> identical sequence on a fresh run.
        var ctx2 = Ctx();
        new RandomUniformKernel().Execute(Node("RandomUniform", Array.Empty<string>(), new[] { "y" }, attrs), ctx2);
        Assert.Equal(got, ctx2.Get("y").Span.ToArray());
    }

    // ---- RandomNormal --------------------------------------------------------------------------

    [Fact]
    public void RandomNormal_Shape_And_Determinism()
    {
        const float seed = 7f, mean = 1f, scale = 2f;
        var attrs = new Dictionary<string, object>
        {
            ["shape"] = new long[] { 5 },
            ["mean"] = mean,
            ["scale"] = scale,
            ["seed"] = seed,
        };

        var ctx = Ctx();
        new RandomNormalKernel().Execute(Node("RandomNormal", Array.Empty<string>(), new[] { "y" }, attrs), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 5 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(ElementType.Float32, y.Dtype);
        float[] got = y.AsFloat().Span.ToArray();

        // Determinism: recompute via the same Box-Muller formula the kernel uses.
        var rng = new Random((int)seed);
        var expected = new float[5];
        for (int i = 0; i < expected.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double z = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Cos(2.0 * System.Math.PI * u2);
            expected[i] = (float)(mean + scale * z);
        }
        Assert.Equal(expected, got);
    }

    // ---- *Like variants copy shape -------------------------------------------------------------

    [Fact]
    public void RandomUniformLike_CopiesShape_And_Deterministic()
    {
        const float seed = 11f;
        var attrs = new Dictionary<string, object> { ["seed"] = seed };
        // Input only contributes its shape (and, by default, its float dtype).
        var ctx = Ctx(("x", F(new[] { 2, 2 }, 0, 0, 0, 0)));
        new RandomUniformLikeKernel().Execute(
            Node("RandomUniformLike", new[] { "x" }, new[] { "y" }, attrs), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(ElementType.Float32, y.Dtype);

        float[] got = y.AsFloat().Span.ToArray();
        var rng = new Random((int)seed);
        var expected = new float[4];
        for (int i = 0; i < expected.Length; i++) expected[i] = (float)(0.0 + 1.0 * rng.NextDouble());
        Assert.Equal(expected, got);
        foreach (float v in got) Assert.True(v >= 0f && v < 1f);
    }

    [Fact]
    public void RandomNormalLike_CopiesShape()
    {
        var attrs = new Dictionary<string, object> { ["seed"] = 3f };
        var ctx = Ctx(("x", F(new[] { 1, 4 }, 0, 0, 0, 0)));
        new RandomNormalLikeKernel().Execute(
            Node("RandomNormalLike", new[] { "x" }, new[] { "y" }, attrs), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 1, 4 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(ElementType.Float32, y.Dtype);
    }

    // ---- registration extension ----------------------------------------------------------------

    [Fact]
    public void AddMiscMathOps_RegistersAll()
    {
        var r = new KernelRegistry().AddMiscMathOps();
        foreach (string op in new[]
                 {
                     "Mod", "BitShift",
                     "RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike",
                 })
        {
            Assert.True(r.TryGet(op, out IKernel? k) && k is not null, $"{op} not registered");
        }
    }
}
