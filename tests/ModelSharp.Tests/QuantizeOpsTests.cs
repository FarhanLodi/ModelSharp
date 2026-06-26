using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Quantize;
using ModelSharp.Graph;
// NOTE: byte/sbyte tensors have no public As* accessor, so tests cast the
// dtype-carrying base tensor directly to the typed view via these helpers.
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the quantization operators (Roadmap C3): DequantizeLinear,
/// QuantizeLinear, DynamicQuantizeLinear and MatMulInteger. Each test builds a
/// <see cref="GraphNode"/> plus a <see cref="GraphContext"/> by hand, runs the kernel in
/// isolation, and checks against hand-computed expected values.
/// </summary>
public class QuantizeOpsTests
{
    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<byte> U8(int[] dims, params byte[] data) =>
        Tensor<byte>.FromArray(new TensorShape(dims), data);

    private static Tensor<sbyte> S8(int[] dims, params sbyte[] data) =>
        Tensor<sbyte>.FromArray(new TensorShape(dims), data);

    private static Tensor<byte> AsU8(Tensor t) => (Tensor<byte>)t;

    private static Tensor<sbyte> AsS8(Tensor t) => (Tensor<sbyte>)t;

    // ---- DequantizeLinear -----------------------------------------------------------------------

    [Fact]
    public void DequantizeLinear_PerTensor_UInt8()
    {
        // y = (x - zp) * scale, scale=2, zp=128.
        var ctx = Ctx(
            ("x", U8(new[] { 4 }, 0, 128, 129, 255)),
            ("s", F(Array.Empty<int>(), 2f)),
            ("z", U8(Array.Empty<int>(), 128)));
        new DequantizeLinearKernel().Execute(
            Node("DequantizeLinear", new[] { "x", "s", "z" }, new[] { "y" }), ctx);

        Tensor<float> y = ctx.Get("y");
        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { -256f, 0f, 2f, 254f }, y.Span.ToArray());
    }

    [Fact]
    public void DequantizeLinear_PerTensor_Int8_NoZeroPoint()
    {
        var ctx = Ctx(
            ("x", S8(new[] { 3 }, -2, 0, 5)),
            ("s", F(Array.Empty<int>(), 0.5f)));
        new DequantizeLinearKernel().Execute(
            Node("DequantizeLinear", new[] { "x", "s" }, new[] { "y" }), ctx);

        Assert.Equal(new[] { -1f, 0f, 2.5f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void DequantizeLinear_PerAxis()
    {
        // Shape [2,3], axis=0 → per-row scale [1, 10], zp [0, 1].
        var ctx = Ctx(
            ("x", U8(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)),
            ("s", F(new[] { 2 }, 1f, 10f)),
            ("z", U8(new[] { 2 }, 0, 1)));
        var attrs = new Dictionary<string, object> { ["axis"] = 0L };
        new DequantizeLinearKernel().Execute(
            Node("DequantizeLinear", new[] { "x", "s", "z" }, new[] { "y" }, attrs), ctx);

        // row0: (1,2,3)*1 ; row1: (4-1,5-1,6-1)*10 = 30,40,50.
        Assert.Equal(new[] { 1f, 2f, 3f, 30f, 40f, 50f }, ctx.Get("y").Span.ToArray());
    }

    // ---- QuantizeLinear -------------------------------------------------------------------------

    [Fact]
    public void QuantizeLinear_PerTensor_UInt8_Default()
    {
        // Default (no zero point) → uint8, zp 0. scale=2.
        var ctx = Ctx(
            ("x", F(new[] { 5 }, 0f, 2f, 3f, 5f, 1000f)),
            ("s", F(Array.Empty<int>(), 2f)));
        new QuantizeLinearKernel().Execute(
            Node("QuantizeLinear", new[] { "x", "s" }, new[] { "y" }), ctx);

        Tensor<byte> y = AsU8(ctx.GetTensor("y"));
        Assert.Equal(ElementType.UInt8, y.Dtype);
        // round-half-to-even: 0/2=0, 2/2=1, 3/2=1.5→2, 5/2=2.5→2, 1000/2=500→sat 255.
        Assert.Equal(new byte[] { 0, 1, 2, 2, 255 }, y.Span.ToArray());
    }

    [Fact]
    public void QuantizeLinear_Int8_WithZeroPoint()
    {
        var ctx = Ctx(
            ("x", F(new[] { 4 }, -260f, -1f, 1f, 260f)),
            ("s", F(Array.Empty<int>(), 1f)),
            ("z", S8(Array.Empty<int>(), 0)));
        new QuantizeLinearKernel().Execute(
            Node("QuantizeLinear", new[] { "x", "s", "z" }, new[] { "y" }), ctx);

        Tensor<sbyte> y = AsS8(ctx.GetTensor("y"));
        Assert.Equal(ElementType.Int8, y.Dtype);
        Assert.Equal(new sbyte[] { -128, -1, 1, 127 }, y.Span.ToArray());
    }

    [Fact]
    public void QuantizeLinear_RoundTrip_PerAxis()
    {
        // Quantize then dequantize per-axis (axis=1) and recover the original values.
        int[] dims = { 2, 2 };
        Tensor<float> x = F(dims, 0f, 4f, 8f, 12f);
        Tensor<float> scale = F(new[] { 2 }, 1f, 2f);
        Tensor<byte> zp = U8(new[] { 2 }, 0, 10);
        var attrs = new Dictionary<string, object> { ["axis"] = 1L };

        var q = Ctx(("x", x), ("s", scale), ("z", zp));
        new QuantizeLinearKernel().Execute(
            Node("QuantizeLinear", new[] { "x", "s", "z" }, new[] { "q" }, attrs), q);
        Tensor qy = q.GetTensor("q");

        var d = Ctx(("x", qy), ("s", scale), ("z", zp));
        new DequantizeLinearKernel().Execute(
            Node("DequantizeLinear", new[] { "x", "s", "z" }, new[] { "y" }, attrs), d);

        Assert.Equal(x.Span.ToArray(), d.Get("y").Span.ToArray());
    }

    // ---- DynamicQuantizeLinear ------------------------------------------------------------------

    [Fact]
    public void DynamicQuantizeLinear_MatchesHandComputed()
    {
        // ONNX example: x = [0, 2, -3, -2.5, 1.34, 0.5].
        Tensor<float> x = F(new[] { 6 }, 0f, 2f, -3f, -2.5f, 1.34f, 0.5f);
        var ctx = Ctx(("x", x));
        new DynamicQuantizeLinearKernel().Execute(
            Node("DynamicQuantizeLinear", new[] { "x" }, new[] { "y", "ys", "yz" }), ctx);

        // xmin=-3, xmax=2 (0 folded in). scale=(2-(-3))/255 = 5/255.
        // Compute in double to match the kernel (which quantizes with a double-precision
        // scale); a float scale would nudge the exact midpoint -2.5/scale = -127.5 off-tie.
        double scale = 5d / 255d;
        Assert.Equal((float)scale, ctx.Get("ys").Span[0], 5);

        // zp = round(-(-3)/scale) = round(3/scale) = round(153) = 153.
        Tensor<byte> yz = AsU8(ctx.GetTensor("yz"));
        Assert.Equal((byte)153, yz.Span[0]);

        // y = saturate(round(x/scale) + 153).
        var expected = new byte[6];
        for (int i = 0; i < 6; i++)
        {
            double q = Math.Round(x.Span[i] / scale, MidpointRounding.ToEven) + 153;
            expected[i] = q <= 0 ? (byte)0 : q >= 255 ? (byte)255 : (byte)q;
        }
        Tensor<byte> y = AsU8(ctx.GetTensor("y"));
        Assert.Equal(ElementType.UInt8, y.Dtype);
        Assert.Equal(expected, y.Span.ToArray());
    }

    // ---- MatMulInteger --------------------------------------------------------------------------

    [Fact]
    public void MatMulInteger_NoZeroPoints()
    {
        // A [2x3] uint8, B [3x2] uint8.
        var ctx = Ctx(
            ("a", U8(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)),
            ("b", U8(new[] { 3, 2 }, 7, 8, 9, 10, 11, 12)));
        new MatMulIntegerKernel().Execute(
            Node("MatMulInteger", new[] { "a", "b" }, new[] { "y" }), ctx);

        Tensor<int> y = ctx.GetTensor("y").AsInt32();
        Assert.Equal(ElementType.Int32, y.Dtype);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        Assert.Equal(new[] { 58, 64, 139, 154 }, y.Span.ToArray());
    }

    [Fact]
    public void MatMulInteger_WithScalarZeroPoints()
    {
        // ONNX example. A uint8 with a_zp=12, B uint8 with b_zp=0.
        var ctx = Ctx(
            ("a", U8(new[] { 4, 3 },
                11, 7, 3,
                10, 6, 2,
                9, 5, 1,
                8, 4, 0)),
            ("b", U8(new[] { 3, 2 },
                1, 4,
                2, 5,
                3, 6)),
            ("az", U8(Array.Empty<int>(), 12)),
            ("bz", U8(Array.Empty<int>(), 0)));
        new MatMulIntegerKernel().Execute(
            Node("MatMulInteger", new[] { "a", "b", "az", "bz" }, new[] { "y" }), ctx);

        Tensor<int> y = ctx.GetTensor("y").AsInt32();
        Assert.Equal(new[] { 4, 2 }, y.Shape.Dimensions.ToArray());
        // Reference result from the ONNX MatMulInteger spec example.
        Assert.Equal(new[] { -38, -83, -44, -98, -50, -113, -56, -128 }, y.Span.ToArray());
    }

    [Fact]
    public void MatMulInteger_SignedWithZeroPoints()
    {
        // int8 A and B with scalar zero points; verify against direct computation.
        var ctx = Ctx(
            ("a", S8(new[] { 2, 2 }, -1, 2, 3, -4)),
            ("b", S8(new[] { 2, 2 }, 5, -6, 7, 8)),
            ("az", S8(Array.Empty<int>(), 1)),
            ("bz", S8(Array.Empty<int>(), -2)));
        new MatMulIntegerKernel().Execute(
            Node("MatMulInteger", new[] { "a", "b", "az", "bz" }, new[] { "y" }), ctx);

        // A-1 = [[-2,1],[2,-5]] ; B+2 = [[7,-4],[9,10]].
        // [[-2*7+1*9, -2*-4+1*10],[2*7-5*9, 2*-4-5*10]] = [[-5,18],[-31,-58]].
        Assert.Equal(new[] { -5, 18, -31, -58 }, ctx.GetTensor("y").AsInt32().Span.ToArray());
    }
}
