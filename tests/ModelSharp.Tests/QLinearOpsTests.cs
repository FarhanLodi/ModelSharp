using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Tests for the <c>QLinear*</c> quantized op family. Each kernel result is compared against an
/// in-test "dequantize → float op → requantize" reference using the same round-half-to-even /
/// saturate math, so the asserts are exact on the integer outputs.
/// </summary>
public class QLinearOpsTests
{
    // ---- reference quant helpers (mirror QuantizeOps) -----------------------------------------

    private static double RoundHalfToEven(double v) => Math.Round(v, MidpointRounding.ToEven);
    private static byte SatU8(double v) => v <= 0 ? (byte)0 : v >= 255 ? (byte)255 : (byte)v;

    private static byte QuantU8(float val, float scale, byte zp) =>
        SatU8(RoundHalfToEven(val / scale) + zp);

    private static float Dequant(byte q, float scale, byte zp) => (q - zp) * scale;

    private static Tensor<byte> U8(TensorShape shape, params byte[] v) => new(shape, v);
    private static Tensor<float> F(TensorShape shape, params float[] v) => new(shape, v);
    private static Tensor<float> Scalar(float v) => new(new TensorShape(), new[] { v });
    private static Tensor<byte> ZpScalar(byte v) => new(new TensorShape(), new[] { v });

    private static IReadOnlyDictionary<string, NamedTensor> Run(ModelGraph g, Dictionary<string, NamedTensor> feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        return engine.Run(feeds);
    }

    // ---- QLinearMatMul ------------------------------------------------------------------------

    [Fact]
    public void QLinearMatMul_Matches_Dequant_Float_Requant_Reference()
    {
        // A: [2,3] uint8, B: [3,2] uint8. Per-tensor scales/zero-points.
        byte[] aq = { 100, 110, 120, 130, 140, 150 };
        byte[] bq = { 10, 20, 30, 40, 50, 60 };
        float aScale = 0.05f, bScale = 0.02f, yScale = 0.1f;
        byte aZp = 128, bZp = 0, yZp = 64;

        var graph = new ModelGraph
        {
            Inputs = new[] { "A", "a_s", "a_z", "B", "b_s", "b_z", "y_s", "y_z" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                new GraphNode("QLinearMatMul", "qmm",
                    new[] { "A", "a_s", "a_z", "B", "b_s", "b_z", "y_s", "y_z" }, new[] { "Y" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["A"] = new NamedTensor("A", U8(new TensorShape(2, 3), aq)),
            ["a_s"] = new NamedTensor("a_s", Scalar(aScale)),
            ["a_z"] = new NamedTensor("a_z", ZpScalar(aZp)),
            ["B"] = new NamedTensor("B", U8(new TensorShape(3, 2), bq)),
            ["b_s"] = new NamedTensor("b_s", Scalar(bScale)),
            ["b_z"] = new NamedTensor("b_z", ZpScalar(bZp)),
            ["y_s"] = new NamedTensor("y_s", Scalar(yScale)),
            ["y_z"] = new NamedTensor("y_z", ZpScalar(yZp)),
        });

        // Reference: dequant, float matmul [2,2], requant.
        float[] a = aq.Select(q => Dequant(q, aScale, aZp)).ToArray();
        float[] b = bq.Select(q => Dequant(q, bScale, bZp)).ToArray();
        var expected = new byte[4];
        for (int m = 0; m < 2; m++)
        for (int n = 0; n < 2; n++)
        {
            float sum = 0f;
            for (int k = 0; k < 3; k++) sum += a[m * 3 + k] * b[k * 2 + n];
            expected[m * 2 + n] = QuantU8(sum, yScale, yZp);
        }

        Tensor y = outputs["Y"].Tensor;
        Assert.Equal(ElementType.UInt8, y.Dtype);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(expected, ((Tensor<byte>)y).Span.ToArray());
    }

    // ---- QLinearAdd / QLinearMul --------------------------------------------------------------

    [Fact]
    public void QLinearAdd_Matches_Reference()
    {
        byte[] aq = { 10, 20, 30, 40 };
        byte[] bq = { 5, 15, 25, 35 };
        float aScale = 0.1f, bScale = 0.2f, yScale = 0.15f;
        byte aZp = 8, bZp = 4, yZp = 16;

        var graph = new ModelGraph
        {
            Inputs = new[] { "A", "a_s", "a_z", "B", "b_s", "b_z", "y_s", "y_z" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                new GraphNode("QLinearAdd", "qadd",
                    new[] { "A", "a_s", "a_z", "B", "b_s", "b_z", "y_s", "y_z" }, new[] { "Y" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["A"] = new NamedTensor("A", U8(new TensorShape(4), aq)),
            ["a_s"] = new NamedTensor("a_s", Scalar(aScale)),
            ["a_z"] = new NamedTensor("a_z", ZpScalar(aZp)),
            ["B"] = new NamedTensor("B", U8(new TensorShape(4), bq)),
            ["b_s"] = new NamedTensor("b_s", Scalar(bScale)),
            ["b_z"] = new NamedTensor("b_z", ZpScalar(bZp)),
            ["y_s"] = new NamedTensor("y_s", Scalar(yScale)),
            ["y_z"] = new NamedTensor("y_z", ZpScalar(yZp)),
        });

        var expected = new byte[4];
        for (int i = 0; i < 4; i++)
            expected[i] = QuantU8(Dequant(aq[i], aScale, aZp) + Dequant(bq[i], bScale, bZp), yScale, yZp);
        Assert.Equal(expected, ((Tensor<byte>)outputs["Y"].Tensor).Span.ToArray());
    }

    [Fact]
    public void QLinearMul_Matches_Reference_With_Broadcast()
    {
        // A [2,2], B [2] broadcast over rows.
        byte[] aq = { 10, 20, 30, 40 };
        byte[] bq = { 5, 6 };
        float aScale = 0.1f, bScale = 0.2f, yScale = 0.05f;
        byte aZp = 2, bZp = 1, yZp = 0;

        var graph = new ModelGraph
        {
            Inputs = new[] { "A", "a_s", "a_z", "B", "b_s", "b_z", "y_s", "y_z" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                new GraphNode("QLinearMul", "qmul",
                    new[] { "A", "a_s", "a_z", "B", "b_s", "b_z", "y_s", "y_z" }, new[] { "Y" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["A"] = new NamedTensor("A", U8(new TensorShape(2, 2), aq)),
            ["a_s"] = new NamedTensor("a_s", Scalar(aScale)),
            ["a_z"] = new NamedTensor("a_z", ZpScalar(aZp)),
            ["B"] = new NamedTensor("B", U8(new TensorShape(2), bq)),
            ["b_s"] = new NamedTensor("b_s", Scalar(bScale)),
            ["b_z"] = new NamedTensor("b_z", ZpScalar(bZp)),
            ["y_s"] = new NamedTensor("y_s", Scalar(yScale)),
            ["y_z"] = new NamedTensor("y_z", ZpScalar(yZp)),
        });

        var expected = new byte[4];
        for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++)
        {
            float prod = Dequant(aq[r * 2 + c], aScale, aZp) * Dequant(bq[c], bScale, bZp);
            expected[r * 2 + c] = QuantU8(prod, yScale, yZp);
        }
        Assert.Equal(expected, ((Tensor<byte>)outputs["Y"].Tensor).Span.ToArray());
    }

    // ---- QLinearConv --------------------------------------------------------------------------

    [Fact]
    public void QLinearConv_Matches_Dequant_Float_Requant_Reference()
    {
        // 1x1x3x3 input, 1x1x2x2 weight, no bias, stride 1, no pad -> 1x1x2x2 output.
        byte[] xq = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
        byte[] wq = { 1, 2, 3, 4 };
        float xScale = 0.1f, wScale = 0.05f, yScale = 0.2f;
        byte xZp = 5, wZp = 0, yZp = 10;

        var graph = new ModelGraph
        {
            Inputs = new[] { "X", "x_s", "x_z", "W", "w_s", "w_z", "y_s", "y_z" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                new GraphNode("QLinearConv", "qconv",
                    new[] { "X", "x_s", "x_z", "W", "w_s", "w_z", "y_s", "y_z" }, new[] { "Y" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["X"] = new NamedTensor("X", U8(new TensorShape(1, 1, 3, 3), xq)),
            ["x_s"] = new NamedTensor("x_s", Scalar(xScale)),
            ["x_z"] = new NamedTensor("x_z", ZpScalar(xZp)),
            ["W"] = new NamedTensor("W", U8(new TensorShape(1, 1, 2, 2), wq)),
            ["w_s"] = new NamedTensor("w_s", Scalar(wScale)),
            ["w_z"] = new NamedTensor("w_z", ZpScalar(wZp)),
            ["y_s"] = new NamedTensor("y_s", Scalar(yScale)),
            ["y_z"] = new NamedTensor("y_z", ZpScalar(yZp)),
        });

        // Reference: dequant, 2x2 valid conv, requant.
        float[] x = xq.Select(q => Dequant(q, xScale, xZp)).ToArray();
        float[] w = wq.Select(q => Dequant(q, wScale, wZp)).ToArray();
        var expected = new byte[4];
        for (int oy = 0; oy < 2; oy++)
        for (int ox = 0; ox < 2; ox++)
        {
            float sum = 0f;
            for (int ky = 0; ky < 2; ky++)
            for (int kx = 0; kx < 2; kx++)
                sum += x[(oy + ky) * 3 + (ox + kx)] * w[ky * 2 + kx];
            expected[oy * 2 + ox] = QuantU8(sum, yScale, yZp);
        }

        Tensor y = outputs["Y"].Tensor;
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(expected, ((Tensor<byte>)y).Span.ToArray());
    }

    // ---- ConvInteger --------------------------------------------------------------------------

    [Fact]
    public void ConvInteger_Computes_Integer_Convolution()
    {
        // (x - x_zp) conv (w) ; w_zp omitted. 1x1x3x3, 1x1x2x2, int32 output.
        byte[] xq = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
        byte[] wq = { 1, 2, 3, 4 };
        byte xZp = 5;

        var graph = new ModelGraph
        {
            Inputs = new[] { "X", "W", "x_z" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                new GraphNode("ConvInteger", "ci", new[] { "X", "W", "x_z" }, new[] { "Y" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["X"] = new NamedTensor("X", U8(new TensorShape(1, 1, 3, 3), xq)),
            ["W"] = new NamedTensor("W", U8(new TensorShape(1, 1, 2, 2), wq)),
            ["x_z"] = new NamedTensor("x_z", ZpScalar(xZp)),
        });

        var expected = new int[4];
        for (int oy = 0; oy < 2; oy++)
        for (int ox = 0; ox < 2; ox++)
        {
            int sum = 0;
            for (int ky = 0; ky < 2; ky++)
            for (int kx = 0; kx < 2; kx++)
                sum += (xq[(oy + ky) * 3 + (ox + kx)] - xZp) * wq[ky * 2 + kx];
            expected[oy * 2 + ox] = sum;
        }

        Tensor y = outputs["Y"].Tensor;
        Assert.Equal(ElementType.Int32, y.Dtype);
        Assert.Equal(expected, ((Tensor<int>)y).Span.ToArray());
    }

    // ---- QLinearGlobalAveragePool -------------------------------------------------------------

    [Fact]
    public void QLinearGlobalAveragePool_Averages_Then_Requants()
    {
        byte[] xq = { 10, 20, 30, 40 };   // 1x1x2x2
        float xScale = 0.1f, yScale = 0.1f;
        byte xZp = 0, yZp = 0;

        var graph = new ModelGraph
        {
            Inputs = new[] { "X", "x_s", "x_z", "y_s", "y_z" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                new GraphNode("QLinearGlobalAveragePool", "qgap",
                    new[] { "X", "x_s", "x_z", "y_s", "y_z" }, new[] { "Y" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["X"] = new NamedTensor("X", U8(new TensorShape(1, 1, 2, 2), xq)),
            ["x_s"] = new NamedTensor("x_s", Scalar(xScale)),
            ["x_z"] = new NamedTensor("x_z", ZpScalar(xZp)),
            ["y_s"] = new NamedTensor("y_s", Scalar(yScale)),
            ["y_z"] = new NamedTensor("y_z", ZpScalar(yZp)),
        });

        float avg = xq.Select(q => Dequant(q, xScale, xZp)).Average();
        byte expected = QuantU8(avg, yScale, yZp);

        Tensor y = outputs["Y"].Tensor;
        Assert.Equal(new[] { 1, 1, 1, 1 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(expected, ((Tensor<byte>)y).Span[0]);
    }
}
