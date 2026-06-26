using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Generators;
using ModelSharp.Cpu.Kernels.Linear;
using ModelSharp.Cpu.Kernels.MathOps;
using ModelSharp.Cpu.Kernels.Nn;
using ModelSharp.Cpu.Kernels.Shape;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the extra-ops batch: Bitwise{And,Or,Xor,Not}, {Hann,Hamming,Blackman}Window,
/// Einsum, Det, Unique, CenterCropPad, Dropout, ConvTranspose, GridSample, MaxRoiPool, Col2Im, Upsample,
/// NonMaxSuppression, Bernoulli, Multinomial. Expected values are hand-computed.
/// </summary>
public class ExtraOpsTests
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

    private static Tensor<int> I32(int[] dims, params int[] data) =>
        Tensor<int>.FromArray(new TensorShape(dims), data);

    private static Tensor<bool> B(int[] dims, params bool[] data) =>
        Tensor<bool>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void AssertClose(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    // ---- Bitwise -------------------------------------------------------------------------------

    [Fact]
    public void BitwiseAnd_Or_Xor_Int32()
    {
        var a = I32(new[] { 4 }, 0b1100, 0b1010, 5, 6);
        var b = I32(new[] { 4 }, 0b1010, 0b0110, 3, 6);

        var c1 = Ctx(("a", a), ("b", b));
        new BitwiseAndKernel().Execute(Node("BitwiseAnd", new[] { "a", "b" }, new[] { "y" }), c1);
        Assert.Equal(new[] { 0b1000, 0b0010, 1, 6 }, c1.GetTensor("y").AsInt32().Span.ToArray());

        var c2 = Ctx(("a", a), ("b", b));
        new BitwiseOrKernel().Execute(Node("BitwiseOr", new[] { "a", "b" }, new[] { "y" }), c2);
        Assert.Equal(new[] { 0b1110, 0b1110, 7, 6 }, c2.GetTensor("y").AsInt32().Span.ToArray());

        var c3 = Ctx(("a", a), ("b", b));
        new BitwiseXorKernel().Execute(Node("BitwiseXor", new[] { "a", "b" }, new[] { "y" }), c3);
        Assert.Equal(new[] { 0b0110, 0b1100, 6, 0 }, c3.GetTensor("y").AsInt32().Span.ToArray());
    }

    [Fact]
    public void BitwiseAnd_Broadcasts()
    {
        var c = Ctx(("a", I32(new[] { 2, 2 }, 0b11, 0b10, 0b01, 0b00)), ("b", I32(new[] { 2 }, 0b10, 0b01)));
        new BitwiseAndKernel().Execute(Node("BitwiseAnd", new[] { "a", "b" }, new[] { "y" }), c);
        // rows AND [2,1]: [3&2,2&1]=[2,0]; [1&2,0&1]=[0,0]
        Assert.Equal(new[] { 2, 0, 0, 0 }, c.GetTensor("y").AsInt32().Span.ToArray());
    }

    [Fact]
    public void BitwiseNot_Int64()
    {
        var c = Ctx(("x", I64(new[] { 3 }, 0, -1, 5)));
        new BitwiseNotKernel().Execute(Node("BitwiseNot", new[] { "x" }, new[] { "y" }), c);
        Assert.Equal(new long[] { -1, 0, -6 }, c.GetTensor("y").AsInt64().Span.ToArray());
    }

    // ---- Windows -------------------------------------------------------------------------------

    [Fact]
    public void HannWindow_Periodic5()
    {
        // periodic: denom=5, w[n]=0.5-0.5cos(2pi n/5)
        var c = Ctx(("size", I64(Array.Empty<int>(), 5)));
        new HannWindowKernel().Execute(Node("HannWindow", new[] { "size" }, new[] { "y" }), c);
        float[] y = c.GetTensor("y").AsFloat().Span.ToArray();
        var exp = new float[5];
        for (int n = 0; n < 5; n++) exp[n] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * n / 5));
        AssertClose(exp, y);
        Assert.Equal(0f, y[0], 4);
    }

    [Fact]
    public void HammingWindow_Symmetric5()
    {
        // periodic=0 -> denom=N-1=4. Symmetric: endpoints equal.
        var c = Ctx(("size", I64(Array.Empty<int>(), 5)));
        new HammingWindowKernel().Execute(Node("HammingWindow", new[] { "size" }, new[] { "y" },
            new Dictionary<string, object> { ["periodic"] = 0L }), c);
        float[] y = c.GetTensor("y").AsFloat().Span.ToArray();
        var exp = new float[5];
        for (int n = 0; n < 5; n++) exp[n] = (float)(0.54347826086956520 - 0.45652173913043480 * Math.Cos(2 * Math.PI * n / 4));
        AssertClose(exp, y);
        Assert.Equal(y[0], y[4], 4); // symmetric
    }

    [Fact]
    public void BlackmanWindow_Periodic8()
    {
        var c = Ctx(("size", I64(Array.Empty<int>(), 8)));
        new BlackmanWindowKernel().Execute(Node("BlackmanWindow", new[] { "size" }, new[] { "y" }), c);
        float[] y = c.GetTensor("y").AsFloat().Span.ToArray();
        var exp = new float[8];
        for (int n = 0; n < 8; n++)
            exp[n] = (float)(0.42 - 0.5 * Math.Cos(2 * Math.PI * n / 8) + 0.08 * Math.Cos(4 * Math.PI * n / 8));
        AssertClose(exp, y);
        Assert.Equal(0f, y[0], 4);
    }

    // ---- Einsum --------------------------------------------------------------------------------

    [Fact]
    public void Einsum_MatMul()
    {
        var c = Ctx(("a", F(new[] { 2, 2 }, 1, 2, 3, 4)), ("b", F(new[] { 2, 2 }, 5, 6, 7, 8)));
        new EinsumKernel().Execute(Node("Einsum", new[] { "a", "b" }, new[] { "y" },
            new Dictionary<string, object> { ["equation"] = "ik,kj->ij" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 19f, 22f, 43f, 50f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Einsum_Transpose()
    {
        var c = Ctx(("a", F(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)));
        new EinsumKernel().Execute(Node("Einsum", new[] { "a" }, new[] { "y" },
            new Dictionary<string, object> { ["equation"] = "ij->ji" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 3, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 1f, 4f, 2f, 5f, 3f, 6f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Einsum_DiagonalAndSum()
    {
        var diag = Ctx(("a", F(new[] { 2, 2 }, 1, 2, 3, 4)));
        new EinsumKernel().Execute(Node("Einsum", new[] { "a" }, new[] { "y" },
            new Dictionary<string, object> { ["equation"] = "ii->i" }), diag);
        AssertClose(new[] { 1f, 4f }, diag.GetTensor("y").AsFloat().Span.ToArray());

        var sum = Ctx(("a", F(new[] { 2, 2 }, 1, 2, 3, 4)));
        new EinsumKernel().Execute(Node("Einsum", new[] { "a" }, new[] { "y" },
            new Dictionary<string, object> { ["equation"] = "ij->" }), sum);
        Tensor s = sum.GetTensor("y");
        Assert.Empty(s.Shape.Dimensions.ToArray());
        AssertClose(new[] { 10f }, s.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Einsum_OuterProduct_ImplicitOutput()
    {
        // i,j with no -> : implicit output is "ij" (each label once, alphabetical).
        var c = Ctx(("a", F(new[] { 2 }, 1, 2)), ("b", F(new[] { 2 }, 3, 4)));
        new EinsumKernel().Execute(Node("Einsum", new[] { "a", "b" }, new[] { "y" },
            new Dictionary<string, object> { ["equation"] = "i,j" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 3f, 4f, 6f, 8f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Einsum_BatchMatMul()
    {
        // 2 batches of 1x2 @ 2x1 -> 1x1. b=0: [1,2]@[5;6]=17; b=1:[3,4]@[7;8]=53.
        var a = F(new[] { 2, 1, 2 }, 1, 2, 3, 4);
        var b = F(new[] { 2, 2, 1 }, 5, 6, 7, 8);
        var c = Ctx(("a", a), ("b", b));
        new EinsumKernel().Execute(Node("Einsum", new[] { "a", "b" }, new[] { "y" },
            new Dictionary<string, object> { ["equation"] = "bij,bjk->bik" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 2, 1, 1 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 17f, 53f }, y.AsFloat().Span.ToArray());
    }

    // ---- Det -----------------------------------------------------------------------------------

    [Fact]
    public void Det_2x2()
    {
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        var c = Ctx(("x", F(new[] { 2, 2 }, 1, 2, 3, 4)));
        new DetKernel().Execute(Node("Det", new[] { "x" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Empty(y.Shape.Dimensions.ToArray());
        Assert.Equal(-2f, y.AsFloat().Span[0], 4);
    }

    [Fact]
    public void Det_3x3_AndBatch()
    {
        // det of [[6,1,1],[4,-2,5],[2,8,7]] = -306
        var c = Ctx(("x", F(new[] { 2, 3, 3 },
            6, 1, 1, 4, -2, 5, 2, 8, 7,
            1, 0, 0, 0, 1, 0, 0, 0, 1)));  // identity -> det 1
        new DetKernel().Execute(Node("Det", new[] { "x" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(-306f, y.AsFloat().Span[0], 2);
        Assert.Equal(1f, y.AsFloat().Span[1], 4);
    }

    // ---- Unique --------------------------------------------------------------------------------

    [Fact]
    public void Unique_Sorted_WithAllOutputs()
    {
        // input [2,3,1,3,5,2] -> sorted uniques [1,2,3,5]
        var c = Ctx(("x", F(new[] { 6 }, 2, 3, 1, 3, 5, 2)));
        new UniqueKernel().Execute(Node("Unique", new[] { "x" }, new[] { "y", "idx", "inv", "cnt" }), c);
        Assert.Equal(new[] { 1f, 2f, 3f, 5f }, c.GetTensor("y").AsFloat().Span.ToArray());
        // first-occurrence indices of 1,2,3,5 in input = 2,0,1,4
        Assert.Equal(new long[] { 2, 0, 1, 4 }, c.GetTensor("idx").AsInt64().Span.ToArray());
        // inverse: each input value's pos in [1,2,3,5]: 2->1,3->2,1->0,3->2,5->3,2->1
        Assert.Equal(new long[] { 1, 2, 0, 2, 3, 1 }, c.GetTensor("inv").AsInt64().Span.ToArray());
        // counts of 1,2,3,5 = 1,2,2,1
        Assert.Equal(new long[] { 1, 2, 2, 1 }, c.GetTensor("cnt").AsInt64().Span.ToArray());
    }

    [Fact]
    public void Unique_Unsorted_FirstSeenOrder()
    {
        var c = Ctx(("x", I64(new[] { 5 }, 4, 1, 4, 2, 1)));
        new UniqueKernel().Execute(Node("Unique", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["sorted"] = 0L }), c);
        Assert.Equal(new long[] { 4, 1, 2 }, c.GetTensor("y").AsInt64().Span.ToArray());
    }

    // ---- CenterCropPad -------------------------------------------------------------------------

    [Fact]
    public void CenterCropPad_Crop()
    {
        // [1,5] crop to [1,3] -> center 3 of 5 -> indices 1,2,3
        var c = Ctx(("x", F(new[] { 1, 5 }, 0, 1, 2, 3, 4)), ("shape", I64(new[] { 2 }, 1, 3)));
        new CenterCropPadKernel().Execute(Node("CenterCropPad", new[] { "x", "shape" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void CenterCropPad_Pad()
    {
        // [3] pad to [5] -> diff=2, outStart=1: [0, a, b, c, 0]
        var c = Ctx(("x", F(new[] { 3 }, 7, 8, 9)), ("shape", I64(new[] { 1 }, 5)));
        new CenterCropPadKernel().Execute(Node("CenterCropPad", new[] { "x", "shape" }, new[] { "y" }), c);
        Assert.Equal(new[] { 0f, 7f, 8f, 9f, 0f }, c.GetTensor("y").AsFloat().Span.ToArray());
    }

    [Fact]
    public void CenterCropPad_AxesSubset()
    {
        // [2,4], crop axis 1 to 2 -> center cols 1,2
        var c = Ctx(("x", F(new[] { 2, 4 }, 0, 1, 2, 3, 4, 5, 6, 7)), ("shape", I64(new[] { 1 }, 2)));
        new CenterCropPadKernel().Execute(Node("CenterCropPad", new[] { "x", "shape" }, new[] { "y" },
            new Dictionary<string, object> { ["axes"] = new long[] { 1 } }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1f, 2f, 5f, 6f }, y.AsFloat().Span.ToArray());
    }

    // ---- Dropout -------------------------------------------------------------------------------

    [Fact]
    public void Dropout_Inference_Identity_WithMask()
    {
        var c = Ctx(("x", F(new[] { 2, 2 }, 1, 2, 3, 4)));
        new DropoutKernel().Execute(Node("Dropout", new[] { "x" }, new[] { "y", "mask" }), c);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, c.GetTensor("y").AsFloat().Span.ToArray());
        Assert.All(c.GetTensor("mask").AsBool().Span.ToArray(), m => Assert.True(m));
    }

    // ---- ConvTranspose -------------------------------------------------------------------------

    [Fact]
    public void ConvTranspose_1d_Stride1()
    {
        // input [1,1,3]=[1,2,3], weight [1,1,2]=[1,1], no pad, stride1.
        // out size = (3-1)*1 + 2 = 4. Scatter: each x at positions k=0,1.
        // y[0]+=1*1; y[1]+=1*1 & 2*1=> y[1]=1+2=3 ... compute:
        // x0=1 -> y0+=1, y1+=1 ; x1=2 -> y1+=2, y2+=2 ; x2=3 -> y2+=3, y3+=3
        // y = [1, 3, 5, 3]
        var c = Ctx(("x", F(new[] { 1, 1, 3 }, 1, 2, 3)), ("w", F(new[] { 1, 1, 2 }, 1, 1)));
        new ConvTransposeKernel().Execute(Node("ConvTranspose", new[] { "x", "w" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 4 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 1f, 3f, 5f, 3f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void ConvTranspose_1d_Stride2()
    {
        // input [1,1,2]=[1,2], weight[1,1,2]=[1,1], stride2. out=(2-1)*2+2=4.
        // x0=1 at out 0,1 ; x1=2 at out 2,3 -> [1,1,2,2]
        var c = Ctx(("x", F(new[] { 1, 1, 2 }, 1, 2)), ("w", F(new[] { 1, 1, 2 }, 1, 1)));
        new ConvTransposeKernel().Execute(Node("ConvTranspose", new[] { "x", "w" }, new[] { "y" },
            new Dictionary<string, object> { ["strides"] = new long[] { 2 } }), c);
        AssertClose(new[] { 1f, 1f, 2f, 2f }, c.GetTensor("y").AsFloat().Span.ToArray());
    }

    [Fact]
    public void ConvTranspose_2d_WithBias()
    {
        // input [1,1,1,1]=[2], weight [1,1,2,2]=[1,2,3,4], bias=[10]. out [1,1,2,2].
        // single point scatters whole kernel * 2 + bias = [12, 14, 16, 18]
        var c = Ctx(
            ("x", F(new[] { 1, 1, 1, 1 }, 2)),
            ("w", F(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4)),
            ("b", F(new[] { 1 }, 10)));
        new ConvTransposeKernel().Execute(Node("ConvTranspose", new[] { "x", "w", "b" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 12f, 14f, 16f, 18f }, y.AsFloat().Span.ToArray());
    }

    // ---- GridSample ----------------------------------------------------------------------------

    [Fact]
    public void GridSample_Bilinear_Identity_AlignCorners()
    {
        // X [1,1,2,2] = [[1,2],[3,4]]. grid sampling the 4 corner pixels with align_corners.
        // corners at (-1,-1)->(0,0)=1, (1,-1)->(x=1,y=0)=2, (-1,1)->(0,1)=3, (1,1)->(1,1)=4
        var grid = F(new[] { 1, 2, 2, 2 },
            -1, -1,  1, -1,
            -1,  1,  1,  1);
        var c = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4)), ("g", grid));
        new GridSampleKernel().Execute(Node("GridSample", new[] { "x", "g" }, new[] { "y" },
            new Dictionary<string, object> { ["align_corners"] = 1L }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 1f, 2f, 3f, 4f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void GridSample_Bilinear_Center_AlignCorners()
    {
        // center (0,0) of [[1,2],[3,4]] with align_corners -> average of all four = 2.5
        var c = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4)), ("g", F(new[] { 1, 1, 1, 2 }, 0, 0)));
        new GridSampleKernel().Execute(Node("GridSample", new[] { "x", "g" }, new[] { "y" },
            new Dictionary<string, object> { ["align_corners"] = 1L }), c);
        Assert.Equal(2.5f, c.GetTensor("y").AsFloat().Span[0], 4);
    }

    [Fact]
    public void GridSample_Nearest_Zeros_OutOfBounds()
    {
        // sample far outside with zeros padding -> 0
        var c = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4)), ("g", F(new[] { 1, 1, 1, 2 }, 5, 5)));
        new GridSampleKernel().Execute(Node("GridSample", new[] { "x", "g" }, new[] { "y" },
            new Dictionary<string, object> { ["mode"] = "nearest", ["padding_mode"] = "zeros" }), c);
        Assert.Equal(0f, c.GetTensor("y").AsFloat().Span[0], 4);
    }

    // ---- MaxRoiPool ----------------------------------------------------------------------------

    [Fact]
    public void MaxRoiPool_FullRegion_2x2()
    {
        // X [1,1,4,4] flat 0..15. roi covers whole image (0,0)-(3,3), pooled 2x2.
        // bins: rows {0,1},{2,3} cols {0,1},{2,3} -> max of each quadrant:
        // q00=max(0,1,4,5)=5; q01=max(2,3,6,7)=7; q10=max(8,9,12,13)=13; q11=max(10,11,14,15)=15
        var c = Ctx(
            ("x", F(new[] { 1, 1, 4, 4 }, Enumerable.Range(0, 16).Select(i => (float)i).ToArray())),
            ("rois", F(new[] { 1, 5 }, 0, 0, 0, 3, 3)));
        new MaxRoiPoolKernel().Execute(Node("MaxRoiPool", new[] { "x", "rois" }, new[] { "y" },
            new Dictionary<string, object> { ["pooled_shape"] = new long[] { 2, 2 } }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 5f, 7f, 13f, 15f }, y.AsFloat().Span.ToArray());
    }

    // ---- Col2Im --------------------------------------------------------------------------------

    [Fact]
    public void Col2Im_InverseOf_NonOverlapping()
    {
        // image 2x2, block 2x2, stride 2 -> a single block (L=1) containing all 4 pixels.
        // input [1, 1*4, 1] holding [a,b,c,d] at rows (ky,kx)=(0,0),(0,1),(1,0),(1,1).
        // output image = [[a,b],[c,d]]
        var c = Ctx(
            ("x", F(new[] { 1, 4, 1 }, 10, 20, 30, 40)),
            ("imshape", I64(new[] { 2 }, 2, 2)),
            ("block", I64(new[] { 2 }, 2, 2)));
        new Col2ImKernel().Execute(Node("Col2Im", new[] { "x", "imshape", "block" }, new[] { "y" },
            new Dictionary<string, object> { ["strides"] = new long[] { 2, 2 } }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 10f, 20f, 30f, 40f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Col2Im_OverlapSums()
    {
        // image 1x3, block 1x2, stride 1 -> L=2 blocks. row (ky,kx)=(0,0) and (0,1).
        // input [1, 1*1*2, 2]: row0=[b0_pixel0, b1_pixel0], row1=[b0_pixel1, b1_pixel1]
        // block0 covers cols 0,1 ; block1 covers cols 1,2. Overlap at col1.
        // Let input rows: kx=0 -> [1, 3]; kx=1 -> [2, 4].
        // col0 = block0 kx0 = 1 ; col1 = block0 kx1 (2) + block1 kx0 (3) = 5 ; col2 = block1 kx1 = 4
        var c = Ctx(
            ("x", F(new[] { 1, 2, 2 }, 1, 3, 2, 4)),
            ("imshape", I64(new[] { 2 }, 1, 3)),
            ("block", I64(new[] { 2 }, 1, 2)));
        new Col2ImKernel().Execute(Node("Col2Im", new[] { "x", "imshape", "block" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 1, 3 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 1f, 5f, 4f }, y.AsFloat().Span.ToArray());
    }

    // ---- Upsample ------------------------------------------------------------------------------

    [Fact]
    public void Upsample_Nearest_2x()
    {
        // [1,1,2,2]=[[1,2],[3,4]] scale 2 on H,W -> each pixel repeated 2x2
        var c = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4)), ("s", F(new[] { 4 }, 1, 1, 2, 2)));
        new UpsampleKernel().Execute(Node("Upsample", new[] { "x", "s" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[]
        {
            1f, 1f, 2f, 2f,
            1f, 1f, 2f, 2f,
            3f, 3f, 4f, 4f,
            3f, 3f, 4f, 4f,
        }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Upsample_Linear_1d_2x()
    {
        // [1,1,2]=[0,10] scale 2 -> out size 4, asymmetric: pos = out/2 -> 0,0.5,1,1.5
        // clamp last to within [0,1]: interp(0)=0, interp(0.5)=5, interp(1)=10, interp(1.5)->lo=1 clamp ->10
        var c = Ctx(("x", F(new[] { 1, 1, 2 }, 0, 10)), ("s", F(new[] { 3 }, 1, 1, 2)));
        new UpsampleKernel().Execute(Node("Upsample", new[] { "x", "s" }, new[] { "y" },
            new Dictionary<string, object> { ["mode"] = "linear" }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 4 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 0f, 5f, 10f, 10f }, y.AsFloat().Span.ToArray());
    }

    // ---- NonMaxSuppression ---------------------------------------------------------------------

    [Fact]
    public void NonMaxSuppression_SuppressesOverlap()
    {
        // 3 boxes; box0 & box1 overlap heavily, box2 disjoint. corners (y1,x1,y2,x2).
        // box0 [0,0,1,1] score 0.9 ; box1 [0,0,1,1.1] score 0.8 (IoU high w/ box0) ; box2 [2,2,3,3] score 0.7
        var boxes = F(new[] { 1, 3, 4 },
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3);
        var scores = F(new[] { 1, 1, 3 }, 0.9f, 0.8f, 0.7f);
        var c = Ctx(
            ("boxes", boxes), ("scores", scores),
            ("maxout", I64(Array.Empty<int>(), 10)),
            ("iou", F(Array.Empty<int>(), 0.5f)));
        new NonMaxSuppressionKernel().Execute(
            Node("NonMaxSuppression", new[] { "boxes", "scores", "maxout", "iou" }, new[] { "y" }), c);
        Tensor y = c.GetTensor("y");
        // box0 kept (highest), box1 suppressed (IoU=1>0.5), box2 kept. -> 2 rows.
        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        long[] sel = y.AsInt64().Span.ToArray();
        Assert.Equal(new long[] { 0, 0, 0, 0, 0, 2 }, sel);
    }

    [Fact]
    public void NonMaxSuppression_MaxOutputLimit()
    {
        var boxes = F(new[] { 1, 2, 4 }, 0, 0, 1, 1, 5, 5, 6, 6);
        var scores = F(new[] { 1, 1, 2 }, 0.9f, 0.8f);
        var c = Ctx(
            ("boxes", boxes), ("scores", scores),
            ("maxout", I64(Array.Empty<int>(), 1)),
            ("iou", F(Array.Empty<int>(), 0.5f)));
        new NonMaxSuppressionKernel().Execute(
            Node("NonMaxSuppression", new[] { "boxes", "scores", "maxout", "iou" }, new[] { "y" }), c);
        // Only 1 allowed per class -> highest score box (box0).
        Assert.Equal(new[] { 1, 3 }, c.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 0, 0, 0 }, c.GetTensor("y").AsInt64().Span.ToArray());
    }

    // ---- Bernoulli / Multinomial (seeded, deterministic checks) --------------------------------

    [Fact]
    public void Bernoulli_P0_And_P1_Deterministic()
    {
        // p=0 -> always 0 ; p=1 -> always 1 (regardless of RNG).
        var c = Ctx(("x", F(new[] { 4 }, 0, 1, 0, 1)));
        new BernoulliKernel().Execute(Node("Bernoulli", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["seed"] = 1.0f }), c);
        Assert.Equal(new[] { 0f, 1f, 0f, 1f }, c.GetTensor("y").AsFloat().Span.ToArray());
    }

    [Fact]
    public void Bernoulli_DtypeOverride_Int64()
    {
        var c = Ctx(("x", F(new[] { 2 }, 1, 0)));
        new BernoulliKernel().Execute(Node("Bernoulli", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["seed"] = 1.0f, ["dtype"] = 7L }), c);
        Assert.Equal(ElementType.Int64, c.GetTensor("y").Dtype);
        Assert.Equal(new long[] { 1, 0 }, c.GetTensor("y").AsInt64().Span.ToArray());
    }

    [Fact]
    public void Multinomial_DegenerateDistribution_AlwaysPicksClass()
    {
        // logits strongly favor class 1 (others ~ -inf via large negatives) -> always index 1.
        var c = Ctx(("x", F(new[] { 1, 3 }, -1000f, 0f, -1000f)));
        new MultinomialKernel().Execute(Node("Multinomial", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["seed"] = 3.0f, ["sample_size"] = 5L }), c);
        Tensor y = c.GetTensor("y");
        Assert.Equal(new[] { 1, 5 }, y.Shape.Dimensions.ToArray());
        Assert.All(y.AsInt32().Span.ToArray(), v => Assert.Equal(1, v));
    }

    // ---- registry wiring -----------------------------------------------------------------------

    [Fact]
    public void Registry_Has_All_New_Ops()
    {
        KernelRegistry reg = KernelRegistry.CreateDefault();
        foreach (string op in new[]
        {
            "BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot",
            "HannWindow", "HammingWindow", "BlackmanWindow",
            "Einsum", "Det", "Unique", "CenterCropPad", "Dropout",
            "ConvTranspose", "GridSample", "MaxRoiPool", "Col2Im", "Upsample",
            "NonMaxSuppression", "Bernoulli", "Multinomial",
        })
        {
            Assert.True(reg.TryGet(op, out IKernel? k) && k is not null, $"missing registration: {op}");
        }
    }
}
