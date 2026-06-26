using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Shape;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the shape / data-movement op family: OneHot, EyeLike, Compress,
/// DepthToSpace, SpaceToDepth, ReverseSequence. Each test builds a <see cref="GraphNode"/> and
/// <see cref="GraphContext"/> by hand (mirroring <c>NewOpsTests</c>) and checks hand-computed values.
/// </summary>
public class DataMovementOpsTests
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

    private static Tensor<bool> B(int[] dims, params bool[] data) =>
        Tensor<bool>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static float[] Range(int n)
    {
        var r = new float[n];
        for (int i = 0; i < n; i++) r[i] = i;
        return r;
    }

    // ---- OneHot --------------------------------------------------------------------------------

    [Fact]
    public void OneHot_DefaultAxis_LastAxisAppended()
    {
        // indices [3] = [1, 0, 2], depth 3, values [off=0, on=1], axis=-1.
        // Output [3,3], identity-ish rows for each index.
        var ctx = Ctx(
            ("ind", I64(new[] { 3 }, 1, 0, 2)),
            ("depth", I64(Array.Empty<int>(), 3)),
            ("vals", F(new[] { 2 }, 0f, 1f)));
        new OneHotKernel().Execute(Node("OneHot", new[] { "ind", "depth", "vals" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 3, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[]
        {
            0f, 1f, 0f,
            1f, 0f, 0f,
            0f, 0f, 1f,
        }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void OneHot_Axis1_NewAxisInserted()
    {
        // indices [2] = [2, 0], depth 3, axis=1 -> output [2,3].
        var ctx = Ctx(
            ("ind", I64(new[] { 2 }, 2, 0)),
            ("depth", I64(Array.Empty<int>(), 3)),
            ("vals", F(new[] { 2 }, -1f, 5f)));
        new OneHotKernel().Execute(Node("OneHot", new[] { "ind", "depth", "vals" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[]
        {
            -1f, -1f, 5f,   // index 2 -> on at position 2
            5f, -1f, -1f,   // index 0 -> on at position 0
        }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void OneHot_NegativeIndex_Wraps()
    {
        // index -1 with depth 3 -> position 2.
        var ctx = Ctx(
            ("ind", I64(new[] { 1 }, -1)),
            ("depth", I64(Array.Empty<int>(), 3)),
            ("vals", I64(new[] { 2 }, 0, 1)));
        new OneHotKernel().Execute(Node("OneHot", new[] { "ind", "depth", "vals" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new long[] { 0, 0, 1 }, y.AsInt64().Span.ToArray());
    }

    // ---- EyeLike -------------------------------------------------------------------------------

    [Fact]
    public void EyeLike_K0_Identity()
    {
        var ctx = Ctx(("x", F(new[] { 3, 3 }, Range(9))));
        new EyeLikeKernel().Execute(Node("EyeLike", new[] { "x" }, new[] { "y" }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[]
        {
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f,
        }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void EyeLike_K1_SuperDiagonal()
    {
        var ctx = Ctx(("x", F(new[] { 3, 3 }, Range(9))));
        new EyeLikeKernel().Execute(Node("EyeLike", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["k"] = 1L }), ctx);
        Assert.Equal(new[]
        {
            0f, 1f, 0f,
            0f, 0f, 1f,
            0f, 0f, 0f,
        }, ctx.GetTensor("y").AsFloat().Span.ToArray());
    }

    [Fact]
    public void EyeLike_NonSquare_KNegative_DtypeOverride()
    {
        // 3x2, k=-1: ones at (1,0) and (2,1). dtype=7 (Int64).
        var ctx = Ctx(("x", F(new[] { 3, 2 }, Range(6))));
        new EyeLikeKernel().Execute(Node("EyeLike", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["k"] = -1L, ["dtype"] = 7L }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new[] { 3, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new long[]
        {
            0, 0,
            1, 0,
            0, 1,
        }, y.AsInt64().Span.ToArray());
    }

    // ---- Compress ------------------------------------------------------------------------------

    [Fact]
    public void Compress_NoAxis_FlattensAndSelects()
    {
        // input [3,2] flattened = [0..5], condition selects flat positions 0,1,4.
        var ctx = Ctx(
            ("x", F(new[] { 3, 2 }, 0, 1, 2, 3, 4, 5)),
            ("cond", B(new[] { 6 }, true, true, false, false, true, false)));
        new CompressKernel().Execute(Node("Compress", new[] { "x", "cond" }, new[] { "y" }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 0f, 1f, 4f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Compress_Axis0_SelectsRows()
    {
        // input [3,2], condition over axis 0 selects rows 0 and 2.
        var ctx = Ctx(
            ("x", F(new[] { 3, 2 }, 1, 2, 3, 4, 5, 6)),
            ("cond", B(new[] { 3 }, true, false, true)));
        new CompressKernel().Execute(Node("Compress", new[] { "x", "cond" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 0L }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1f, 2f, 5f, 6f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void Compress_Axis1_ShortCondition()
    {
        // input [2,3], condition over axis 1 length 2 (shorter than 3) -> selects col 1 only.
        var ctx = Ctx(
            ("x", F(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)),
            ("cond", B(new[] { 2 }, false, true)));
        new CompressKernel().Execute(Node("Compress", new[] { "x", "cond" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 2, 1 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 2f, 5f }, y.AsFloat().Span.ToArray());
    }

    // ---- DepthToSpace --------------------------------------------------------------------------

    private static readonly float[] D2sDcrExpected =
    {
        0f, 4f, 1f, 5f,
        8f, 12f, 9f, 13f,
        2f, 6f, 3f, 7f,
        10f, 14f, 11f, 15f,
    };

    [Fact]
    public void DepthToSpace_DCR_1x4x2x2()
    {
        // Input [1,4,2,2] with channel c_k = [[4k,4k+1],[4k+2,4k+3]], flat 0..15.
        var ctx = Ctx(("x", F(new[] { 1, 4, 2, 2 }, Range(16))));
        new DepthToSpaceKernel().Execute(Node("DepthToSpace", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["blocksize"] = 2L, ["mode"] = "DCR" }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(D2sDcrExpected, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void DepthToSpace_CRD_1x4x2x2_EqualsDcrWhenSingleOutChannel()
    {
        // With C/(b*b) == 1, DCR and CRD coincide; the result matches the DCR case above.
        var ctx = Ctx(("x", F(new[] { 1, 4, 2, 2 }, Range(16))));
        new DepthToSpaceKernel().Execute(Node("DepthToSpace", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["blocksize"] = 2L, ["mode"] = "CRD" }), ctx);
        Assert.Equal(D2sDcrExpected, ctx.GetTensor("y").AsFloat().Span.ToArray());
    }

    [Fact]
    public void DepthToSpace_CRD_1x8x2x2_DistinctFromDcr()
    {
        // Two output channels expose the DCR/CRD difference.
        // CRD: c_in = ch*(b*b) + by*b + bx, so ch=0 uses channels 0..3, ch=1 uses 4..7.
        var ctx = Ctx(("x", F(new[] { 1, 8, 2, 2 }, Range(32))));
        new DepthToSpaceKernel().Execute(Node("DepthToSpace", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["blocksize"] = 2L, ["mode"] = "CRD" }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 1, 2, 4, 4 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[]
        {
            // output channel 0 (from input channels 0..3)
            0f, 4f, 1f, 5f,
            8f, 12f, 9f, 13f,
            2f, 6f, 3f, 7f,
            10f, 14f, 11f, 15f,
            // output channel 1 (from input channels 4..7)
            16f, 20f, 17f, 21f,
            24f, 28f, 25f, 29f,
            18f, 22f, 19f, 23f,
            26f, 30f, 27f, 31f,
        }, y.AsFloat().Span.ToArray());
    }

    // ---- SpaceToDepth --------------------------------------------------------------------------

    [Fact]
    public void SpaceToDepth_InverseOfDepthToSpace_DCR()
    {
        // Round-trip: SpaceToDepth(DepthToSpace_DCR(x)) == x.
        var x = F(new[] { 1, 4, 2, 2 }, Range(16));
        var c1 = Ctx(("x", x));
        new DepthToSpaceKernel().Execute(Node("DepthToSpace", new[] { "x" }, new[] { "m" },
            new Dictionary<string, object> { ["blocksize"] = 2L, ["mode"] = "DCR" }), c1);
        Tensor mid = c1.GetTensor("m");
        Assert.Equal(new[] { 1, 1, 4, 4 }, mid.Shape.Dimensions.ToArray());

        var c2 = Ctx(("m", mid));
        new SpaceToDepthKernel().Execute(Node("SpaceToDepth", new[] { "m" }, new[] { "y" },
            new Dictionary<string, object> { ["blocksize"] = 2L }), c2);
        Tensor y = c2.GetTensor("y");
        Assert.Equal(new[] { 1, 4, 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(Range(16), y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void SpaceToDepth_DirectComputation()
    {
        // Input [1,1,4,4] flat 0..15. blocksize 2 -> [1,4,2,2].
        // Output channel c=(by*b+bx)*1+0, so the four DCR sub-grids of the 4x4 image.
        var ctx = Ctx(("x", F(new[] { 1, 1, 4, 4 }, Range(16))));
        new SpaceToDepthKernel().Execute(Node("SpaceToDepth", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["blocksize"] = 2L }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 1, 4, 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[]
        {
            // channel 0 (by=0,bx=0): rows/cols 0,2 -> 0,2,8,10
            0f, 2f, 8f, 10f,
            // channel 1 (by=0,bx=1): cols 1,3 -> 1,3,9,11
            1f, 3f, 9f, 11f,
            // channel 2 (by=1,bx=0): rows 1,3 -> 4,6,12,14
            4f, 6f, 12f, 14f,
            // channel 3 (by=1,bx=1): -> 5,7,13,15
            5f, 7f, 13f, 15f,
        }, y.AsFloat().Span.ToArray());
    }

    // ---- ReverseSequence -----------------------------------------------------------------------

    [Fact]
    public void ReverseSequence_DefaultAxes_TimeMajor()
    {
        // Input [time=4, batch=2]. time_axis=0, batch_axis=1 (defaults).
        // Column 0 (batch 0) seq_len=4 -> fully reversed; column 1 (batch 1) seq_len=2 -> first 2 reversed.
        //   time:  0  1  2  3
        //   b0:    0  2  4  6
        //   b1:    1  3  5  7
        var ctx = Ctx(
            ("x", F(new[] { 4, 2 }, 0, 1, 2, 3, 4, 5, 6, 7)),
            ("lens", I64(new[] { 2 }, 4, 2)));
        new ReverseSequenceKernel().Execute(Node("ReverseSequence", new[] { "x", "lens" }, new[] { "y" }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 4, 2 }, y.Shape.Dimensions.ToArray());
        // b0 reversed: 6,4,2,0 ; b1 first-2 reversed then tail: 3,1,5,7.
        Assert.Equal(new[]
        {
            6f, 3f,
            4f, 1f,
            2f, 5f,
            0f, 7f,
        }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void ReverseSequence_BatchMajor()
    {
        // Input [batch=2, time=3]. batch_axis=0, time_axis=1.
        //   b0 = [0,1,2] seq_len 3 -> [2,1,0]
        //   b1 = [3,4,5] seq_len 1 -> unchanged [3,4,5]
        var ctx = Ctx(
            ("x", F(new[] { 2, 3 }, 0, 1, 2, 3, 4, 5)),
            ("lens", I64(new[] { 2 }, 3, 1)));
        new ReverseSequenceKernel().Execute(Node("ReverseSequence", new[] { "x", "lens" }, new[] { "y" },
            new Dictionary<string, object> { ["batch_axis"] = 0L, ["time_axis"] = 1L }), ctx);
        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[]
        {
            2f, 1f, 0f,
            3f, 4f, 5f,
        }, y.AsFloat().Span.ToArray());
    }

    // ---- dtype preservation --------------------------------------------------------------------

    [Fact]
    public void Compress_And_ReverseSequence_Preserve_Int32()
    {
        var cp = Ctx(("x", I32(new[] { 4 }, 5, 6, 7, 8)),
            ("cond", B(new[] { 4 }, false, true, true, false)));
        new CompressKernel().Execute(Node("Compress", new[] { "x", "cond" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 0L }), cp);
        Assert.Equal(ElementType.Int32, cp.GetTensor("y").Dtype);
        Assert.Equal(new[] { 6, 7 }, cp.GetTensor("y").AsInt32().Span.ToArray());

        var rs = Ctx(("x", I32(new[] { 2, 3 }, 0, 1, 2, 3, 4, 5)), ("lens", I64(new[] { 2 }, 2, 3)));
        new ReverseSequenceKernel().Execute(Node("ReverseSequence", new[] { "x", "lens" }, new[] { "y" },
            new Dictionary<string, object> { ["batch_axis"] = 0L, ["time_axis"] = 1L }), rs);
        Assert.Equal(ElementType.Int32, rs.GetTensor("y").Dtype);
        // b0 seq2 -> [1,0,2]; b1 seq3 -> [5,4,3].
        Assert.Equal(new[] { 1, 0, 2, 5, 4, 3 }, rs.GetTensor("y").AsInt32().Span.ToArray());
    }
}
