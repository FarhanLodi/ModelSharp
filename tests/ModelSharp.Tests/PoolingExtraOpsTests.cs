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
/// Direct-kernel tests for the extra pooling / argmax operators (GlobalMaxPool, LpPool,
/// GlobalLpPool, Hardmax, MaxUnpool). Each test builds a <see cref="GraphNode"/> plus a
/// <see cref="GraphContext"/> by hand and checks hand-computed expected values.
/// </summary>
public class PoolingExtraOpsTests
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

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    // ---- GlobalMaxPool -------------------------------------------------------------------------

    [Fact]
    public void GlobalMaxPool_TakesMaxOverSpatial()
    {
        // [1,2,2,2]: channel 0 = {1,2,3,4}, channel 1 = {-1,-2,-8,0}.
        var ctx = Ctx(("x", F(new[] { 1, 2, 2, 2 }, 1, 2, 3, 4, -1, -2, -8, 0)));
        new GlobalMaxPoolKernel().Execute(Node("GlobalMaxPool", new[] { "x" }, new[] { "y" }), ctx);

        Tensor<float> y = ctx.Get("y");
        Assert.Equal(new[] { 1, 2, 1, 1 }, y.Shape.Dimensions.ToArray());
        Close(new[] { 4f, 0f }, y.Span.ToArray());
    }

    // ---- LpPool --------------------------------------------------------------------------------

    [Fact]
    public void LpPool_P2_KnownWindow()
    {
        // [1,1,2,2] single 2x2 window, p=2 → sqrt(3^2 + 4^2 + 0 + 0) = 5.
        var attrs = new Dictionary<string, object>
        {
            ["kernel_shape"] = new long[] { 2, 2 },
            ["strides"] = new long[] { 2, 2 },
            ["p"] = 2f,
        };
        var ctx = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 3, 4, 0, 0)));
        new LpPoolKernel().Execute(Node("LpPool", new[] { "x" }, new[] { "y" }, attrs), ctx);

        Tensor<float> y = ctx.Get("y");
        Assert.Equal(new[] { 1, 1, 1, 1 }, y.Shape.Dimensions.ToArray());
        Close(new[] { 5f }, y.Span.ToArray());
    }

    [Fact]
    public void LpPool_P1_AbsSum()
    {
        // p=1 over a 2x2 window with negatives → |1|+|−2|+|3|+|−4| = 10.
        var attrs = new Dictionary<string, object>
        {
            ["kernel_shape"] = new long[] { 2, 2 },
            ["strides"] = new long[] { 2, 2 },
            ["p"] = 1f,
        };
        var ctx = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 1, -2, 3, -4)));
        new LpPoolKernel().Execute(Node("LpPool", new[] { "x" }, new[] { "y" }, attrs), ctx);
        Close(new[] { 10f }, ctx.Get("y").Span.ToArray());
    }

    // ---- GlobalLpPool --------------------------------------------------------------------------

    [Fact]
    public void GlobalLpPool_P2_NormOverSpatial()
    {
        // [1,1,2,2] = {3,4,0,0} → sqrt(9+16) = 5.
        var attrs = new Dictionary<string, object> { ["p"] = 2f };
        var ctx = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 3, 4, 0, 0)));
        new GlobalLpPoolKernel().Execute(Node("GlobalLpPool", new[] { "x" }, new[] { "y" }, attrs), ctx);

        Tensor<float> y = ctx.Get("y");
        Assert.Equal(new[] { 1, 1, 1, 1 }, y.Shape.Dimensions.ToArray());
        Close(new[] { 5f }, y.Span.ToArray());
    }

    // ---- Hardmax -------------------------------------------------------------------------------

    [Fact]
    public void Hardmax_AxisLast_OneHot()
    {
        // [2,3], axis=-1 (default). Row maxima at index 2 and index 0 (tie → first wins).
        var ctx = Ctx(("x", F(new[] { 2, 3 }, 1, 2, 3, 5, 5, 1)));
        new HardmaxKernel().Execute(Node("Hardmax", new[] { "x" }, new[] { "y" }), ctx);

        Tensor<float> y = ctx.Get("y");
        Assert.Equal(new[] { 2, 3 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 0f, 0f, 1f, 1f, 0f, 0f }, y.Span.ToArray());
    }

    [Fact]
    public void Hardmax_Axis0_OneHotPerColumn()
    {
        // [2,3], axis=0: column-wise argmax. Column maxima at rows 1,0,1.
        var attrs = new Dictionary<string, object> { ["axis"] = 0L };
        var ctx = Ctx(("x", F(new[] { 2, 3 }, 1, 9, 2, 4, 5, 6)));
        new HardmaxKernel().Execute(Node("Hardmax", new[] { "x" }, new[] { "y" }, attrs), ctx);

        // row0: [0,1,0], row1: [1,0,1]
        Assert.Equal(new[] { 0f, 1f, 0f, 1f, 0f, 1f }, ctx.Get("y").Span.ToArray());
    }

    // ---- MaxUnpool -----------------------------------------------------------------------------

    [Fact]
    public void MaxUnpool_RoundTripWithMaxPoolIndices()
    {
        // 4x4 input, 2x2 kernel, stride 2 → 2x2 MaxPool output. Maxima at flat positions:
        //   window (0,0): rows 0-1 cols 0-1, max value 6 at (1,1) → flat 5
        //   window (0,1): cols 2-3,            max value 8 at (1,3) → flat 7
        //   window (1,0): rows 2-3 cols 0-1,   max value 14 at (3,1) → flat 13
        //   window (1,1): cols 2-3,            max value 16 at (3,3) → flat 15
        Tensor<float> vals = F(new[] { 1, 1, 2, 2 }, 6, 8, 14, 16);
        Tensor<long> idx = I64(new[] { 1, 1, 2, 2 }, 5, 7, 13, 15);

        var attrs = new Dictionary<string, object>
        {
            ["kernel_shape"] = new long[] { 2, 2 },
            ["strides"] = new long[] { 2, 2 },
        };
        var ctx = Ctx(("x", vals), ("i", idx));
        new MaxUnpoolKernel().Execute(Node("MaxUnpool", new[] { "x", "i" }, new[] { "y" }, attrs), ctx);

        Tensor<float> y = ctx.Get("y");
        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Shape.Dimensions.ToArray());

        var expected = new float[16];
        expected[5] = 6f; expected[7] = 8f; expected[13] = 14f; expected[15] = 16f;
        Assert.Equal(expected, y.Span.ToArray());
    }

    [Fact]
    public void MaxUnpool_UsesOutputShapeInput()
    {
        Tensor<float> vals = F(new[] { 1, 1, 1, 1 }, 9f);
        Tensor<long> idx = I64(new[] { 1, 1, 1, 1 }, 4); // center of a 3x3 plane
        Tensor<long> outShape = I64(new[] { 4 }, 1, 1, 3, 3);

        var ctx = Ctx(("x", vals), ("i", idx), ("s", outShape));
        new MaxUnpoolKernel().Execute(Node("MaxUnpool", new[] { "x", "i", "s" }, new[] { "y" }), ctx);

        Tensor<float> y = ctx.Get("y");
        Assert.Equal(new[] { 1, 1, 3, 3 }, y.Shape.Dimensions.ToArray());
        var expected = new float[9];
        expected[4] = 9f;
        Assert.Equal(expected, y.Span.ToArray());
    }

    // ---- registration --------------------------------------------------------------------------

    [Fact]
    public void AddPoolingExtraOps_RegistersAllFive()
    {
        var r = new KernelRegistry().AddPoolingExtraOps();
        foreach (string op in new[] { "GlobalMaxPool", "LpPool", "GlobalLpPool", "Hardmax", "MaxUnpool" })
        {
            Assert.True(r.TryGet(op, out IKernel? k));
            Assert.NotNull(k);
            Assert.Equal(op, k!.OpType);
        }
    }
}
