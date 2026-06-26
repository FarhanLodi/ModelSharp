using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Activations;
using ModelSharp.Cpu.Kernels.MathOps;
using ModelSharp.Cpu.Kernels.Nn;
using ModelSharp.Cpu.Kernels.Reduction;
using ModelSharp.Cpu.Kernels.Shape;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the newly added CPU operators. Each test builds a
/// <see cref="GraphNode"/> plus a <see cref="GraphContext"/> by hand (mirroring the LSTM
/// kernel tests), runs the kernel in isolation, and checks hand-computed expected values.
/// </summary>
public class NewOpsTests
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

    // ---- elementwise unary math ----------------------------------------------------------------

    [Fact]
    public void Sin_Cos()
    {
        var ctx = Ctx(("x", F(new[] { 3 }, 0f, MathF.PI / 2f, MathF.PI)));
        new SinKernel().Execute(Node("Sin", new[] { "x" }, new[] { "y" }), ctx);
        Close(new[] { 0f, 1f, 0f }, ctx.Get("y").Span.ToArray(), 1e-5f);

        var ctx2 = Ctx(("x", F(new[] { 2 }, 0f, MathF.PI)));
        new CosKernel().Execute(Node("Cos", new[] { "x" }, new[] { "y" }), ctx2);
        Close(new[] { 1f, -1f }, ctx2.Get("y").Span.ToArray(), 1e-5f);
    }

    [Fact]
    public void Reciprocal_Floor_Ceil()
    {
        var ctx = Ctx(("x", F(new[] { 2 }, 2f, 4f)));
        new ReciprocalKernel().Execute(Node("Reciprocal", new[] { "x" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 0.5f, 0.25f }, ctx.Get("y").Span.ToArray());

        var cf = Ctx(("x", F(new[] { 3 }, 1.5f, -1.5f, 2.0f)));
        new FloorKernel().Execute(Node("Floor", new[] { "x" }, new[] { "y" }), cf);
        Assert.Equal(new[] { 1f, -2f, 2f }, cf.Get("y").Span.ToArray());

        var cc = Ctx(("x", F(new[] { 2 }, 1.2f, -1.2f)));
        new CeilKernel().Execute(Node("Ceil", new[] { "x" }, new[] { "y" }), cc);
        Assert.Equal(new[] { 2f, -1f }, cc.Get("y").Span.ToArray());
    }

    [Fact]
    public void Round_HalfToEven_And_Sign()
    {
        var ctx = Ctx(("x", F(new[] { 4 }, 0.5f, 1.5f, 2.5f, 3.5f)));
        new RoundKernel().Execute(Node("Round", new[] { "x" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 0f, 2f, 2f, 4f }, ctx.Get("y").Span.ToArray());

        var cs = Ctx(("x", F(new[] { 3 }, -3f, 0f, 5f)));
        new SignKernel().Execute(Node("Sign", new[] { "x" }, new[] { "y" }), cs);
        Assert.Equal(new[] { -1f, 0f, 1f }, cs.Get("y").Span.ToArray());
    }

    // ---- reductions ----------------------------------------------------------------------------

    private static Tensor<float> Reduce(IKernel k, Tensor<float> x, long[]? axes, long keepdims, long noop = 0)
    {
        var attrs = new Dictionary<string, object> { ["keepdims"] = keepdims };
        if (axes is not null) attrs["axes"] = axes;
        if (noop != 0) attrs["noop_with_empty_axes"] = noop;
        var ctx = Ctx(("x", x));
        k.Execute(Node(k.OpType, new[] { "x" }, new[] { "y" }, attrs), ctx);
        return ctx.Get("y");
    }

    [Fact]
    public void ReduceSum_Axes_Keepdims_NegativeAxis()
    {
        Tensor<float> x = F(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6);

        Assert.Equal(new[] { 6f, 15f }, Reduce(new ReduceSumKernel(), x, new long[] { 1 }, 0).Span.ToArray());
        Assert.Equal(new[] { 5f, 7f, 9f }, Reduce(new ReduceSumKernel(), x, new long[] { 0 }, 0).Span.ToArray());

        Tensor<float> neg = Reduce(new ReduceSumKernel(), x, new long[] { -1 }, 1);
        Assert.Equal(new[] { 2, 1 }, neg.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 6f, 15f }, neg.Span.ToArray());

        Assert.Equal(new[] { 21f }, Reduce(new ReduceSumKernel(), x, Array.Empty<long>(), 0).Span.ToArray());
    }

    [Fact]
    public void ReduceSum_NoopWithEmptyAxes_IsIdentity()
    {
        Tensor<float> x = F(new[] { 2, 2 }, 1, 2, 3, 4);
        Tensor<float> y = Reduce(new ReduceSumKernel(), x, null, 1, noop: 1);
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, y.Span.ToArray());
    }

    [Fact]
    public void ReduceSum_AxesFromInput()
    {
        var ctx = Ctx(("x", F(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)), ("axes", I64(new[] { 1 }, 1)));
        new ReduceSumKernel().Execute(
            Node("ReduceSum", new[] { "x", "axes" }, new[] { "y" },
                new Dictionary<string, object> { ["keepdims"] = 0L }), ctx);
        Assert.Equal(new[] { 6f, 15f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void ReduceMax_Min_Prod_L2()
    {
        Tensor<float> x = F(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6);
        Assert.Equal(new[] { 3f, 6f }, Reduce(new ReduceMaxKernel(), x, new long[] { 1 }, 0).Span.ToArray());
        Assert.Equal(new[] { 1f, 4f }, Reduce(new ReduceMinKernel(), x, new long[] { 1 }, 0).Span.ToArray());
        Assert.Equal(new[] { 6f, 120f }, Reduce(new ReduceProdKernel(), x, new long[] { 1 }, 0).Span.ToArray());
        Close(new[] { MathF.Sqrt(14f), MathF.Sqrt(77f) },
            Reduce(new ReduceL2Kernel(), x, new long[] { 1 }, 0).Span.ToArray());
        Close(new[] { 6f, 15f }, Reduce(new ReduceL1Kernel(), x, new long[] { 1 }, 0).Span.ToArray());
    }

    // ---- argmax / argmin -----------------------------------------------------------------------

    [Fact]
    public void ArgMax_ArgMin_Axis_Keepdims()
    {
        Tensor<float> x = F(new[] { 2, 3 }, 1, 5, 2, 4, 3, 6);

        var c1 = Ctx(("x", x));
        new ArgMaxKernel().Execute(Node("ArgMax", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L, ["keepdims"] = 0L }), c1);
        Assert.Equal(new long[] { 1, 2 }, c1.GetTensor("y").AsInt64().Span.ToArray());

        var c2 = Ctx(("x", x));
        new ArgMinKernel().Execute(Node("ArgMin", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L, ["keepdims"] = 0L }), c2);
        Assert.Equal(new long[] { 0, 1 }, c2.GetTensor("y").AsInt64().Span.ToArray());

        var c3 = Ctx(("x", x));
        new ArgMaxKernel().Execute(Node("ArgMax", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L, ["keepdims"] = 1L }), c3);
        Assert.Equal(new[] { 2, 1 }, c3.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new long[] { 1, 2 }, c3.GetTensor("y").AsInt64().Span.ToArray());
    }

    [Fact]
    public void ArgMax_SelectLastIndex()
    {
        Tensor<float> x = F(new[] { 3 }, 1, 3, 3);

        var first = Ctx(("x", x));
        new ArgMaxKernel().Execute(Node("ArgMax", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 0L, ["keepdims"] = 0L }), first);
        Assert.Equal(new long[] { 1 }, first.GetTensor("y").AsInt64().Span.ToArray());

        var last = Ctx(("x", x));
        new ArgMaxKernel().Execute(Node("ArgMax", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 0L, ["keepdims"] = 0L, ["select_last_index"] = 1L }), last);
        Assert.Equal(new long[] { 2 }, last.GetTensor("y").AsInt64().Span.ToArray());
    }

    // ---- log softmax ---------------------------------------------------------------------------

    [Fact]
    public void LogSoftmax_LastAxis()
    {
        var ctx = Ctx(("x", F(new[] { 3 }, 1, 2, 3)));
        new LogSoftmaxKernel().Execute(Node("LogSoftmax", new[] { "x" }, new[] { "y" }), ctx);
        Close(new[] { -2.407606f, -1.407606f, -0.407606f }, ctx.Get("y").Span.ToArray());
    }

    // ---- split ---------------------------------------------------------------------------------

    [Fact]
    public void Split_EqualByNumOutputs()
    {
        var ctx = Ctx(("x", F(new[] { 6 }, 1, 2, 3, 4, 5, 6)));
        new SplitKernel().Execute(Node("Split", new[] { "x" }, new[] { "a", "b", "c" },
            new Dictionary<string, object> { ["axis"] = 0L }), ctx);
        Assert.Equal(new[] { 1f, 2f }, ctx.Get("a").Span.ToArray());
        Assert.Equal(new[] { 3f, 4f }, ctx.Get("b").Span.ToArray());
        Assert.Equal(new[] { 5f, 6f }, ctx.Get("c").Span.ToArray());
    }

    [Fact]
    public void Split_Uneven_NumOutputs()
    {
        var ctx = Ctx(("x", F(new[] { 5 }, 1, 2, 3, 4, 5)));
        new SplitKernel().Execute(Node("Split", new[] { "x" }, new[] { "a", "b" },
            new Dictionary<string, object> { ["axis"] = 0L, ["num_outputs"] = 2L }), ctx);
        Assert.Equal(new[] { 1f, 2f, 3f }, ctx.Get("a").Span.ToArray());
        Assert.Equal(new[] { 4f, 5f }, ctx.Get("b").Span.ToArray());
    }

    [Fact]
    public void Split_SizesInput_Axis1()
    {
        var ctx = Ctx(("x", F(new[] { 2, 4 }, 0, 1, 2, 3, 4, 5, 6, 7)), ("split", I64(new[] { 2 }, 2, 2)));
        new SplitKernel().Execute(Node("Split", new[] { "x", "split" }, new[] { "a", "b" },
            new Dictionary<string, object> { ["axis"] = 1L }), ctx);
        Assert.Equal(new[] { 0f, 1f, 4f, 5f }, ctx.Get("a").Span.ToArray());
        Assert.Equal(new[] { 2f, 3f, 6f, 7f }, ctx.Get("b").Span.ToArray());
    }

    // ---- pad -----------------------------------------------------------------------------------

    [Fact]
    public void Pad_Constant_Rows()
    {
        var ctx = Ctx(("x", F(new[] { 2, 2 }, 1, 2, 3, 4)), ("pads", I64(new[] { 4 }, 1, 0, 1, 0)));
        new PadKernel().Execute(Node("Pad", new[] { "x", "pads" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 4, 2 }, ctx.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 0f, 0f, 1f, 2f, 3f, 4f, 0f, 0f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void Pad_Constant_Value_Cols()
    {
        var ctx = Ctx(("x", F(new[] { 2, 2 }, 1, 2, 3, 4)),
            ("pads", I64(new[] { 4 }, 0, 1, 0, 1)), ("v", F(new[] { 1 }, 9f)));
        new PadKernel().Execute(Node("Pad", new[] { "x", "pads", "v" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 9f, 1f, 2f, 9f, 9f, 3f, 4f, 9f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void Pad_Negative_Crops()
    {
        var ctx = Ctx(("x", F(new[] { 1, 4 }, 10, 20, 30, 40)), ("pads", I64(new[] { 4 }, 0, -1, 0, -1)));
        new PadKernel().Execute(Node("Pad", new[] { "x", "pads" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 1, 2 }, ctx.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 20f, 30f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void Pad_Reflect_And_Edge()
    {
        var r = Ctx(("x", F(new[] { 3 }, 1, 2, 3)), ("pads", I64(new[] { 2 }, 2, 2)));
        new PadKernel().Execute(Node("Pad", new[] { "x", "pads" }, new[] { "y" },
            new Dictionary<string, object> { ["mode"] = "reflect" }), r);
        Assert.Equal(new[] { 3f, 2f, 1f, 2f, 3f, 2f, 1f }, r.Get("y").Span.ToArray());

        var e = Ctx(("x", F(new[] { 3 }, 1, 2, 3)), ("pads", I64(new[] { 2 }, 2, 1)));
        new PadKernel().Execute(Node("Pad", new[] { "x", "pads" }, new[] { "y" },
            new Dictionary<string, object> { ["mode"] = "edge" }), e);
        Assert.Equal(new[] { 1f, 1f, 1f, 2f, 3f, 3f }, e.Get("y").Span.ToArray());
    }

    // ---- tile ----------------------------------------------------------------------------------

    [Fact]
    public void Tile_1D_And_2D()
    {
        var c1 = Ctx(("x", F(new[] { 2 }, 1, 2)), ("r", I64(new[] { 1 }, 3)));
        new TileKernel().Execute(Node("Tile", new[] { "x", "r" }, new[] { "y" }), c1);
        Assert.Equal(new[] { 1f, 2f, 1f, 2f, 1f, 2f }, c1.Get("y").Span.ToArray());

        var c2 = Ctx(("x", F(new[] { 2, 2 }, 1, 2, 3, 4)), ("r", I64(new[] { 2 }, 2, 1)));
        new TileKernel().Execute(Node("Tile", new[] { "x", "r" }, new[] { "y" }), c2);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f }, c2.Get("y").Span.ToArray());

        var c3 = Ctx(("x", F(new[] { 2, 2 }, 1, 2, 3, 4)), ("r", I64(new[] { 2 }, 1, 2)));
        new TileKernel().Execute(Node("Tile", new[] { "x", "r" }, new[] { "y" }), c3);
        Assert.Equal(new[] { 1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f }, c3.Get("y").Span.ToArray());
    }

    // ---- range ---------------------------------------------------------------------------------

    [Fact]
    public void Range_Int64()
    {
        var ctx = Ctx(("s", I64(new[] { 1 }, 1)), ("l", I64(new[] { 1 }, 8)), ("d", I64(new[] { 1 }, 2)));
        new RangeKernel().Execute(Node("Range", new[] { "s", "l", "d" }, new[] { "y" }), ctx);
        Assert.Equal(new long[] { 1, 3, 5, 7 }, ctx.GetTensor("y").AsInt64().Span.ToArray());
    }

    [Fact]
    public void Range_Float_And_Negative_Delta()
    {
        var c1 = Ctx(("s", F(new[] { 1 }, 0f)), ("l", F(new[] { 1 }, 1f)), ("d", F(new[] { 1 }, 0.3f)));
        new RangeKernel().Execute(Node("Range", new[] { "s", "l", "d" }, new[] { "y" }), c1);
        Close(new[] { 0f, 0.3f, 0.6f, 0.9f }, c1.Get("y").Span.ToArray(), 1e-5f);

        var c2 = Ctx(("s", I64(new[] { 1 }, 5)), ("l", I64(new[] { 1 }, 0)), ("d", I64(new[] { 1 }, -1)));
        new RangeKernel().Execute(Node("Range", new[] { "s", "l", "d" }, new[] { "y" }), c2);
        Assert.Equal(new long[] { 5, 4, 3, 2, 1 }, c2.GetTensor("y").AsInt64().Span.ToArray());
    }

    // ---- trilu ---------------------------------------------------------------------------------

    [Fact]
    public void Trilu_Upper_Lower_And_K()
    {
        Tensor<float> M() => F(new[] { 3, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9);

        var up = Ctx(("x", M()));
        new TriluKernel().Execute(Node("Trilu", new[] { "x" }, new[] { "y" }), up);
        Assert.Equal(new[] { 1f, 2f, 3f, 0f, 5f, 6f, 0f, 0f, 9f }, up.Get("y").Span.ToArray());

        var lo = Ctx(("x", M()));
        new TriluKernel().Execute(Node("Trilu", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["upper"] = 0L }), lo);
        Assert.Equal(new[] { 1f, 0f, 0f, 4f, 5f, 0f, 7f, 8f, 9f }, lo.Get("y").Span.ToArray());

        var upk = Ctx(("x", M()), ("k", I64(Array.Empty<int>(), 1)));
        new TriluKernel().Execute(Node("Trilu", new[] { "x", "k" }, new[] { "y" }), upk);
        Assert.Equal(new[] { 0f, 2f, 3f, 0f, 0f, 6f, 0f, 0f, 0f }, upk.Get("y").Span.ToArray());
    }

    // ---- cumsum --------------------------------------------------------------------------------

    [Fact]
    public void CumSum_Normal_Exclusive_Reverse()
    {
        Tensor<float> X() => F(new[] { 4 }, 1, 2, 3, 4);

        var n = Ctx(("x", X()), ("axis", I64(Array.Empty<int>(), 0)));
        new CumSumKernel().Execute(Node("CumSum", new[] { "x", "axis" }, new[] { "y" }), n);
        Assert.Equal(new[] { 1f, 3f, 6f, 10f }, n.Get("y").Span.ToArray());

        var ex = Ctx(("x", X()), ("axis", I64(Array.Empty<int>(), 0)));
        new CumSumKernel().Execute(Node("CumSum", new[] { "x", "axis" }, new[] { "y" },
            new Dictionary<string, object> { ["exclusive"] = 1L }), ex);
        Assert.Equal(new[] { 0f, 1f, 3f, 6f }, ex.Get("y").Span.ToArray());

        var rev = Ctx(("x", X()), ("axis", I64(Array.Empty<int>(), 0)));
        new CumSumKernel().Execute(Node("CumSum", new[] { "x", "axis" }, new[] { "y" },
            new Dictionary<string, object> { ["reverse"] = 1L }), rev);
        Assert.Equal(new[] { 10f, 9f, 7f, 4f }, rev.Get("y").Span.ToArray());
    }

    [Fact]
    public void CumSum_2D_Axis1()
    {
        var ctx = Ctx(("x", F(new[] { 2, 3 }, 1, 2, 3, 4, 5, 6)), ("axis", I64(Array.Empty<int>(), 1)));
        new CumSumKernel().Execute(Node("CumSum", new[] { "x", "axis" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 1f, 3f, 6f, 4f, 9f, 15f }, ctx.Get("y").Span.ToArray());
    }

    // ---- topk ----------------------------------------------------------------------------------

    [Fact]
    public void TopK_Largest_And_Smallest_1D()
    {
        var big = Ctx(("x", F(new[] { 4 }, 1, 4, 2, 3)), ("k", I64(new[] { 1 }, 2)));
        new TopKKernel().Execute(Node("TopK", new[] { "x", "k" }, new[] { "v", "i" }), big);
        Assert.Equal(new[] { 4f, 3f }, big.Get("v").Span.ToArray());
        Assert.Equal(new long[] { 1, 3 }, big.GetTensor("i").AsInt64().Span.ToArray());

        var small = Ctx(("x", F(new[] { 4 }, 1, 4, 2, 3)), ("k", I64(new[] { 1 }, 2)));
        new TopKKernel().Execute(Node("TopK", new[] { "x", "k" }, new[] { "v", "i" },
            new Dictionary<string, object> { ["largest"] = 0L }), small);
        Assert.Equal(new[] { 1f, 2f }, small.Get("v").Span.ToArray());
        Assert.Equal(new long[] { 0, 2 }, small.GetTensor("i").AsInt64().Span.ToArray());
    }

    [Fact]
    public void TopK_2D_Axis1()
    {
        var ctx = Ctx(("x", F(new[] { 2, 4 }, 0, 1, 2, 3, 7, 6, 5, 4)), ("k", I64(new[] { 1 }, 2)));
        new TopKKernel().Execute(Node("TopK", new[] { "x", "k" }, new[] { "v", "i" },
            new Dictionary<string, object> { ["axis"] = 1L }), ctx);
        Assert.Equal(new[] { 2, 2 }, ctx.GetTensor("v").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 3f, 2f, 7f, 6f }, ctx.Get("v").Span.ToArray());
        Assert.Equal(new long[] { 3, 2, 0, 1 }, ctx.GetTensor("i").AsInt64().Span.ToArray());
    }

    // ---- variadic ------------------------------------------------------------------------------

    [Fact]
    public void Variadic_Min_Max_Sum_Mean()
    {
        Tensor<float> A() => F(new[] { 3 }, 1, 5, 3);
        Tensor<float> B() => F(new[] { 3 }, 4, 2, 6);

        var mn = Ctx(("a", A()), ("b", B()));
        new MinKernel().Execute(Node("Min", new[] { "a", "b" }, new[] { "y" }), mn);
        Assert.Equal(new[] { 1f, 2f, 3f }, mn.Get("y").Span.ToArray());

        var mx = Ctx(("a", A()), ("b", B()));
        new MaxKernel().Execute(Node("Max", new[] { "a", "b" }, new[] { "y" }), mx);
        Assert.Equal(new[] { 4f, 5f, 6f }, mx.Get("y").Span.ToArray());

        var sm = Ctx(("a", A()), ("b", B()), ("c", F(new[] { 3 }, 1, 1, 1)));
        new SumKernel().Execute(Node("Sum", new[] { "a", "b", "c" }, new[] { "y" }), sm);
        Assert.Equal(new[] { 6f, 8f, 10f }, sm.Get("y").Span.ToArray());

        var me = Ctx(("a", A()), ("b", B()));
        new MeanKernel().Execute(Node("Mean", new[] { "a", "b" }, new[] { "y" }), me);
        Assert.Equal(new[] { 2.5f, 3.5f, 4.5f }, me.Get("y").Span.ToArray());
    }

    [Fact]
    public void Variadic_Sum_Broadcasts()
    {
        var ctx = Ctx(("a", F(new[] { 1, 3 }, 1, 2, 3)), ("b", F(new[] { 2, 1 }, 10, 20)));
        new SumKernel().Execute(Node("Sum", new[] { "a", "b" }, new[] { "y" }), ctx);
        Assert.Equal(new[] { 2, 3 }, ctx.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 11f, 12f, 13f, 21f, 22f, 23f }, ctx.Get("y").Span.ToArray());
    }

    // ---- activations ---------------------------------------------------------------------------

    [Fact]
    public void Elu_Selu_HardSigmoid()
    {
        var elu = Ctx(("x", F(new[] { 3 }, -1, 0, 2)));
        new EluKernel().Execute(Node("Elu", new[] { "x" }, new[] { "y" }), elu);
        Close(new[] { MathF.Exp(-1f) - 1f, 0f, 2f }, elu.Get("y").Span.ToArray());

        var selu = Ctx(("x", F(new[] { 2 }, 2f, -1f)));
        new SeluKernel().Execute(Node("Selu", new[] { "x" }, new[] { "y" }), selu);
        Close(new[] { 2.1014020f, -1.1113307f }, selu.Get("y").Span.ToArray(), 1e-4f);

        var hs = Ctx(("x", F(new[] { 4 }, 2f, -3f, 0f, 3f)));
        new HardSigmoidKernel().Execute(Node("HardSigmoid", new[] { "x" }, new[] { "y" }), hs);
        Close(new[] { 0.9f, 0f, 0.5f, 1f }, hs.Get("y").Span.ToArray());
    }

    [Fact]
    public void Softplus_Softsign_Mish_PRelu()
    {
        var sp = Ctx(("x", F(new[] { 2 }, 0f, 1f)));
        new SoftplusKernel().Execute(Node("Softplus", new[] { "x" }, new[] { "y" }), sp);
        Close(new[] { MathF.Log(2f), MathF.Log(1f + MathF.E) }, sp.Get("y").Span.ToArray());

        var ss = Ctx(("x", F(new[] { 2 }, 1f, -3f)));
        new SoftsignKernel().Execute(Node("Softsign", new[] { "x" }, new[] { "y" }), ss);
        Assert.Equal(new[] { 0.5f, -0.75f }, ss.Get("y").Span.ToArray());

        var mish = Ctx(("x", F(new[] { 2 }, 0f, 1f)));
        new MishKernel().Execute(Node("Mish", new[] { "x" }, new[] { "y" }), mish);
        Close(new[] { 0f, 0.8650984f }, mish.Get("y").Span.ToArray(), 1e-3f);

        var pr = Ctx(("x", F(new[] { 2 }, -2f, 3f)), ("s", F(new[] { 1 }, 0.1f)));
        new PReluKernel().Execute(Node("PRelu", new[] { "x", "s" }, new[] { "y" }), pr);
        Close(new[] { -0.2f, 3f }, pr.Get("y").Span.ToArray());
    }

    // ---- gather / scatter elements -------------------------------------------------------------

    [Fact]
    public void GatherElements_Axis0_Axis1()
    {
        var a1 = Ctx(("d", F(new[] { 2, 2 }, 1, 2, 3, 4)), ("i", I64(new[] { 2, 2 }, 0, 0, 1, 0)));
        new GatherElementsKernel().Execute(Node("GatherElements", new[] { "d", "i" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L }), a1);
        Assert.Equal(new[] { 1f, 1f, 4f, 3f }, a1.Get("y").Span.ToArray());

        var a0 = Ctx(("d", F(new[] { 2, 2 }, 1, 2, 3, 4)), ("i", I64(new[] { 2, 2 }, 0, 0, 1, 0)));
        new GatherElementsKernel().Execute(Node("GatherElements", new[] { "d", "i" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 0L }), a0);
        Assert.Equal(new[] { 1f, 2f, 3f, 2f }, a0.Get("y").Span.ToArray());
    }

    [Fact]
    public void ScatterElements_None_And_Add()
    {
        var none = Ctx(("d", F(new[] { 2, 2 }, 1, 2, 3, 4)),
            ("i", I64(new[] { 2, 1 }, 1, 0)), ("u", F(new[] { 2, 1 }, 9, 8)));
        new ScatterElementsKernel().Execute(Node("ScatterElements", new[] { "d", "i", "u" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L }), none);
        Assert.Equal(new[] { 1f, 9f, 8f, 4f }, none.Get("y").Span.ToArray());

        var add = Ctx(("d", F(new[] { 2, 2 }, 1, 2, 3, 4)),
            ("i", I64(new[] { 2, 1 }, 1, 0)), ("u", F(new[] { 2, 1 }, 9, 8)));
        new ScatterElementsKernel().Execute(Node("ScatterElements", new[] { "d", "i", "u" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = 1L, ["reduction"] = "add" }), add);
        Assert.Equal(new[] { 1f, 11f, 11f, 4f }, add.Get("y").Span.ToArray());
    }

    // ---- gather / scatter ND -------------------------------------------------------------------

    [Fact]
    public void GatherND_Elements_And_Rows()
    {
        var el = Ctx(("d", F(new[] { 2, 2 }, 0, 1, 2, 3)), ("i", I64(new[] { 2, 2 }, 0, 0, 1, 1)));
        new GatherNDKernel().Execute(Node("GatherND", new[] { "d", "i" }, new[] { "y" }), el);
        Assert.Equal(new[] { 2 }, el.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 0f, 3f }, el.Get("y").Span.ToArray());

        var rows = Ctx(("d", F(new[] { 2, 2 }, 0, 1, 2, 3)), ("i", I64(new[] { 2, 1 }, 1, 0)));
        new GatherNDKernel().Execute(Node("GatherND", new[] { "d", "i" }, new[] { "y" }), rows);
        Assert.Equal(new[] { 2, 2 }, rows.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 2f, 3f, 0f, 1f }, rows.Get("y").Span.ToArray());
    }

    [Fact]
    public void ScatterND_Elements_And_Slice()
    {
        var el = Ctx(("d", F(new[] { 4 }, 1, 2, 3, 4)),
            ("i", I64(new[] { 2, 1 }, 0, 2)), ("u", F(new[] { 2 }, 10, 30)));
        new ScatterNDKernel().Execute(Node("ScatterND", new[] { "d", "i", "u" }, new[] { "y" }), el);
        Assert.Equal(new[] { 10f, 2f, 30f, 4f }, el.Get("y").Span.ToArray());

        var slice = Ctx(("d", F(new[] { 2, 2 }, 1, 2, 3, 4)),
            ("i", I64(new[] { 1, 1 }, 0)), ("u", F(new[] { 1, 2 }, 5, 6)));
        new ScatterNDKernel().Execute(Node("ScatterND", new[] { "d", "i", "u" }, new[] { "y" }), slice);
        Assert.Equal(new[] { 5f, 6f, 3f, 4f }, slice.Get("y").Span.ToArray());
    }

    // ---- average pool --------------------------------------------------------------------------

    [Fact]
    public void AveragePool_2x2_Stride2()
    {
        var ctx = Ctx(("x", F(new[] { 1, 1, 4, 4 },
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)));
        new AveragePoolKernel().Execute(Node("AveragePool", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object> { ["kernel_shape"] = new long[] { 2, 2 }, ["strides"] = new long[] { 2, 2 } }), ctx);
        Assert.Equal(new[] { 1, 1, 2, 2 }, ctx.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 2.5f, 4.5f, 10.5f, 12.5f }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void AveragePool_CountIncludePad()
    {
        var excl = Ctx(("x", F(new[] { 1, 1, 1, 2 }, 2, 4)));
        new AveragePoolKernel().Execute(Node("AveragePool", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object>
            {
                ["kernel_shape"] = new long[] { 1, 2 }, ["strides"] = new long[] { 1, 1 },
                ["pads"] = new long[] { 0, 1, 0, 0 },
            }), excl);
        Assert.Equal(new[] { 2f, 3f }, excl.Get("y").Span.ToArray());

        var incl = Ctx(("x", F(new[] { 1, 1, 1, 2 }, 2, 4)));
        new AveragePoolKernel().Execute(Node("AveragePool", new[] { "x" }, new[] { "y" },
            new Dictionary<string, object>
            {
                ["kernel_shape"] = new long[] { 1, 2 }, ["strides"] = new long[] { 1, 1 },
                ["pads"] = new long[] { 0, 1, 0, 0 }, ["count_include_pad"] = 1L,
            }), incl);
        Assert.Equal(new[] { 1f, 3f }, incl.Get("y").Span.ToArray());
    }

    // ---- resize --------------------------------------------------------------------------------

    [Fact]
    public void Resize_Nearest_Upsample()
    {
        var ctx = Ctx(("x", F(new[] { 1, 1, 2, 2 }, 1, 2, 3, 4)),
            ("scales", F(new[] { 4 }, 1, 1, 2, 2)));
        new ResizeKernel().Execute(Node("Resize", new[] { "x", "", "scales" }, new[] { "y" },
            new Dictionary<string, object>
            {
                ["mode"] = "nearest",
                ["coordinate_transformation_mode"] = "asymmetric",
                ["nearest_mode"] = "floor",
            }), ctx);
        Assert.Equal(new[] { 1, 1, 4, 4 }, ctx.GetTensor("y").Shape.Dimensions.ToArray());
        Assert.Equal(new[]
        {
            1f, 1f, 2f, 2f,
            1f, 1f, 2f, 2f,
            3f, 3f, 4f, 4f,
            3f, 3f, 4f, 4f,
        }, ctx.Get("y").Span.ToArray());
    }

    [Fact]
    public void Resize_Linear_AlignCorners()
    {
        var ctx = Ctx(("x", F(new[] { 1, 1, 1, 2 }, 1, 3)),
            ("scales", F(new[] { 4 }, 1, 1, 1, 2)));
        new ResizeKernel().Execute(Node("Resize", new[] { "x", "", "scales" }, new[] { "y" },
            new Dictionary<string, object>
            {
                ["mode"] = "linear",
                ["coordinate_transformation_mode"] = "align_corners",
            }), ctx);
        Assert.Equal(new[] { 1, 1, 1, 4 }, ctx.GetTensor("y").Shape.Dimensions.ToArray());
        Close(new[] { 1f, 1.6666667f, 2.3333333f, 3f }, ctx.Get("y").Span.ToArray(), 1e-5f);
    }

    // ---- dtype preservation --------------------------------------------------------------------

    [Fact]
    public void Split_And_Tile_Preserve_Int64()
    {
        var sp = Ctx(("x", I64(new[] { 4 }, 5, 6, 7, 8)));
        new SplitKernel().Execute(Node("Split", new[] { "x" }, new[] { "a", "b" },
            new Dictionary<string, object> { ["axis"] = 0L }), sp);
        Assert.Equal(ElementType.Int64, sp.GetTensor("a").Dtype);
        Assert.Equal(new long[] { 5, 6 }, sp.GetTensor("a").AsInt64().Span.ToArray());
        Assert.Equal(new long[] { 7, 8 }, sp.GetTensor("b").AsInt64().Span.ToArray());

        var ti = Ctx(("x", I32(new[] { 2 }, 1, 2)), ("r", I64(new[] { 1 }, 2)));
        new TileKernel().Execute(Node("Tile", new[] { "x", "r" }, new[] { "y" }), ti);
        Assert.Equal(ElementType.Int32, ti.GetTensor("y").Dtype);
        Assert.Equal(new[] { 1, 2, 1, 2 }, ti.GetTensor("y").AsInt32().Span.ToArray());
    }
}
