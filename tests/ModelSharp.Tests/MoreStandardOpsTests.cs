using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Loss;
using ModelSharp.Cpu.Kernels.Nn;
using ModelSharp.Cpu.Kernels.Rnn;
using ModelSharp.Cpu.Kernels.Shape;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the final batch of standard ops: CastLike, Scatter, RNN, AffineGrid,
/// RoiAlign, DeformConv, NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss, plus an engine-level
/// SequenceMap test. Each builds a <see cref="GraphNode"/> / <see cref="GraphContext"/> by hand and
/// checks hand-computed values against exact ONNX semantics.
/// </summary>
public class MoreStandardOpsTests
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

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    // ---- CastLike ------------------------------------------------------------------------------

    [Fact]
    public void CastLike_CastsToTargetDtype()
    {
        var ctx = Ctx(("x", F(new[] { 3 }, 1.9f, -2.1f, 3.5f)), ("like", I64(new[] { 1 }, 0L)));
        new CastLikeKernel().Execute(Node("CastLike", new[] { "x", "like" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        // truncation toward zero
        Assert.Equal(new long[] { 1, -2, 3 }, y.AsInt64().Span.ToArray());
    }

    // ---- Scatter (deprecated alias of ScatterElements) -----------------------------------------

    [Fact]
    public void Scatter_AssignsAlongAxis()
    {
        // data [2,3] zeros; indices/updates write the diagonal-ish positions along axis 1.
        var data = F(new[] { 2, 3 }, 0, 0, 0, 0, 0, 0);
        var idx = I64(new[] { 2, 2 }, 1, 0, 2, 1);
        var upd = F(new[] { 2, 2 }, 5f, 6f, 7f, 8f);
        var ctx = Ctx(("d", data), ("i", idx), ("u", upd));
        new ScatterKernel().Execute(
            Node("Scatter", new[] { "d", "i", "u" }, new[] { "y" },
                new Dictionary<string, object> { ["axis"] = 1L }), ctx);

        // row0: pos1=5, pos0=6 -> [6,5,0]; row1: pos2=7, pos1=8 -> [0,8,7]
        Assert.Equal(new[] { 6f, 5f, 0f, 0f, 8f, 7f }, ctx.Get("y").Span.ToArray());
    }

    // ---- RNN -----------------------------------------------------------------------------------

    [Fact]
    public void Rnn_Forward_Relu_NoBias_SingleStep()
    {
        // S=1,B=1,I=2,H=2,D=1. W=[[1,0],[0,1]] (identity), R=0, no bias, Relu activation.
        // Ht = Relu(Xt) => Relu([2,-3]) = [2,0].
        var X = F(new[] { 1, 1, 2 }, 2f, -3f);
        var W = F(new[] { 1, 2, 2 }, 1f, 0f, 0f, 1f);
        var R = F(new[] { 1, 2, 2 }, 0f, 0f, 0f, 0f);
        var ctx = Ctx(("X", X), ("W", W), ("R", R));
        new RnnKernel().Execute(
            Node("RNN", new[] { "X", "W", "R" }, new[] { "Y", "Yh" },
                new Dictionary<string, object> { ["activations"] = new[] { "Relu" } }), ctx);

        Assert.Equal(new[] { 2f, 0f }, ctx.Get("Yh").Span.ToArray());
        Assert.Equal(new[] { 1, 1, 1, 2 }, ctx.GetTensor("Y").Shape.Dimensions.ToArray());
    }

    [Fact]
    public void Rnn_Recurrence_TwoSteps_Tanh()
    {
        // S=2,B=1,I=1,H=1. W=[1], R=[1], no bias, Tanh.
        // step0: h = tanh(x0) = tanh(0.5)
        // step1: h = tanh(x1 + h0) = tanh(0.5 + tanh(0.5))
        var X = F(new[] { 2, 1, 1 }, 0.5f, 0.5f);
        var W = F(new[] { 1, 1, 1 }, 1f);
        var R = F(new[] { 1, 1, 1 }, 1f);
        var ctx = Ctx(("X", X), ("W", W), ("R", R));
        new RnnKernel().Execute(Node("RNN", new[] { "X", "W", "R" }, new[] { "", "Yh" }), ctx);

        float h0 = MathF.Tanh(0.5f);
        float h1 = MathF.Tanh(0.5f + h0);
        Assert.Equal(h1, ctx.Get("Yh").Span[0], 5);
    }

    // ---- AffineGrid ----------------------------------------------------------------------------

    [Fact]
    public void AffineGrid_Identity_2D_MatchesNormalizedGrid()
    {
        // theta = identity affine [[1,0,0],[0,1,0]], size [1,1,2,2], align_corners=1.
        // Base coords over a 2x2 align_corners grid are {-1,1}. Identity -> grid == base.
        var theta = F(new[] { 1, 2, 3 }, 1f, 0f, 0f, 0f, 1f, 0f);
        var size = I64(new[] { 4 }, 1, 1, 2, 2);
        var ctx = Ctx(("t", theta), ("s", size));
        new AffineGridKernel().Execute(
            Node("AffineGrid", new[] { "t", "s" }, new[] { "g" },
                new Dictionary<string, object> { ["align_corners"] = 1L }), ctx);

        Tensor g = ctx.GetTensor("g");
        Assert.Equal(new[] { 1, 2, 2, 2 }, g.Shape.Dimensions.ToArray());
        // grid[h,w] = (x,y); top-left (-1,-1), top-right (1,-1), bottom-left (-1,1), bottom-right (1,1)
        Assert.Equal(new[] { -1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f }, g.AsFloat().Span.ToArray());
    }

    // ---- RoiAlign ------------------------------------------------------------------------------

    [Fact]
    public void RoiAlign_AvgPool_ConstantPlane_ReturnsConstant()
    {
        // 1x1x4x4 plane of all 7s. A single RoI covering the whole map pooled to 2x2 must yield 7
        // everywhere regardless of sampling (bilinear of a constant field is the constant).
        var X = F(new[] { 1, 1, 4, 4 }, Enumerable.Repeat(7f, 16).ToArray());
        var rois = F(new[] { 1, 4 }, 0f, 0f, 3f, 3f);
        var batch = I64(new[] { 1 }, 0L);
        var ctx = Ctx(("X", X), ("rois", rois), ("b", batch));
        new RoiAlignKernel().Execute(
            Node("RoiAlign", new[] { "X", "rois", "b" }, new[] { "y" },
                new Dictionary<string, object>
                {
                    ["output_height"] = 2L,
                    ["output_width"] = 2L,
                    ["sampling_ratio"] = 2L,
                    ["spatial_scale"] = 1f,
                }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        foreach (float v in y.AsFloat().Span.ToArray()) Assert.Equal(7f, v, 4);
    }

    // ---- DeformConv ----------------------------------------------------------------------------

    [Fact]
    public void DeformConv_ZeroOffset_EqualsPlainConv()
    {
        // 1x1x3x3 input, 1x1x2x2 weight of ones, zero offset, no mask -> equals a plain 2x2 conv.
        // Input:
        //   1 2 3
        //   4 5 6
        //   7 8 9
        // Output[0,0] = 1+2+4+5 = 12; [0,1] = 2+3+5+6 = 16; [1,0] = 4+5+7+8 = 24; [1,1] = 5+6+8+9 = 28
        var X = F(new[] { 1, 1, 3, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        var W = F(new[] { 1, 1, 2, 2 }, 1, 1, 1, 1);
        // offset shape [N, offGroup*2*kH*kW, oH, oW] = [1, 8, 2, 2], all zero.
        var off = F(new[] { 1, 8, 2, 2 }, new float[8 * 4]);
        var ctx = Ctx(("X", X), ("W", W), ("off", off));
        new DeformConvKernel().Execute(
            Node("DeformConv", new[] { "X", "W", "off" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 12f, 16f, 24f, 28f }, y.AsFloat().Span.ToArray());
    }

    [Fact]
    public void DeformConv_IntegerOffset_ShiftsSamplePoint()
    {
        // 1x1x3x3 input, 1x1x1x1 weight=1, output 3x3 (stride1, no pad). A constant dx=+1 offset on
        // every tap shifts each sample one column right; the rightmost column samples out of bounds (0).
        var X = F(new[] { 1, 1, 3, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        var W = F(new[] { 1, 1, 1, 1 }, 1f);
        // offset [1, 2, 3, 3]: channel0 = dy (all 0), channel1 = dx (all 1).
        var offData = new float[2 * 9];
        for (int i = 9; i < 18; i++) offData[i] = 1f;
        var off = F(new[] { 1, 2, 3, 3 }, offData);
        var ctx = Ctx(("X", X), ("W", W), ("off", off));
        new DeformConvKernel().Execute(
            Node("DeformConv", new[] { "X", "W", "off" }, new[] { "y" }), ctx);

        // each position picks the value one column to the right; last column -> 0.
        Assert.Equal(new[] { 2f, 3f, 0f, 5f, 6f, 0f, 8f, 9f, 0f }, ctx.Get("y").Span.ToArray());
    }

    // ---- NegativeLogLikelihoodLoss -------------------------------------------------------------

    [Fact]
    public void Nll_MeanReduction_UnweightedAveragesNegInputs()
    {
        // input [2,3] log-probs, target [2]=[0,2]. picked = (-0.1, -0.3). mean = 0.2.
        var input = F(new[] { 2, 3 }, -0.1f, -0.5f, -0.9f, -0.7f, -0.4f, -0.3f);
        var target = I64(new[] { 2 }, 0L, 2L);
        var ctx = Ctx(("x", input), ("t", target));
        new NegativeLogLikelihoodLossKernel().Execute(
            Node("NegativeLogLikelihoodLoss", new[] { "x", "t" }, new[] { "loss" }), ctx);

        Assert.Empty(ctx.GetTensor("loss").Shape.Dimensions.ToArray()); // scalar
        Assert.Equal(0.2f, ctx.Get("loss").Span[0], 5);
    }

    [Fact]
    public void Nll_NoneReduction_PerElementAndIgnoreIndex()
    {
        // target [3]=[1,ignore=−1 modeled via ignore_index, 0]; weights default 1.
        var input = F(new[] { 3, 2 }, -0.2f, -0.8f, -0.5f, -0.5f, -0.9f, -0.1f);
        var target = I64(new[] { 3 }, 1L, -1L, 0L);
        var ctx = Ctx(("x", input), ("t", target));
        new NegativeLogLikelihoodLossKernel().Execute(
            Node("NegativeLogLikelihoodLoss", new[] { "x", "t" }, new[] { "loss" },
                new Dictionary<string, object> { ["reduction"] = "none", ["ignore_index"] = -1L }), ctx);

        // loss[0] = -input[0,1] = 0.8; loss[1] ignored = 0; loss[2] = -input[2,0] = 0.9
        Assert.Equal(new[] { 0.8f, 0f, 0.9f }, ctx.Get("loss").Span.ToArray(), new FloatCmp());
    }

    [Fact]
    public void Nll_WeightedMean_NormalizesBySumOfWeights()
    {
        // input [2,2] log-probs, target [2]=[0,1], weights=[2,1].
        // losses: -2*(-0.1)=0.2, -1*(-0.4)=0.4; weighted mean = (0.2+0.4)/(2+1) = 0.2.
        var input = F(new[] { 2, 2 }, -0.1f, -0.9f, -0.6f, -0.4f);
        var target = I64(new[] { 2 }, 0L, 1L);
        var weight = F(new[] { 2 }, 2f, 1f);
        var ctx = Ctx(("x", input), ("t", target), ("w", weight));
        new NegativeLogLikelihoodLossKernel().Execute(
            Node("NegativeLogLikelihoodLoss", new[] { "x", "t", "w" }, new[] { "loss" }), ctx);

        Assert.Equal(0.2f, ctx.Get("loss").Span[0], 5);
    }

    // ---- SoftmaxCrossEntropyLoss ---------------------------------------------------------------

    [Fact]
    public void SoftmaxCrossEntropyLoss_MatchesManualLogSoftmaxNll()
    {
        // scores [1,3] = [1,2,3], label=2. log-softmax then NLL on class 2.
        var scores = F(new[] { 1, 3 }, 1f, 2f, 3f);
        var label = I64(new[] { 1 }, 2L);
        var ctx = Ctx(("s", scores), ("l", label));
        new SoftmaxCrossEntropyLossKernel().Execute(
            Node("SoftmaxCrossEntropyLoss", new[] { "s", "l" }, new[] { "loss", "logp" }), ctx);

        // manual: logsumexp(1,2,3) = 3 + log(e^-2+e^-1+1); loss = -(3 - lse) = lse - 3
        float lse = 3f + MathF.Log(MathF.Exp(-2f) + MathF.Exp(-1f) + 1f);
        float expected = lse - 3f;
        Assert.Equal(expected, ctx.Get("loss").Span[0], 5);

        // log_prob output present, shape == scores, sums-to-ish-softmax check
        Tensor logp = ctx.GetTensor("logp");
        Assert.Equal(new[] { 1, 3 }, logp.Shape.Dimensions.ToArray());
        float sumExp = logp.AsFloat().Span.ToArray().Sum(v => MathF.Exp(v));
        Assert.Equal(1f, sumExp, 4);
    }

    // ---- SequenceMap (end-to-end through the engine) -------------------------------------------

    [Fact]
    public void SequenceMap_AddsScalar_ToEachSequenceElement()
    {
        // Build a sequence of two tensors via SequenceConstruct, map "+10" over it via SequenceMap,
        // then read element 0 and element 1 back out with SequenceAt.
        var body = new ModelGraph
        {
            Inputs = new[] { "elem" },
            Outputs = new[] { "mapped" },
            Nodes = new[] { new GraphNode("Add", "add", new[] { "elem", "ten" }, new[] { "mapped" }) },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y0", "y1" },
            Initializers = new Dictionary<string, Tensor>
            {
                ["ten"] = F(new[] { 1 }, 10f),
                ["i0"] = I64(Array.Empty<int>(), 0L),
                ["i1"] = I64(Array.Empty<int>(), 1L),
            },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "seq", new[] { "a", "b" }, new[] { "s" }),
                new GraphNode("SequenceMap", "map", new[] { "s" }, new[] { "ms" },
                    new Dictionary<string, object> { ["body"] = body }),
                new GraphNode("SequenceAt", "at0", new[] { "ms", "i0" }, new[] { "y0" }),
                new GraphNode("SequenceAt", "at1", new[] { "ms", "i1" }, new[] { "y1" }),
            },
        };

        using var engine = new ManagedCpuEngine(graph);
        var outputs = engine.Run(new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", F(new[] { 1 }, 1f)),
            ["b"] = new NamedTensor("b", F(new[] { 1 }, 2f)),
        });

        Assert.Equal(11f, outputs["y0"].Data.Span[0]);
        Assert.Equal(12f, outputs["y1"].Data.Span[0]);
    }

    private sealed class FloatCmp : IEqualityComparer<float>
    {
        public bool Equals(float a, float b) => MathF.Abs(a - b) < 1e-4f;
        public int GetHashCode(float v) => v.GetHashCode();
    }
}
