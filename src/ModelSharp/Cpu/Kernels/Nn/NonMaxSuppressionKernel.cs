using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>NonMaxSuppression</c>. Inputs: <c>boxes</c> <c>[batch, spatial, 4]</c>, <c>scores</c>
/// <c>[batch, classes, spatial]</c>, and the optional scalar inputs <c>max_output_boxes_per_class</c>
/// (default 0 -&gt; none), <c>iou_threshold</c> (default 0), <c>score_threshold</c>. Output is an
/// int64 <c>[num_selected, 3]</c> tensor of <c>(batch, class, box)</c> indices. The
/// <c>center_point_box</c> attribute (default 0) selects box format: 0 = (y1, x1, y2, x2) corners,
/// 1 = (x_center, y_center, width, height).
/// </summary>
public sealed class NonMaxSuppressionKernel : IKernel
{
    public string OpType => "NonMaxSuppression";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> boxes = ctx.Get(node.Inputs[0]);
        Tensor<float> scores = ctx.Get(node.Inputs[1]);
        int centerBox = (int)Attr.Int(node, "center_point_box", 0);

        long maxPerClass = ReadScalarOrDefault(node, ctx, 2, 0);
        float iouThresh = (float)ReadScalarFloatOrDefault(node, ctx, 3, 0f);
        bool hasScoreThresh = node.Inputs.Count > 4 && !string.IsNullOrEmpty(node.Inputs[4]);
        float scoreThresh = hasScoreThresh ? (float)ReadScalarFloatOrDefault(node, ctx, 4, 0f) : 0f;

        ReadOnlySpan<int> bd = boxes.Shape.Dimensions;
        ReadOnlySpan<int> sd = scores.Shape.Dimensions;
        int batch = bd[0], spatial = bd[1];
        int classes = sd[1];
        Span<float> bs = boxes.Span;
        float[] ss = scores.Span.ToArray();
        int bBatch = spatial * 4;
        int sBatch = classes * spatial;

        var selected = new List<long>(); // flat triples (batch, class, box)

        for (int n = 0; n < batch; n++)
        for (int c = 0; c < classes; c++)
        {
            // Candidate boxes for this (batch, class), filtered by score threshold, sorted desc.
            var cand = new List<int>();
            int scoreBase = n * sBatch + c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                if (!hasScoreThresh || ss[scoreBase + i] > scoreThresh) cand.Add(i);
            }
            cand.Sort((a, b) => ss[scoreBase + b].CompareTo(ss[scoreBase + a]));

            var kept = new List<int>();
            foreach (int idx in cand)
            {
                if (maxPerClass > 0 && kept.Count >= maxPerClass) break;
                bool suppress = false;
                foreach (int k in kept)
                {
                    if (Iou(bs, n * bBatch, idx, k, centerBox) > iouThresh) { suppress = true; break; }
                }
                if (suppress) continue;
                kept.Add(idx);
                selected.Add(n); selected.Add(c); selected.Add(idx);
            }
        }

        int num = selected.Count / 3;
        ctx.Set(node.Outputs[0], Tensor<long>.FromArray(new TensorShape(num, 3), selected.ToArray()));
    }

    private static float Iou(Span<float> bs, int batchBase, int i, int j, int centerBox)
    {
        Box(bs, batchBase + i * 4, centerBox, out float ay1, out float ax1, out float ay2, out float ax2);
        Box(bs, batchBase + j * 4, centerBox, out float by1, out float bx1, out float by2, out float bx2);

        float areaA = MathF.Max(0, ax2 - ax1) * MathF.Max(0, ay2 - ay1);
        float areaB = MathF.Max(0, bx2 - bx1) * MathF.Max(0, by2 - by1);
        float ix1 = MathF.Max(ax1, bx1), iy1 = MathF.Max(ay1, by1);
        float ix2 = MathF.Min(ax2, bx2), iy2 = MathF.Min(ay2, by2);
        float iw = MathF.Max(0, ix2 - ix1), ih = MathF.Max(0, iy2 - iy1);
        float inter = iw * ih;
        float denom = areaA + areaB - inter;
        return denom <= 0 ? 0f : inter / denom;
    }

    private static void Box(Span<float> bs, int off, int centerBox,
        out float y1, out float x1, out float y2, out float x2)
    {
        if (centerBox == 1)
        {
            float xc = bs[off], yc = bs[off + 1], w = bs[off + 2], h = bs[off + 3];
            x1 = xc - w / 2; x2 = xc + w / 2; y1 = yc - h / 2; y2 = yc + h / 2;
        }
        else
        {
            float a = bs[off], b = bs[off + 1], cc = bs[off + 2], d = bs[off + 3];
            // Corners may be given in either order; normalize to min/max.
            y1 = MathF.Min(a, cc); y2 = MathF.Max(a, cc);
            x1 = MathF.Min(b, d); x2 = MathF.Max(b, d);
        }
    }

    private static long ReadScalarOrDefault(GraphNode node, GraphContext ctx, int input, long dflt)
        => node.Inputs.Count > input && !string.IsNullOrEmpty(node.Inputs[input])
            ? TensorInts.Read(ctx.GetTensor(node.Inputs[input]))[0] : dflt;

    private static double ReadScalarFloatOrDefault(GraphNode node, GraphContext ctx, int input, float dflt)
    {
        if (node.Inputs.Count <= input || string.IsNullOrEmpty(node.Inputs[input])) return dflt;
        Tensor t = ctx.GetTensor(node.Inputs[input]);
        return t.Dtype == ElementType.Float32 ? t.AsFloat().Span[0] : TensorInts.Read(t)[0];
    }
}
