using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>MaxRoiPool</c>: max-pools regions of interest from a feature map to a fixed grid. <c>X</c>
/// is <c>[N,C,H,W]</c>; <c>rois</c> is <c>[num_rois, 5]</c> rows of <c>(batch_index, x1, y1, x2, y2)</c>
/// in input coordinates. The required <c>pooled_shape</c> attribute gives the output grid
/// <c>[ph, pw]</c> and <c>spatial_scale</c> (default 1) multiplies the ROI coordinates before
/// rounding. Output is <c>[num_rois, C, ph, pw]</c>. Float32.
/// </summary>
public sealed class MaxRoiPoolKernel : IKernel
{
    public string OpType => "MaxRoiPool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> rois = ctx.Get(node.Inputs[1]);
        int[] pooled = Attr.Ints(node, "pooled_shape")
            ?? throw new ModelSharpException("MaxRoiPool: 'pooled_shape' attribute is required.");
        float scale = Attr.Float(node, "spatial_scale", 1f);
        int ph = pooled[0], pw = pooled[1];

        ReadOnlySpan<int> xd = x.Shape.Dimensions;
        int C = xd[1], H = xd[2], W = xd[3];
        int numRois = rois.Shape[0];
        Span<float> xs = x.Span, rs = rois.Span;

        var y = new Tensor<float>(new TensorShape(numRois, C, ph, pw));
        Span<float> ys = y.Span;
        int xN = C * H * W, xC = H * W;
        int yR = C * ph * pw, yC = ph * pw;

        for (int r = 0; r < numRois; r++)
        {
            int n = (int)MathF.Round(rs[r * 5 + 0]);
            int x1 = (int)MathF.Round(rs[r * 5 + 1] * scale);
            int y1 = (int)MathF.Round(rs[r * 5 + 2] * scale);
            int x2 = (int)MathF.Round(rs[r * 5 + 3] * scale);
            int y2 = (int)MathF.Round(rs[r * 5 + 4] * scale);

            int roiH = Math.Max(y2 - y1 + 1, 1);
            int roiW = Math.Max(x2 - x1 + 1, 1);
            float binH = (float)roiH / ph;
            float binW = (float)roiW / pw;

            for (int c = 0; c < C; c++)
            {
                int xBase = n * xN + c * xC;
                int yBase = r * yR + c * yC;
                for (int oy = 0; oy < ph; oy++)
                for (int ox = 0; ox < pw; ox++)
                {
                    int hStart = y1 + (int)MathF.Floor(oy * binH);
                    int wStart = x1 + (int)MathF.Floor(ox * binW);
                    int hEnd = y1 + (int)MathF.Ceiling((oy + 1) * binH);
                    int wEnd = x1 + (int)MathF.Ceiling((ox + 1) * binW);
                    hStart = Math.Clamp(hStart, 0, H);
                    hEnd = Math.Clamp(hEnd, 0, H);
                    wStart = Math.Clamp(wStart, 0, W);
                    wEnd = Math.Clamp(wEnd, 0, W);

                    float best = (hStart >= hEnd || wStart >= wEnd) ? 0f : float.NegativeInfinity;
                    for (int iy = hStart; iy < hEnd; iy++)
                    for (int ix = wStart; ix < wEnd; ix++)
                    {
                        float v = xs[xBase + iy * W + ix];
                        if (v > best) best = v;
                    }
                    ys[yBase + oy * pw + ox] = best;
                }
            }
        }
        ctx.Set(node.Outputs[0], y);
    }
}
