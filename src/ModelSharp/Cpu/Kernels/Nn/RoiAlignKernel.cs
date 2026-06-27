using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>RoiAlign</c>: pools each region-of-interest from the feature map <c>X</c> [N,C,H,W] into
/// a fixed <c>output_height × output_width</c> grid using bilinear interpolation at sub-sampled
/// points (no snapping). Inputs: <c>X</c>, <c>rois</c> [K,4] = (x1,y1,x2,y2) in input scale,
/// <c>batch_indices</c> [K]. Output: <c>[K,C,output_height,output_width]</c>. Supports
/// <c>spatial_scale</c>, <c>sampling_ratio</c>, <c>mode</c> = avg (default) / max, and
/// <c>coordinate_transformation_mode</c> = half_pixel (default) / output_half_pixel. Float32.
/// </summary>
public sealed class RoiAlignKernel : IKernel
{
    public string OpType => "RoiAlign";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> X = ctx.Get(node.Inputs[0]);
        Tensor<float> rois = ctx.Get(node.Inputs[1]);
        long[] batchIdx = TensorInts.Read(ctx.GetTensor(node.Inputs[2]));

        ReadOnlySpan<int> xd = X.Shape.Dimensions;
        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];
        int K = rois.Shape.Dimensions[0];

        int ph = (int)Attr.Int(node, "output_height", 1);
        int pw = (int)Attr.Int(node, "output_width", 1);
        float scale = Attr.Float(node, "spatial_scale", 1f);
        int samplingRatio = (int)Attr.Int(node, "sampling_ratio", 0);
        bool maxMode = Attr.Str(node, "mode", "avg") == "max";
        // half_pixel (opset16 default) subtracts 0.5; output_half_pixel (legacy) does not.
        bool halfPixel = Attr.Str(node, "coordinate_transformation_mode", "half_pixel") == "half_pixel";
        float roiOffset = halfPixel ? 0.5f : 0f;

        Span<float> xs = X.Span, rs = rois.Span;
        var y = new float[K * C * ph * pw];
        int xN = C * H * W, xC = H * W;

        for (int k = 0; k < K; k++)
        {
            int n = (int)batchIdx[k];
            if (n < 0 || n >= N) throw new ModelSharpException($"RoiAlign: batch index {n} out of range.");

            float startX = rs[k * 4 + 0] * scale - roiOffset;
            float startY = rs[k * 4 + 1] * scale - roiOffset;
            float endX = rs[k * 4 + 2] * scale - roiOffset;
            float endY = rs[k * 4 + 3] * scale - roiOffset;

            float roiW = endX - startX;
            float roiH = endY - startY;
            if (!halfPixel)
            {
                // Legacy mode clamps RoI to a minimum size of 1.
                roiW = MathF.Max(roiW, 1f);
                roiH = MathF.Max(roiH, 1f);
            }
            float binH = roiH / ph;
            float binW = roiW / pw;

            int gridH = samplingRatio > 0 ? samplingRatio : (int)MathF.Ceiling(roiH / ph);
            int gridW = samplingRatio > 0 ? samplingRatio : (int)MathF.Ceiling(roiW / pw);
            if (gridH <= 0) gridH = 1;
            if (gridW <= 0) gridW = 1;
            float count = gridH * gridW;

            for (int c = 0; c < C; c++)
            {
                int plane = n * xN + c * xC;
                for (int oy = 0; oy < ph; oy++)
                for (int ox = 0; ox < pw; ox++)
                {
                    float acc = 0f;
                    float best = float.NegativeInfinity;
                    for (int iy = 0; iy < gridH; iy++)
                    {
                        float yy = startY + oy * binH + (iy + 0.5f) * binH / gridH;
                        for (int ix = 0; ix < gridW; ix++)
                        {
                            float xx = startX + ox * binW + (ix + 0.5f) * binW / gridW;
                            float v = Bilinear(xs, plane, xx, yy, H, W);
                            if (maxMode) best = MathF.Max(best, v);
                            else acc += v;
                        }
                    }
                    y[((k * C + c) * ph + oy) * pw + ox] = maxMode ? best : acc / count;
                }
            }
        }

        ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(K, C, ph, pw), y));
    }

    // Bilinear sample; out-of-bounds (< -1 or >= size) contributes 0, matching ONNX RoiAlign.
    private static float Bilinear(Span<float> xs, int plane, float x, float y, int H, int W)
    {
        if (y < -1f || y > H || x < -1f || x > W) return 0f;
        if (y < 0f) y = 0f;
        if (x < 0f) x = 0f;
        int y0 = (int)y, x0 = (int)x;
        int y1, x1;
        float ly, lx;
        if (y0 >= H - 1) { y1 = y0 = H - 1; ly = 0f; } else { y1 = y0 + 1; ly = y - y0; }
        if (x0 >= W - 1) { x1 = x0 = W - 1; lx = 0f; } else { x1 = x0 + 1; lx = x - x0; }
        float hy = 1f - ly, hx = 1f - lx;
        float v00 = xs[plane + y0 * W + x0];
        float v01 = xs[plane + y0 * W + x1];
        float v10 = xs[plane + y1 * W + x0];
        float v11 = xs[plane + y1 * W + x1];
        return (v00 * hx + v01 * lx) * hy + (v10 * hx + v11 * lx) * ly;
    }
}
