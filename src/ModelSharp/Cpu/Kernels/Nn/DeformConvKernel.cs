using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>DeformConv</c> (opset 19): 2-D deformable convolution (v2). Inputs:
/// <c>X</c> [N,C,H,W], <c>W</c> [oC, C/group, kH, kW], <c>offset</c>
/// [N, offset_group·2·kH·kW, oH, oW], optional <c>B</c> [oC], optional <c>mask</c>
/// [N, offset_group·kH·kW, oH, oW]. Each output position samples the input at kernel taps shifted
/// by the per-tap learned (dy,dx) offsets (bilinear interpolation, zero padding outside) and, when
/// a mask is given, scaled by the per-tap modulation weight. Supports <c>strides</c>, <c>pads</c>,
/// <c>dilations</c>, <c>group</c>, and <c>offset_group</c>. Float32.
/// </summary>
public sealed class DeformConvKernel : IKernel
{
    public string OpType => "DeformConv";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> X = ctx.Get(node.Inputs[0]);
        Tensor<float> Wt = ctx.Get(node.Inputs[1]);
        Tensor<float> offset = ctx.Get(node.Inputs[2]);
        Tensor<float>? B = node.Inputs.Count > 3 && node.Inputs[3].Length != 0 ? ctx.Get(node.Inputs[3]) : null;
        Tensor<float>? mask = node.Inputs.Count > 4 && node.Inputs[4].Length != 0 ? ctx.Get(node.Inputs[4]) : null;

        ReadOnlySpan<int> xd = X.Shape.Dimensions;
        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];
        ReadOnlySpan<int> wd = Wt.Shape.Dimensions;
        int oC = wd[0], cPerG = wd[1], kH = wd[2], kW = wd[3];

        int group = (int)Attr.Int(node, "group", 1);
        int offGroup = (int)Attr.Int(node, "offset_group", 1);
        int[] strides = Attr.Ints(node, "strides", new[] { 1, 1 });
        int[] dil = Attr.Ints(node, "dilations", new[] { 1, 1 });
        int[] pads = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 });
        int sH = strides[0], sW = strides[1], dH = dil[0], dW = dil[1];
        int pT = pads[0], pL = pads[1];

        ReadOnlySpan<int> od = offset.Shape.Dimensions;
        int oH = od[2], oW = od[3];
        int kArea = kH * kW;
        int cPerOffG = C / offGroup;

        Span<float> xs = X.Span, ws = Wt.Span, os = offset.Span;
        Span<float> ms = mask is null ? default : mask.Span;
        Span<float> bs = B is null ? default : B.Span;

        var y = new float[N * oC * oH * oW];
        int xN = C * H * W, xC = H * W;
        int oN = (offGroup * 2 * kArea) * oH * oW; // offset per batch
        int mN = mask is null ? 0 : (offGroup * kArea) * oH * oW;

        for (int n = 0; n < N; n++)
        for (int oc = 0; oc < oC; oc++)
        {
            int g = oc / (oC / group);
            int cStart = g * cPerG;
            for (int yh = 0; yh < oH; yh++)
            for (int yw = 0; yw < oW; yw++)
            {
                float acc = B is not null ? bs[oc] : 0f;
                for (int ci = 0; ci < cPerG; ci++)
                {
                    int c = cStart + ci;
                    int og = c / cPerOffG;            // which offset group this channel belongs to
                    int plane = n * xN + c * xC;
                    int wBase = ((oc * cPerG + ci) * kH) * kW;
                    for (int ky = 0; ky < kH; ky++)
                    for (int kx = 0; kx < kW; kx++)
                    {
                        int tap = ky * kW + kx;
                        // offset layout: [offGroup, 2, kH, kW, oH, oW]
                        int offBase = n * oN + (((og * 2 + 0) * kArea + tap) * oH + yh) * oW + yw;
                        int offBaseX = n * oN + (((og * 2 + 1) * kArea + tap) * oH + yh) * oW + yw;
                        float dy = os[offBase];
                        float dx = os[offBaseX];

                        float baseY = yh * sH - pT + ky * dH;
                        float baseX = yw * sW - pL + kx * dW;
                        float sy = baseY + dy;
                        float sx = baseX + dx;

                        float v = Bilinear(xs, plane, sx, sy, H, W);
                        if (mask is not null)
                        {
                            int mBase = n * mN + ((og * kArea + tap) * oH + yh) * oW + yw;
                            v *= ms[mBase];
                        }
                        acc += v * ws[wBase + ky * kW + kx];
                    }
                }
                y[((n * oC + oc) * oH + yh) * oW + yw] = acc;
            }
        }

        ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(N, oC, oH, oW), y));
    }

    // Bilinear sample with zero padding outside [0,H)×[0,W).
    private static float Bilinear(Span<float> xs, int plane, float x, float y, int H, int W)
    {
        if (y <= -1f || y >= H || x <= -1f || x >= W) return 0f;
        int y0 = (int)MathF.Floor(y), x0 = (int)MathF.Floor(x);
        int y1 = y0 + 1, x1 = x0 + 1;
        float ly = y - y0, lx = x - x0, hy = 1f - ly, hx = 1f - lx;
        float v00 = In(xs, plane, y0, x0, H, W);
        float v01 = In(xs, plane, y0, x1, H, W);
        float v10 = In(xs, plane, y1, x0, H, W);
        float v11 = In(xs, plane, y1, x1, H, W);
        return (v00 * hx + v01 * lx) * hy + (v10 * hx + v11 * lx) * ly;
    }

    private static float In(Span<float> xs, int plane, int y, int x, int H, int W)
        => (y < 0 || y >= H || x < 0 || x >= W) ? 0f : xs[plane + y * W + x];
}
