using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>GridSample</c> for 2-D inputs. <c>X</c> is <c>[N,C,H,W]</c>, <c>grid</c> is
/// <c>[N,Hout,Wout,2]</c> holding (x, y) sampling coordinates normalized to <c>[-1, 1]</c>; the
/// output is <c>[N,C,Hout,Wout]</c>. Supports <c>mode</c> = bilinear (default) or nearest,
/// <c>padding_mode</c> = zeros (default) / border / reflection, and <c>align_corners</c> (default 0).
/// Float32.
/// </summary>
public sealed class GridSampleKernel : IKernel
{
    public string OpType => "GridSample";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> grid = ctx.Get(node.Inputs[1]);
        ReadOnlySpan<int> xd = x.Shape.Dimensions;
        ReadOnlySpan<int> gd = grid.Shape.Dimensions;
        if (xd.Length != 4 || gd.Length != 4 || gd[3] != 2)
            throw new ModelSharpException("GridSample: only 2-D X [N,C,H,W] and grid [N,Hout,Wout,2] are supported.");

        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];
        int Ho = gd[1], Wo = gd[2];
        string mode = Attr.Str(node, "mode", "bilinear");
        string pad = Attr.Str(node, "padding_mode", "zeros");
        bool align = Attr.Int(node, "align_corners", 0) != 0;
        bool nearest = mode is "nearest";

        var y = new Tensor<float>(new TensorShape(N, C, Ho, Wo));
        Span<float> xs = x.Span, gs = grid.Span, ys = y.Span;

        int xN = C * H * W, xC = H * W;
        int yN = C * Ho * Wo, yC = Ho * Wo;
        int gN = Ho * Wo * 2;

        for (int n = 0; n < N; n++)
        for (int oy = 0; oy < Ho; oy++)
        for (int ox = 0; ox < Wo; ox++)
        {
            int gOff = n * gN + (oy * Wo + ox) * 2;
            float gx = gs[gOff], gy = gs[gOff + 1];
            float fx = Denorm(gx, W, align);
            float fy = Denorm(gy, H, align);

            if (nearest)
            {
                int ix = (int)MathF.Round(fx);
                int iy = (int)MathF.Round(fy);
                for (int c = 0; c < C; c++)
                    ys[n * yN + c * yC + oy * Wo + ox] = Sample(xs, n * xN + c * xC, ix, iy, H, W, pad);
            }
            else
            {
                int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
                int x1 = x0 + 1, y1 = y0 + 1;
                float wx1 = fx - x0, wy1 = fy - y0;
                float wx0 = 1f - wx1, wy0 = 1f - wy1;
                for (int c = 0; c < C; c++)
                {
                    int b = n * xN + c * xC;
                    float v00 = Sample(xs, b, x0, y0, H, W, pad);
                    float v01 = Sample(xs, b, x1, y0, H, W, pad);
                    float v10 = Sample(xs, b, x0, y1, H, W, pad);
                    float v11 = Sample(xs, b, x1, y1, H, W, pad);
                    ys[n * yN + c * yC + oy * Wo + ox] =
                        (v00 * wx0 + v01 * wx1) * wy0 + (v10 * wx0 + v11 * wx1) * wy1;
                }
            }
        }
        ctx.Set(node.Outputs[0], y);
    }

    // Map normalized [-1,1] coordinate to a pixel index.
    private static float Denorm(float v, int size, bool align)
        => align ? (v + 1f) * 0.5f * (size - 1)
                 : ((v + 1f) * size - 1f) * 0.5f;

    private static float Sample(Span<float> xs, int planeBase, int ix, int iy, int H, int W, string pad)
    {
        switch (pad)
        {
            case "border":
                ix = Math.Clamp(ix, 0, W - 1);
                iy = Math.Clamp(iy, 0, H - 1);
                break;
            case "reflection":
                ix = Reflect(ix, W);
                iy = Reflect(iy, H);
                break;
            default: // zeros
                if (ix < 0 || ix >= W || iy < 0 || iy >= H) return 0f;
                break;
        }
        return xs[planeBase + iy * W + ix];
    }

    // Reflect an index into [0, size-1] (reflect without repeating the edge: period 2*size).
    private static int Reflect(int i, int size)
    {
        if (size == 1) return 0;
        int period = 2 * size;
        int m = ((i % period) + period) % period;
        return m < size ? m : period - 1 - m;
    }
}
