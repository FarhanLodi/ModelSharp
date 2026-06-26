using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// 2-D average pooling (NCHW). Supports kernel_shape, strides, pads, auto_pad, ceil_mode, and
/// count_include_pad. When count_include_pad=0 (default) the divisor is the number of in-bounds
/// elements in the window; when 1 it is the full kernel size (padded cells contribute 0).
/// </summary>
public sealed class AveragePoolKernel : IKernel
{
    public string OpType => "AveragePool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        if (xd.Length != 4) throw new ModelSharpException("AveragePool currently supports 4-D NCHW only.");
        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];

        int[] k = Attr.Ints(node, "kernel_shape", new[] { 1, 1 });
        int kH = k[0], kW = k[1];
        int[] strides = Attr.Ints(node, "strides", new[] { kH, kW });
        int sH = strides[0], sW = strides[1];
        bool ceilMode = Attr.Int(node, "ceil_mode", 0) != 0;
        bool countIncludePad = Attr.Int(node, "count_include_pad", 0) != 0;

        int padTop, padLeft, padBottom, padRight;
        string autoPad = Attr.Str(node, "auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            bool upper = autoPad == "SAME_UPPER";
            Nd.SamePad(H, kH, sH, 1, upper, out padTop, out padBottom);
            Nd.SamePad(W, kW, sW, 1, upper, out padLeft, out padRight);
        }
        else if (autoPad == "VALID")
        {
            padTop = padLeft = padBottom = padRight = 0;
        }
        else
        {
            int[] p = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 });
            padTop = p[0]; padLeft = p[1]; padBottom = p[2]; padRight = p[3];
        }

        int outH = OutSize(H, kH, sH, padTop, padBottom, ceilMode);
        int outW = OutSize(W, kW, sW, padLeft, padRight, ceilMode);

        var y = new Tensor<float>(new TensorShape(N, C, outH, outW));
        System.Span<float> xs = x.Span, ys = y.Span;
        int xC = H * W, xN = C * xC, yC = outH * outW, yN = C * yC;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        for (int oy = 0; oy < outH; oy++)
        for (int ox = 0; ox < outW; ox++)
        {
            float sum = 0f;
            int valid = 0;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * sH - padTop + ky;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * sW - padLeft + kx;
                    if (ix < 0 || ix >= W) continue;
                    sum += xs[n * xN + c * xC + iy * W + ix];
                    valid++;
                }
            }
            int denom = countIncludePad ? kH * kW : valid;
            ys[n * yN + c * yC + oy * outW + ox] = denom > 0 ? sum / denom : 0f;
        }

        ctx.Set(node.Outputs[0], y);
    }

    private static int OutSize(int inSize, int k, int stride, int padBegin, int padEnd, bool ceilMode)
    {
        int numer = inSize + padBegin + padEnd - k;
        return (ceilMode ? (numer + stride - 1) / stride : numer / stride) + 1;
    }
}
