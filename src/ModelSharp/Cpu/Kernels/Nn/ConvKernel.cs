using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>2-D convolution (NCHW). Supports strides, pads, auto_pad, dilations, group, optional bias.</summary>
public sealed class ConvKernel : IKernel
{
    public string OpType => "Conv";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> w = ctx.Get(node.Inputs[1]);
        bool hasBias = node.Inputs.Count > 2 && node.Inputs[2].Length > 0;
        Tensor<float>? bias = hasBias ? ctx.Get(node.Inputs[2]) : null;

        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        System.ReadOnlySpan<int> wd = w.Shape.Dimensions;
        if (xd.Length != 4 || wd.Length != 4)
            throw new ModelSharpException("Conv currently supports 4-D NCHW tensors only.");

        int N = xd[0], H = xd[2], W = xd[3];
        int cout = wd[0], cinPerGroup = wd[1], kH = wd[2], kW = wd[3];

        int group = (int)Attr.Int(node, "group", 1);
        int[] strides = Attr.Ints(node, "strides", new[] { 1, 1 });
        int[] dil = Attr.Ints(node, "dilations", new[] { 1, 1 });
        int sH = strides[0], sW = strides[1], dH = dil[0], dW = dil[1];

        int padTop, padLeft, padBottom, padRight;
        string autoPad = Attr.Str(node, "auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            bool upper = autoPad == "SAME_UPPER";
            Nd.SamePad(H, kH, sH, dH, upper, out padTop, out padBottom);
            Nd.SamePad(W, kW, sW, dW, upper, out padLeft, out padRight);
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

        int outH = (H + padTop + padBottom - (dH * (kH - 1) + 1)) / sH + 1;
        int outW = (W + padLeft + padRight - (dW * (kW - 1) + 1)) / sW + 1;
        int outPerGroup = cout / group;

        var y = new Tensor<float>(new TensorShape(N, cout, outH, outW));
        System.Span<float> xs = x.Span, ws = w.Span, ys = y.Span;
        System.Span<float> bs = bias is null ? default : bias.Span;

        int xN = xd[1] * H * W, xC = H * W;
        int wO = cinPerGroup * kH * kW, wC = kH * kW;
        int yN = cout * outH * outW, yC = outH * outW;

        for (int n = 0; n < N; n++)
        for (int g = 0; g < group; g++)
        for (int ocg = 0; ocg < outPerGroup; ocg++)
        {
            int oc = g * outPerGroup + ocg;
            float b0 = bias is null ? 0f : bs[oc];
            for (int oy = 0; oy < outH; oy++)
            for (int ox = 0; ox < outW; ox++)
            {
                float sum = b0;
                for (int icg = 0; icg < cinPerGroup; icg++)
                {
                    int ic = g * cinPerGroup + icg;
                    int xBase = n * xN + ic * xC;
                    int wBase = oc * wO + icg * wC;
                    for (int ky = 0; ky < kH; ky++)
                    {
                        int iy = oy * sH - padTop + ky * dH;
                        if (iy < 0 || iy >= H) continue;
                        int xRow = xBase + iy * W;
                        int wRow = wBase + ky * kW;
                        for (int kx = 0; kx < kW; kx++)
                        {
                            int ix = ox * sW - padLeft + kx * dW;
                            if (ix < 0 || ix >= W) continue;
                            sum += xs[xRow + ix] * ws[wRow + kx];
                        }
                    }
                }
                ys[n * yN + oc * yC + oy * outW + ox] = sum;
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}
