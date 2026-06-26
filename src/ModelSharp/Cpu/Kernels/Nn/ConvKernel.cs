using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// Convolution supporting 1-D (NCW) and 2-D (NCHW) inputs. Strides, pads, auto_pad, dilations,
/// group and an optional bias are all honored. The 1-D case (e.g. wav2vec2's feature extractor)
/// is lifted to the 2-D path with a singleton height axis (H=kH=1), so the arithmetic is shared
/// and the 2-D result is byte-for-byte what it always was.
/// </summary>
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
        int spatial = xd.Length - 2;
        if ((spatial != 1 && spatial != 2) || wd.Length != xd.Length)
            throw new ModelSharpException("Conv supports 1-D (NCW) or 2-D (NCHW) tensors only.");
        bool is1d = spatial == 1;

        // Lift the 1-D layout into the 2-D one with a singleton height axis.
        int N = xd[0];
        int H = is1d ? 1 : xd[2];
        int W = is1d ? xd[2] : xd[3];
        int cout = wd[0], cinPerGroup = wd[1];
        int kH = is1d ? 1 : wd[2];
        int kW = is1d ? wd[2] : wd[3];

        int group = (int)Attr.Int(node, "group", 1);
        int[] strides = Attr.Ints(node, "strides", is1d ? new[] { 1 } : new[] { 1, 1 });
        int[] dil = Attr.Ints(node, "dilations", is1d ? new[] { 1 } : new[] { 1, 1 });
        int sH = is1d ? 1 : strides[0];
        int sW = is1d ? strides[0] : strides[1];
        int dH = is1d ? 1 : dil[0];
        int dW = is1d ? dil[0] : dil[1];

        int padTop, padLeft, padBottom, padRight;
        string autoPad = Attr.Str(node, "auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            bool upper = autoPad == "SAME_UPPER";
            if (is1d) { padTop = padBottom = 0; }
            else Nd.SamePad(H, kH, sH, dH, upper, out padTop, out padBottom);
            Nd.SamePad(W, kW, sW, dW, upper, out padLeft, out padRight);
        }
        else if (autoPad == "VALID")
        {
            padTop = padLeft = padBottom = padRight = 0;
        }
        else if (is1d)
        {
            // 1-D pads are [begin_w, end_w]; height has no padding.
            int[] p = Attr.Ints(node, "pads", new[] { 0, 0 });
            padTop = padBottom = 0; padLeft = p[0]; padRight = p[1];
        }
        else
        {
            int[] p = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 });
            padTop = p[0]; padLeft = p[1]; padBottom = p[2]; padRight = p[3];
        }

        int outH = (H + padTop + padBottom - (dH * (kH - 1) + 1)) / sH + 1;
        int outW = (W + padLeft + padRight - (dW * (kW - 1) + 1)) / sW + 1;
        int outPerGroup = cout / group;

        // The flat NCHW layout with outH==1 is identical to the squeezed NCW layout, so 1-D
        // output is allocated already-squeezed and the indexing below is unchanged.
        var y = new Tensor<float>(is1d
            ? new TensorShape(N, cout, outW)
            : new TensorShape(N, cout, outH, outW));
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
