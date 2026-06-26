using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>ConvTranspose</c> (transposed / fractionally-strided convolution) for 1-D (NCW) and
/// 2-D (NCHW) inputs. The weight layout is <c>[C_in, C_out/group, kH, kW]</c>. Strides, pads,
/// dilations, group, output_padding and an optional bias are honored. The 1-D case is lifted to the
/// 2-D path with a singleton height axis. Output spatial size is
/// <c>(in-1)*stride - padBegin - padEnd + dilation*(k-1) + 1 + output_padding</c>.
/// </summary>
public sealed class ConvTransposeKernel : IKernel
{
    public string OpType => "ConvTranspose";

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
            throw new ModelSharpException("ConvTranspose supports 1-D (NCW) or 2-D (NCHW) tensors only.");
        bool is1d = spatial == 1;

        int N = xd[0];
        int H = is1d ? 1 : xd[2];
        int W = is1d ? xd[2] : xd[3];
        int cin = wd[0];
        int coutPerGroup = wd[1];
        int kH = is1d ? 1 : wd[2];
        int kW = is1d ? wd[2] : wd[3];

        int group = (int)Attr.Int(node, "group", 1);
        int[] strides = Attr.Ints(node, "strides", is1d ? new[] { 1 } : new[] { 1, 1 });
        int[] dil = Attr.Ints(node, "dilations", is1d ? new[] { 1 } : new[] { 1, 1 });
        int[] outPad = Attr.Ints(node, "output_padding", is1d ? new[] { 0 } : new[] { 0, 0 });
        int sH = is1d ? 1 : strides[0];
        int sW = is1d ? strides[0] : strides[1];
        int dH = is1d ? 1 : dil[0];
        int dW = is1d ? dil[0] : dil[1];
        int opH = is1d ? 0 : outPad[0];
        int opW = is1d ? outPad[0] : outPad[1];

        int padTop, padLeft, padBottom, padRight;
        if (is1d)
        {
            int[] p = Attr.Ints(node, "pads", new[] { 0, 0 });
            padTop = padBottom = 0; padLeft = p[0]; padRight = p[1];
        }
        else
        {
            int[] p = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 });
            padTop = p[0]; padLeft = p[1]; padBottom = p[2]; padRight = p[3];
        }

        int cout = coutPerGroup * group;
        int cinPerGroup = cin / group;

        int outH = (H - 1) * sH - padTop - padBottom + dH * (kH - 1) + 1 + opH;
        int outW = (W - 1) * sW - padLeft - padRight + dW * (kW - 1) + 1 + opW;

        // explicit output_shape attribute overrides computed spatial sizes (pads inferred away).
        int[]? outShapeAttr = Attr.Ints(node, "output_shape");
        if (outShapeAttr is not null)
        {
            if (is1d) outW = outShapeAttr[outShapeAttr.Length - 1];
            else { outH = outShapeAttr[outShapeAttr.Length - 2]; outW = outShapeAttr[outShapeAttr.Length - 1]; }
        }

        var y = new Tensor<float>(is1d
            ? new TensorShape(N, cout, outW)
            : new TensorShape(N, cout, outH, outW));
        System.Span<float> xs = x.Span, ws = w.Span, ys = y.Span;
        System.Span<float> bs = bias is null ? default : bias.Span;

        int xN = cin * H * W, xC = H * W;
        int wIn = coutPerGroup * kH * kW, wOut = kH * kW;
        int yN = cout * outH * outW, yC = outH * outW;

        // Scatter-add: each input element distributes over the kernel footprint in the output.
        for (int n = 0; n < N; n++)
        for (int g = 0; g < group; g++)
        for (int icg = 0; icg < cinPerGroup; icg++)
        {
            int ic = g * cinPerGroup + icg;
            int xBase = n * xN + ic * xC;
            int wBase = ic * wIn;
            for (int iy = 0; iy < H; iy++)
            for (int ix = 0; ix < W; ix++)
            {
                float v = xs[xBase + iy * W + ix];
                if (v == 0f) continue;
                for (int ocg = 0; ocg < coutPerGroup; ocg++)
                {
                    int oc = g * coutPerGroup + ocg;
                    int wOcBase = wBase + ocg * wOut;
                    int yBase = n * yN + oc * yC;
                    for (int ky = 0; ky < kH; ky++)
                    {
                        int oy = iy * sH - padTop + ky * dH;
                        if (oy < 0 || oy >= outH) continue;
                        int wRow = wOcBase + ky * kW;
                        int yRow = yBase + oy * outW;
                        for (int kx = 0; kx < kW; kx++)
                        {
                            int ox = ix * sW - padLeft + kx * dW;
                            if (ox < 0 || ox >= outW) continue;
                            ys[yRow + ox] += v * ws[wRow + kx];
                        }
                    }
                }
            }
        }

        if (bias is not null)
        {
            for (int n = 0; n < N; n++)
            for (int oc = 0; oc < cout; oc++)
            {
                float b0 = bs[oc];
                int yBase = n * yN + oc * yC;
                for (int i = 0; i < yC; i++) ys[yBase + i] += b0;
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}
