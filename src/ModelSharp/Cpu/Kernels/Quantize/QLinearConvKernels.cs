using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Quantize;

/// <summary>
/// ONNX <c>QLinearConv</c>: quantized convolution. Inputs are
/// <c>(x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B])</c>. <c>x</c>/<c>w</c> are int8/uint8;
/// the optional bias <c>B</c> is int32 in units of <c>x_scale * w_scale</c>. Implemented as
/// dequantize(x) ⊛ dequantize(w) (+ bias·x_scale·w_scale) requantized to <c>y_zp</c>'s dtype, sharing
/// the <see cref="Nn.ConvKernel"/> arithmetic (1-D NCW and 2-D NCHW, strides/pads/dilations/group).
/// The weight is dequantized per-output-channel (axis 0) when <c>w_scale</c> is a vector.
/// </summary>
public sealed class QLinearConvKernel : IKernel
{
    public string OpType => "QLinearConv";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor xQ = ctx.GetTensor(node.Inputs[0]);
        Tensor<float> xScale = ctx.Get(node.Inputs[1]);
        Tensor xZp = ctx.GetTensor(node.Inputs[2]);
        Tensor wQ = ctx.GetTensor(node.Inputs[3]);
        Tensor<float> wScale = ctx.Get(node.Inputs[4]);
        Tensor wZp = ctx.GetTensor(node.Inputs[5]);
        Tensor<float> yScale = ctx.Get(node.Inputs[6]);
        Tensor yZp = ctx.GetTensor(node.Inputs[7]);

        float[] x = QLinearOps.Dequantize(xQ, xScale, xZp, 1);   // per-tensor (typical) input scale
        float[] w = QLinearOps.Dequantize(wQ, wScale, wZp, 0);   // per-output-channel weight scale

        // Bias is int32 quantized in x_scale*w_scale units; pre-scale it to float per output channel.
        float[]? bias = null;
        if (node.Inputs.Count > 8 && node.Inputs[8].Length > 0)
        {
            Tensor bT = ctx.GetTensor(node.Inputs[8]);
            double[] bi = QuantizeOps.ReadIntAsDoubles(bT);
            int cout = wQ.Shape.Dimensions[0];
            Span<float> xs = xScale.Span; Span<float> ws = wScale.Span;
            bool wPerChan = wScale.Length > 1;
            double xs0 = xs.Length > 0 ? xs[0] : 1d;
            bias = new float[bi.Length];
            for (int i = 0; i < bi.Length; i++)
            {
                double wsc = wPerChan ? ws[i % cout] : (ws.Length > 0 ? ws[0] : 1d);
                bias[i] = (float)(bi[i] * xs0 * wsc);
            }
        }

        float[] y = ConvFloat(x, xQ.Shape, w, wQ.Shape, bias, node, out TensorShape yShape);
        ctx.Set(node.Outputs[0], QLinearOps.Quantize(y, yShape, yScale, yZp));
    }

    /// <summary>
    /// Float convolution mirroring <see cref="Nn.ConvKernel"/> (1-D lifted into the 2-D path),
    /// operating on already-dequantized buffers and returning the flat output buffer + shape.
    /// </summary>
    internal static float[] ConvFloat(
        float[] x, TensorShape xShape, float[] w, TensorShape wShape, float[]? bias,
        GraphNode node, out TensorShape yShape)
    {
        ReadOnlySpan<int> xd = xShape.Dimensions;
        ReadOnlySpan<int> wd = wShape.Dimensions;
        int spatial = xd.Length - 2;
        if ((spatial != 1 && spatial != 2) || wd.Length != xd.Length)
            throw new ModelSharpException("QLinearConv supports 1-D (NCW) or 2-D (NCHW) tensors only.");
        bool is1d = spatial == 1;

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
        else if (autoPad == "VALID") { padTop = padLeft = padBottom = padRight = 0; }
        else if (is1d)
        {
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

        yShape = is1d ? new TensorShape(N, cout, outW) : new TensorShape(N, cout, outH, outW);
        var ys = new float[(int)yShape.Length];

        int xN = xd[1] * H * W, xC = H * W;
        int wO = cinPerGroup * kH * kW, wC = kH * kW;
        int yN = cout * outH * outW, yC = outH * outW;

        for (int n = 0; n < N; n++)
        for (int g = 0; g < group; g++)
        for (int ocg = 0; ocg < outPerGroup; ocg++)
        {
            int oc = g * outPerGroup + ocg;
            float b0 = bias is null ? 0f : bias[oc];
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
                            sum += x[xRow + ix] * w[wRow + kx];
                        }
                    }
                }
                ys[n * yN + oc * yC + oy * outW + ox] = sum;
            }
        }
        return ys;
    }
}

/// <summary>
/// ONNX <c>ConvInteger</c>: integer convolution <c>(x - x_zp) ⊛ (w - w_zp)</c> with int32
/// accumulation. Inputs <c>(x, w, [x_zp], [w_zp])</c>; <c>x</c>/<c>w</c> are int8/uint8 and the
/// output is Int32 (no scaling). Shares the convolution arithmetic by treating the zero-point-shifted
/// integers as floats (exact for the integer magnitudes involved) and casting the result back.
/// </summary>
public sealed class ConvIntegerKernel : IKernel
{
    public string OpType => "ConvInteger";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor xQ = ctx.GetTensor(node.Inputs[0]);
        Tensor wQ = ctx.GetTensor(node.Inputs[1]);
        double xZp = node.Inputs.Count > 2 && node.Inputs[2].Length > 0
            ? QuantizeOps.ReadIntAsDoubles(ctx.GetTensor(node.Inputs[2]))[0] : 0d;
        double wZp = node.Inputs.Count > 3 && node.Inputs[3].Length > 0
            ? QuantizeOps.ReadIntAsDoubles(ctx.GetTensor(node.Inputs[3]))[0] : 0d;

        double[] xd = QuantizeOps.ReadIntAsDoubles(xQ);
        double[] wd = QuantizeOps.ReadIntAsDoubles(wQ);
        var x = new float[xd.Length];
        for (int i = 0; i < xd.Length; i++) x[i] = (float)(xd[i] - xZp);
        var w = new float[wd.Length];
        for (int i = 0; i < wd.Length; i++) w[i] = (float)(wd[i] - wZp);

        float[] yf = QLinearConvKernel.ConvFloat(x, xQ.Shape, w, wQ.Shape, null, node, out TensorShape yShape);
        var y = new int[yf.Length];
        for (int i = 0; i < yf.Length; i++) y[i] = (int)MathF.Round(yf[i]);
        ctx.Set(node.Outputs[0], new Tensor<int>(yShape, y));
    }
}

/// <summary>
/// ONNX <c>QLinearGlobalAveragePool</c> (com.microsoft): quantized global average pool over the
/// spatial dims. Inputs <c>(X, x_scale, x_zp, y_scale, y_zp)</c>; dequantizes, averages each
/// (N, C) plane, then requantizes. Output keeps the spatial dims as size 1 (NCHW → NC11).
/// </summary>
public sealed class QLinearGlobalAveragePoolKernel : IKernel
{
    public string OpType => "QLinearGlobalAveragePool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor xQ = ctx.GetTensor(node.Inputs[0]);
        Tensor<float> xScale = ctx.Get(node.Inputs[1]);
        Tensor xZp = ctx.GetTensor(node.Inputs[2]);
        Tensor<float> yScale = ctx.Get(node.Inputs[3]);
        Tensor yZp = ctx.GetTensor(node.Inputs[4]);

        float[] x = QLinearOps.Dequantize(xQ, xScale, xZp, 1);
        ReadOnlySpan<int> d = xQ.Shape.Dimensions;
        if (d.Length < 3)
            throw new ModelSharpException("QLinearGlobalAveragePool expects an N×C×spatial tensor.");
        int N = d[0], C = d[1];
        int spatial = 1; for (int i = 2; i < d.Length; i++) spatial *= d[i];

        var outDims = new int[d.Length];
        outDims[0] = N; outDims[1] = C;
        for (int i = 2; i < d.Length; i++) outDims[i] = 1;

        var y = new float[N * C];
        for (int nc = 0; nc < N * C; nc++)
        {
            float sum = 0f;
            int baseIdx = nc * spatial;
            for (int s = 0; s < spatial; s++) sum += x[baseIdx + s];
            y[nc] = sum / spatial;
        }

        ctx.Set(node.Outputs[0], QLinearOps.Quantize(y, new TensorShape(outDims), yScale, yZp));
    }
}
