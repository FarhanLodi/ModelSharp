using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>Col2Im</c> for 2-D images: rearranges column blocks back into an image, summing
/// overlapping values. <c>input</c> is <c>[N, C*kH*kW, L]</c>, <c>image_shape</c> is the 1-D
/// <c>[H, W]</c> output spatial size, and <c>block_shape</c> is <c>[kH, kW]</c>. Supports the
/// <c>strides</c>, <c>dilations</c> and <c>pads</c> attributes (defaults 1 / 1 / 0). Output is
/// <c>[N, C, H, W]</c>. Float32.
/// </summary>
public sealed class Col2ImKernel : IKernel
{
    public string OpType => "Col2Im";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> input = ctx.Get(node.Inputs[0]);
        long[] imageShape = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        long[] blockShape = TensorInts.Read(ctx.GetTensor(node.Inputs[2]));
        if (imageShape.Length != 2 || blockShape.Length != 2)
            throw new ModelSharpException("Col2Im: only 2-D image_shape / block_shape are supported.");

        int H = (int)imageShape[0], W = (int)imageShape[1];
        int kH = (int)blockShape[0], kW = (int)blockShape[1];

        int[] strides = Attr.Ints(node, "strides", new[] { 1, 1 });
        int[] dil = Attr.Ints(node, "dilations", new[] { 1, 1 });
        int[] pads = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 }); // [top, left, bottom, right]
        int sH = strides[0], sW = strides[1];
        int dH = dil[0], dW = dil[1];
        int padTop = pads[0], padLeft = pads[1], padBottom = pads[2], padRight = pads[3];

        ReadOnlySpan<int> id = input.Shape.Dimensions;
        int N = id[0];
        int chKk = id[1];
        int L = id[2];
        int kk = kH * kW;
        if (chKk % kk != 0) throw new ModelSharpException("Col2Im: input channel dim not divisible by block size.");
        int C = chKk / kk;

        int outH = (H + padTop + padBottom - (dH * (kH - 1) + 1)) / sH + 1;
        int outW = (W + padLeft + padRight - (dW * (kW - 1) + 1)) / sW + 1;
        if (outH * outW != L)
            throw new ModelSharpException($"Col2Im: L={L} does not match computed blocks {outH}x{outW}.");

        var y = new Tensor<float>(new TensorShape(N, C, H, W));
        Span<float> xs = input.Span, ys = y.Span;
        int xN = chKk * L;
        int yN = C * H * W, yC = H * W;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        for (int ky = 0; ky < kH; ky++)
        for (int kx = 0; kx < kW; kx++)
        {
            int row = (c * kk) + (ky * kW + kx);
            int xRowBase = n * xN + row * L;
            for (int by = 0; by < outH; by++)
            for (int bx = 0; bx < outW; bx++)
            {
                int l = by * outW + bx;
                int iy = by * sH - padTop + ky * dH;
                int ix = bx * sW - padLeft + kx * dW;
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                ys[n * yN + c * yC + iy * W + ix] += xs[xRowBase + l];
            }
        }
        ctx.Set(node.Outputs[0], y);
    }
}
