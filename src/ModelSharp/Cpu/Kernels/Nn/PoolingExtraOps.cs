using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// Global max pool: input [N, C, ...spatial] → [N, C, 1, 1, ...] taking the maximum over all
/// spatial dimensions per (n, c). Mirrors <see cref="GlobalAveragePoolKernel"/> but with max.
/// </summary>
public sealed class GlobalMaxPoolKernel : IKernel
{
    public string OpType => "GlobalMaxPool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        if (xd.Length < 3) throw new ModelSharpException("GlobalMaxPool expects at least a 3-D [N, C, ...] input.");
        int N = xd[0], C = xd[1];
        int spatial = 1;
        for (int i = 2; i < xd.Length; i++) spatial *= xd[i];

        // Output mirrors the input rank with every spatial axis collapsed to 1.
        int[] outDims = new int[xd.Length];
        outDims[0] = N;
        outDims[1] = C;
        for (int i = 2; i < xd.Length; i++) outDims[i] = 1;

        var y = new Tensor<float>(new TensorShape(outDims));
        System.Span<float> xs = x.Span, ys = y.Span;
        int xChannel = spatial, xBatch = C * spatial;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int b = n * xBatch + c * xChannel;
            float m = float.NegativeInfinity;
            for (int i = 0; i < spatial; i++) { float v = xs[b + i]; if (v > m) m = v; }
            ys[n * C + c] = m;
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// 2-D Lp pooling (NCHW): sliding-window output = (Σ |x|^p over the window)^(1/p).
/// Supports kernel_shape, strides, pads and p (default 2). Mirrors <see cref="AveragePoolKernel"/>
/// windowing; out-of-bound (padded) cells contribute 0.
/// </summary>
public sealed class LpPoolKernel : IKernel
{
    public string OpType => "LpPool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        if (xd.Length != 4) throw new ModelSharpException("LpPool currently supports 4-D NCHW only.");
        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];

        int[] k = Attr.Ints(node, "kernel_shape", new[] { 1, 1 });
        int kH = k[0], kW = k[1];
        int[] strides = Attr.Ints(node, "strides", new[] { kH, kW });
        int sH = strides[0], sW = strides[1];
        float p = Attr.Float(node, "p", 2f);

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
            int[] pad = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 });
            padTop = pad[0]; padLeft = pad[1]; padBottom = pad[2]; padRight = pad[3];
        }

        int outH = (H + padTop + padBottom - kH) / sH + 1;
        int outW = (W + padLeft + padRight - kW) / sW + 1;

        var y = new Tensor<float>(new TensorShape(N, C, outH, outW));
        System.Span<float> xs = x.Span, ys = y.Span;
        int xC = H * W, xN = C * xC, yC = outH * outW, yN = C * yC;
        float invP = 1f / p;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        for (int oy = 0; oy < outH; oy++)
        for (int ox = 0; ox < outW; ox++)
        {
            float sum = 0f;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * sH - padTop + ky;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * sW - padLeft + kx;
                    if (ix < 0 || ix >= W) continue;
                    float v = MathF.Abs(xs[n * xN + c * xC + iy * W + ix]);
                    sum += MathF.Pow(v, p);
                }
            }
            ys[n * yN + c * yC + oy * outW + ox] = MathF.Pow(sum, invP);
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Global Lp pooling: input [N, C, ...spatial] → [N, C, 1, 1, ...]. Computes the Lp norm
/// (Σ |x|^p)^(1/p) over all spatial dimensions per (n, c). Attr p (default 2).
/// </summary>
public sealed class GlobalLpPoolKernel : IKernel
{
    public string OpType => "GlobalLpPool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        if (xd.Length < 3) throw new ModelSharpException("GlobalLpPool expects at least a 3-D [N, C, ...] input.");
        int N = xd[0], C = xd[1];
        int spatial = 1;
        for (int i = 2; i < xd.Length; i++) spatial *= xd[i];
        float p = Attr.Float(node, "p", 2f);
        float invP = 1f / p;

        int[] outDims = new int[xd.Length];
        outDims[0] = N;
        outDims[1] = C;
        for (int i = 2; i < xd.Length; i++) outDims[i] = 1;

        var y = new Tensor<float>(new TensorShape(outDims));
        System.Span<float> xs = x.Span, ys = y.Span;
        int xChannel = spatial, xBatch = C * spatial;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int b = n * xBatch + c * xChannel;
            float sum = 0f;
            for (int i = 0; i < spatial; i++) sum += MathF.Pow(MathF.Abs(xs[b + i]), p);
            ys[n * C + c] = MathF.Pow(sum, invP);
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Hardmax along an axis (default -1): emits a one-hot tensor with 1.0 at the argmax position
/// of each slice and 0 elsewhere. On ties the first (lowest-index) maximum wins.
/// </summary>
public sealed class HardmaxKernel : IKernel
{
    public string OpType => "Hardmax";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        long axisAttr = Attr.Int(node, "axis", -1);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        if (axis < 0 || axis >= rank) throw new ModelSharpException($"Hardmax axis {axisAttr} is out of range for rank {rank}.");

        int axisSize = dims[axis];
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        var y = new Tensor<float>(x.Shape); // zero-filled
        System.Span<float> xs = x.Span, ys = y.Span;
        for (int o = 0; o < outer; o++)
        for (int q = 0; q < inner; q++)
        {
            int baseIdx = o * axisSize * inner + q;
            float mx = float.NegativeInfinity;
            int argMax = 0;
            for (int s = 0; s < axisSize; s++)
            {
                float v = xs[baseIdx + s * inner];
                if (v > mx) { mx = v; argMax = s; }
            }
            ys[baseIdx + argMax * inner] = 1f;
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// MaxUnpool: scatters the values of X back into a zero tensor at the flat input positions given
/// by the int64 index tensor I (as produced by MaxPool). The output shape comes from the optional
/// third input (output_shape) when present, otherwise it is computed from kernel_shape, strides
/// and pads. Indices are flat offsets within each (n, c) channel plane.
/// </summary>
public sealed class MaxUnpoolKernel : IKernel
{
    public string OpType => "MaxUnpool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<long> ind = ctx.GetTensor(node.Inputs[1]).AsInt64();
        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        if (xd.Length != 4) throw new ModelSharpException("MaxUnpool currently supports 4-D NCHW only.");
        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];

        int outH, outW;
        bool hasShapeInput = node.Inputs.Count > 2 && !string.IsNullOrEmpty(node.Inputs[2]);
        if (hasShapeInput)
        {
            Tensor<long> shape = ctx.GetTensor(node.Inputs[2]).AsInt64();
            System.Span<long> ss = shape.Span;
            if (ss.Length != 4) throw new ModelSharpException("MaxUnpool output_shape must have 4 entries (NCHW).");
            outH = (int)ss[2];
            outW = (int)ss[3];
        }
        else
        {
            int[] k = Attr.Ints(node, "kernel_shape", new[] { 1, 1 });
            int kH = k[0], kW = k[1];
            int[] strides = Attr.Ints(node, "strides", new[] { kH, kW });
            int sH = strides[0], sW = strides[1];
            int[] pad = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 });
            int padTop = pad[0], padLeft = pad[1], padBottom = pad[2], padRight = pad[3];
            // Inverse of the floor-mode MaxPool output formula.
            outH = (H - 1) * sH - padTop - padBottom + kH;
            outW = (W - 1) * sW - padLeft - padRight + kW;
        }

        var y = new Tensor<float>(new TensorShape(N, C, outH, outW)); // zero-filled
        System.Span<float> xs = x.Span, ys = y.Span;
        System.Span<long> idx = ind.Span;
        int xC = H * W, xN = C * xC, yC = outH * outW, yN = C * yC;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int xBase = n * xN + c * xC;
            int yBase = n * yN + c * yC;
            for (int i = 0; i < xC; i++)
            {
                long flat = idx[xBase + i];
                if (flat < 0 || flat >= yC)
                    throw new ModelSharpException($"MaxUnpool index {flat} out of range for channel plane size {yC}.");
                ys[yBase + (int)flat] = xs[xBase + i];
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}
