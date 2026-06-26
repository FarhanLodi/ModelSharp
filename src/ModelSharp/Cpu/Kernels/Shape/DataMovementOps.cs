using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>OneHot</c>: inputs <c>indices</c>, <c>depth</c> (scalar), and <c>values</c> = [off, on].
/// The output has rank <c>indices.rank + 1</c>; a new axis of size <c>depth</c> is inserted at
/// <c>axis</c> (default -1, i.e. appended last). Each output element is <c>on</c> where the new
/// axis coordinate equals the (negative-normalized) index and <c>off</c> otherwise. The output
/// dtype follows <c>values</c> (dtype-generic for Float32 / Int64 / Int32 / Boolean).
/// </summary>
public sealed class OneHotKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "OneHot";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor indicesT = ctx.GetTensor(node.Inputs[0]);
        Tensor depthT = ctx.GetTensor(node.Inputs[1]);
        Tensor valuesT = ctx.GetTensor(node.Inputs[2]);

        long[] indices = TensorInts.Read(indicesT);
        long depth = TensorInts.Read(depthT)[0];
        if (depth <= 0)
            throw new ModelSharpException($"OneHot depth must be positive but was {depth}.");
        int d = (int)depth;

        int[] inDims = indicesT.Shape.Dimensions.ToArray();
        int inRank = inDims.Length;
        int outRank = inRank + 1;
        long axisAttr = Attr.Int(node, "axis", -1);
        int axis = (int)(axisAttr < 0 ? axisAttr + outRank : axisAttr);
        if (axis < 0 || axis >= outRank)
            throw new ModelSharpException($"OneHot axis {axisAttr} is out of range for output rank {outRank}.");

        ctx.Set(node.Outputs[0], valuesT.Dtype switch
        {
            ElementType.Int64 => Build<long>(valuesT.AsInt64(), indices, inDims, d, axis),
            ElementType.Int32 => Build<int>(valuesT.AsInt32(), indices, inDims, d, axis),
            ElementType.Boolean => Build<bool>(valuesT.AsBool(), indices, inDims, d, axis),
            _ => Build<float>(valuesT.AsFloat(), indices, inDims, d, axis),
        });
    }

    private static Tensor<T> Build<T>(Tensor<T> values, long[] indices, int[] inDims, int depth, int axis)
        where T : unmanaged
    {
        T off = values.Span[0];
        T on = values.Span[1];

        int inRank = inDims.Length;
        int outRank = inRank + 1;
        var outDims = new int[outRank];
        for (int i = 0, j = 0; i < outRank; i++)
            outDims[i] = i == axis ? depth : inDims[j++];

        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> ys = y.Span;
        ys.Fill(off);

        int[] outStrides = Nd.Strides(outDims);
        int[] inStrides = Nd.Strides(inDims);
        int n = indices.Length;
        for (int p = 0; p < n; p++)
        {
            int gi = (int)indices[p];
            if (gi < 0) gi += depth;
            if (gi < 0 || gi >= depth) continue; // out-of-range stays all-off per ONNX

            // Map the input flat position p (over inDims) into the output flat position,
            // placing the matched index along the new axis.
            int dst = gi * outStrides[axis];
            for (int i = 0; i < inRank; i++)
            {
                int c = (p / inStrides[i]) % inDims[i];
                int outAxis = i < axis ? i : i + 1;
                dst += c * outStrides[outAxis];
            }
            ys[dst] = on;
        }
        return y;
    }
}

/// <summary>
/// ONNX <c>EyeLike</c>: produces a 2-D identity-like tensor with the same shape as the input
/// (which must be rank 2). Ones are placed on the <c>k</c>-th diagonal (attribute <c>k</c>,
/// default 0; positive shifts above the main diagonal, negative below). The output dtype is the
/// optional <c>dtype</c> attribute (an ONNX TensorProto data_type int) or, if absent, the input's.
/// Dtype-generic for Float32 / Int64 / Int32 / Boolean.
/// </summary>
public sealed class EyeLikeKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "EyeLike";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        ReadOnlySpan<int> dims = input.Shape.Dimensions;
        if (dims.Length != 2)
            throw new ModelSharpException($"EyeLike requires a 2-D input but rank was {dims.Length}.");
        int rows = dims[0], cols = dims[1];
        int k = (int)Attr.Int(node, "k", 0);

        ElementType outType = node.Attributes.TryGetValue("dtype", out object? dt)
            ? MapOnnxType(Convert.ToInt64(dt))
            : input.Dtype;

        var shape = new TensorShape(rows, cols);
        ctx.Set(node.Outputs[0], outType switch
        {
            ElementType.Int64 => Build<long>(shape, rows, cols, k, 1L),
            ElementType.Int32 => Build<int>(shape, rows, cols, k, 1),
            ElementType.Boolean => Build<bool>(shape, rows, cols, k, true),
            ElementType.Float64 => Build<double>(shape, rows, cols, k, 1d),
            _ => Build<float>(shape, rows, cols, k, 1f),
        });
    }

    private static Tensor<T> Build<T>(TensorShape shape, int rows, int cols, int k, T one)
        where T : unmanaged
    {
        var y = new Tensor<T>(shape);
        Span<T> ys = y.Span;
        for (int r = 0; r < rows; r++)
        {
            int c = r + k;
            if (c >= 0 && c < cols) ys[r * cols + c] = one;
        }
        return y;
    }

    private static ElementType MapOnnxType(long to) => to switch
    {
        1 => ElementType.Float32,
        6 => ElementType.Int32,
        7 => ElementType.Int64,
        9 => ElementType.Boolean,
        11 => ElementType.Float64,
        _ => throw new ModelSharpException($"EyeLike 'dtype' data_type {to} is not supported."),
    };
}

/// <summary>
/// ONNX <c>Compress</c>: selects sub-tensors of <c>input</c> along <c>axis</c> at the positions where
/// the 1-D boolean <c>condition</c> is true. If <c>axis</c> is omitted, the input is flattened first
/// and individual elements are selected. <c>condition</c> may be shorter than the axis length (extra
/// entries are treated as false). Dtype-preserving for the input.
/// </summary>
public sealed class CompressKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Compress";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        long[] cond = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));

        bool hasAxis = node.Attributes.ContainsKey("axis");
        int rank = input.Shape.Rank;
        int axis = hasAxis ? (int)Attr.Int(node, "axis", 0) : 0;
        if (hasAxis && axis < 0) axis += rank;

        ctx.Set(node.Outputs[0], input.Dtype switch
        {
            ElementType.Int64 => Run<long>(input.AsInt64(), cond, hasAxis, axis),
            ElementType.Int32 => Run<int>(input.AsInt32(), cond, hasAxis, axis),
            ElementType.Boolean => Run<bool>(input.AsBool(), cond, hasAxis, axis),
            ElementType.Float64 => Run<double>((Tensor<double>)input, cond, hasAxis, axis),
            _ => Run<float>(input.AsFloat(), cond, hasAxis, axis),
        });
    }

    private static Tensor<T> Run<T>(Tensor<T> input, long[] cond, bool hasAxis, int axis) where T : unmanaged
    {
        Span<T> xs = input.Span;

        // Selected condition positions (only those that index within the axis length).
        var sel = new List<int>();

        if (!hasAxis)
        {
            int n = xs.Length;
            int limit = Math.Min(cond.Length, n);
            for (int i = 0; i < limit; i++) if (cond[i] != 0) sel.Add(i);

            var flat = new Tensor<T>(new TensorShape(sel.Count));
            Span<T> fs = flat.Span;
            for (int i = 0; i < sel.Count; i++) fs[i] = xs[sel[i]];
            return flat;
        }

        ReadOnlySpan<int> dims = input.Shape.Dimensions;
        int axisDim = dims[axis];
        int condLimit = Math.Min(cond.Length, axisDim);
        for (int i = 0; i < condLimit; i++) if (cond[i] != 0) sel.Add(i);

        int rank = dims.Length;
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        int[] outDims = dims.ToArray();
        outDims[axis] = sel.Count;
        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> ys = y.Span;

        int newAxis = sel.Count;
        for (int o = 0; o < outer; o++)
        {
            for (int s = 0; s < sel.Count; s++)
            {
                int srcRow = (o * axisDim + sel[s]) * inner;
                int dstRow = (o * newAxis + s) * inner;
                xs.Slice(srcRow, inner).CopyTo(ys.Slice(dstRow, inner));
            }
        }
        return y;
    }
}

/// <summary>
/// ONNX <c>DepthToSpace</c>: rearranges data from the depth dimension into spatial blocks for an
/// <c>[N, C, H, W]</c> input, producing <c>[N, C/(b*b), H*b, W*b]</c> where <c>b</c> is the
/// <c>blocksize</c> attribute. The <c>mode</c> attribute is "DCR" (depth-column-row, default) or
/// "CRD" (column-row-depth), which differ in how the depth axis factors into block / channel.
/// Dtype-preserving.
/// </summary>
public sealed class DepthToSpaceKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "DepthToSpace";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        int b = (int)Attr.Int(node, "blocksize", 0);
        if (b <= 0) throw new ModelSharpException("DepthToSpace requires a positive 'blocksize'.");
        string mode = Attr.Str(node, "mode", "DCR");

        ctx.Set(node.Outputs[0], input.Dtype switch
        {
            ElementType.Int64 => Run<long>(input.AsInt64(), b, mode),
            ElementType.Int32 => Run<int>(input.AsInt32(), b, mode),
            ElementType.Boolean => Run<bool>(input.AsBool(), b, mode),
            ElementType.Float64 => Run<double>((Tensor<double>)input, b, mode),
            _ => Run<float>(input.AsFloat(), b, mode),
        });
    }

    private static Tensor<T> Run<T>(Tensor<T> input, int b, string mode) where T : unmanaged
    {
        ReadOnlySpan<int> dims = input.Shape.Dimensions;
        if (dims.Length != 4)
            throw new ModelSharpException($"DepthToSpace requires a 4-D [N,C,H,W] input but rank was {dims.Length}.");
        int n = dims[0], c = dims[1], h = dims[2], w = dims[3];
        if (c % (b * b) != 0)
            throw new ModelSharpException($"DepthToSpace: channel {c} is not divisible by blocksize^2 ({b * b}).");
        bool dcr = mode == "DCR";
        if (!dcr && mode != "CRD")
            throw new ModelSharpException($"DepthToSpace: unsupported mode '{mode}'.");

        int oc = c / (b * b);
        int oh = h * b, ow = w * b;
        var y = new Tensor<T>(new TensorShape(n, oc, oh, ow));
        Span<T> xs = input.Span, ys = y.Span;

        // Output strides.
        int oCs = oh * ow, oHs = ow;
        // Input row stride.
        int iHs = w;

        for (int bn = 0; bn < n; bn++)
        for (int ch = 0; ch < oc; ch++)
        for (int by = 0; by < b; by++)
        for (int bx = 0; bx < b; bx++)
        {
            // Map output channel + block offset back to the input channel.
            // DCR: c_in = (by*b + bx)*oc + ch ;  CRD: c_in = ch*(b*b) + by*b + bx.
            int cin = dcr ? (by * b + bx) * oc + ch : ch * (b * b) + by * b + bx;
            int inBase = ((bn * c + cin) * h) * w;
            int outChBase = (bn * oc + ch) * oCs;
            for (int y0 = 0; y0 < h; y0++)
            {
                int srcRow = inBase + y0 * iHs;
                int dstRow = outChBase + (y0 * b + by) * oHs + bx;
                for (int x0 = 0; x0 < w; x0++)
                    ys[dstRow + x0 * b] = xs[srcRow + x0];
            }
        }
        return y;
    }
}

/// <summary>
/// ONNX <c>SpaceToDepth</c>: the inverse of <see cref="DepthToSpaceKernel"/> (DCR layout). For an
/// <c>[N, C, H, W]</c> input it produces <c>[N, C*b*b, H/b, W/b]</c> where <c>b</c> is the
/// <c>blocksize</c> attribute, moving <c>b×b</c> spatial blocks into the depth dimension.
/// Dtype-preserving.
/// </summary>
public sealed class SpaceToDepthKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "SpaceToDepth";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        int b = (int)Attr.Int(node, "blocksize", 0);
        if (b <= 0) throw new ModelSharpException("SpaceToDepth requires a positive 'blocksize'.");

        ctx.Set(node.Outputs[0], input.Dtype switch
        {
            ElementType.Int64 => Run<long>(input.AsInt64(), b),
            ElementType.Int32 => Run<int>(input.AsInt32(), b),
            ElementType.Boolean => Run<bool>(input.AsBool(), b),
            ElementType.Float64 => Run<double>((Tensor<double>)input, b),
            _ => Run<float>(input.AsFloat(), b),
        });
    }

    private static Tensor<T> Run<T>(Tensor<T> input, int b) where T : unmanaged
    {
        ReadOnlySpan<int> dims = input.Shape.Dimensions;
        if (dims.Length != 4)
            throw new ModelSharpException($"SpaceToDepth requires a 4-D [N,C,H,W] input but rank was {dims.Length}.");
        int n = dims[0], c = dims[1], h = dims[2], w = dims[3];
        if (h % b != 0 || w % b != 0)
            throw new ModelSharpException($"SpaceToDepth: H ({h}) and W ({w}) must be divisible by blocksize ({b}).");

        int oc = c * b * b;
        int oh = h / b, ow = w / b;
        var y = new Tensor<T>(new TensorShape(n, oc, oh, ow));
        Span<T> xs = input.Span, ys = y.Span;

        int oCs = oh * ow, oHs = ow;
        int iCs = h * w, iHs = w;

        for (int bn = 0; bn < n; bn++)
        for (int ch = 0; ch < c; ch++)
        for (int by = 0; by < b; by++)
        for (int bx = 0; bx < b; bx++)
        {
            // DCR output channel for this block offset (inverse of DepthToSpace).
            int cout = (by * b + bx) * c + ch;
            int inChBase = (bn * c + ch) * iCs;
            int outChBase = (bn * oc + cout) * oCs;
            for (int y0 = 0; y0 < oh; y0++)
            {
                int srcRow = inChBase + (y0 * b + by) * iHs + bx;
                int dstRow = outChBase + y0 * oHs;
                for (int x0 = 0; x0 < ow; x0++)
                    ys[dstRow + x0] = xs[srcRow + x0 * b];
            }
        }
        return y;
    }
}

/// <summary>
/// ONNX <c>ReverseSequence</c>: reverses the first <c>sequence_lens[b]</c> entries along the
/// <c>time_axis</c> (default 0) for each batch index along the <c>batch_axis</c> (default 1); the
/// remaining entries are copied unchanged. The two axes must be distinct. Dtype-preserving.
/// </summary>
public sealed class ReverseSequenceKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "ReverseSequence";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        long[] seqLens = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        int rank = input.Shape.Rank;
        int batchAxis = (int)Attr.Int(node, "batch_axis", 1);
        int timeAxis = (int)Attr.Int(node, "time_axis", 0);
        if (batchAxis < 0) batchAxis += rank;
        if (timeAxis < 0) timeAxis += rank;
        if (batchAxis == timeAxis)
            throw new ModelSharpException("ReverseSequence: batch_axis and time_axis must differ.");

        ctx.Set(node.Outputs[0], input.Dtype switch
        {
            ElementType.Int64 => Run<long>(input.AsInt64(), seqLens, batchAxis, timeAxis),
            ElementType.Int32 => Run<int>(input.AsInt32(), seqLens, batchAxis, timeAxis),
            ElementType.Boolean => Run<bool>(input.AsBool(), seqLens, batchAxis, timeAxis),
            ElementType.Float64 => Run<double>((Tensor<double>)input, seqLens, batchAxis, timeAxis),
            _ => Run<float>(input.AsFloat(), seqLens, batchAxis, timeAxis),
        });
    }

    private static Tensor<T> Run<T>(Tensor<T> input, long[] seqLens, int batchAxis, int timeAxis)
        where T : unmanaged
    {
        ReadOnlySpan<int> dims = input.Shape.Dimensions;
        int rank = dims.Length;
        int batchDim = dims[batchAxis];
        int timeDim = dims[timeAxis];
        int[] strides = Nd.Strides(dims);

        var y = new Tensor<T>(input.Shape);
        Span<T> xs = input.Span, ys = y.Span;
        int total = xs.Length;

        for (int p = 0; p < total; p++)
        {
            int batch = (p / strides[batchAxis]) % batchDim;
            int time = (p / strides[timeAxis]) % timeDim;
            int len = (int)seqLens[batch];
            int srcTime = (time < len) ? (len - 1 - time) : time;
            // Source flat index is p with the time coordinate replaced by srcTime.
            int src = p + (srcTime - time) * strides[timeAxis];
            ys[p] = xs[src];
        }
        return y;
    }
}
