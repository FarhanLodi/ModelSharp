using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Sequence;

/// <summary>
/// Shared helpers for the <c>Sequence*</c> kernels: a positive-index resolver (ONNX allows a
/// negative position for insert/erase/at) and a dtype-generic copy-out of a slice along an axis.
/// </summary>
internal static class SeqOps
{
    /// <summary>
    /// Normalizes a (possibly negative) sequence position. For read/erase the valid range is
    /// [-n, n-1]; for insert the valid range is [-n-1, n] (n is an allowed append point).
    /// </summary>
    public static int ResolvePosition(long pos, int n, bool forInsert, GraphNode node)
    {
        int max = forInsert ? n : n - 1;
        int min = forInsert ? -n - 1 : -n;
        long p = pos < 0 ? pos + n + (forInsert ? 1 : 0) : pos;
        if (pos < min || pos > max)
            throw new ModelSharpException(
                $"{node.OpType} node '{node.Name}': position {pos} out of range for sequence length {n}.");
        return (int)p;
    }

    /// <summary>Reads an optional integer scalar input (the position operand), or null when omitted.</summary>
    public static long? OptionalIntScalar(GraphNode node, GraphContext ctx, int inputIndex)
    {
        if (node.Inputs.Count <= inputIndex || node.Inputs[inputIndex].Length == 0)
            return null;
        long[] v = TensorInts.Read(ctx.GetTensor(node.Inputs[inputIndex]));
        return v.Length > 0 ? v[0] : null;
    }
}

/// <summary>
/// ONNX <c>SequenceEmpty</c>: produces an empty tensor sequence. The (ignored here) <c>dtype</c>
/// attribute declares the element type; an empty sequence carries no tensors so it is dtype-free
/// until the first <c>SequenceInsert</c>.
/// </summary>
public sealed class SequenceEmptyKernel : IKernel
{
    public string OpType => "SequenceEmpty";

    public void Execute(GraphNode node, GraphContext ctx) =>
        ctx.SetSeq(node.Outputs[0], SeqValue.Sequence(Array.Empty<Tensor>()));
}

/// <summary>
/// ONNX <c>SequenceConstruct</c>: builds a sequence from its (one or more) tensor inputs, in order.
/// </summary>
public sealed class SequenceConstructKernel : IKernel
{
    public string OpType => "SequenceConstruct";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        var items = new List<Tensor>(node.Inputs.Count);
        foreach (string inName in node.Inputs)
            items.Add(ctx.GetTensor(inName));
        ctx.SetSeq(node.Outputs[0], SeqValue.Sequence(items));
    }
}

/// <summary>
/// ONNX <c>SequenceInsert</c>: returns a new sequence with <c>tensor</c> inserted at <c>position</c>
/// (default = append at the end; negative positions count from the end). The input sequence is not
/// mutated (value semantics).
/// </summary>
public sealed class SequenceInsertKernel : IKernel
{
    public string OpType => "SequenceInsert";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        SeqValue seq = ctx.GetSeq(node.Inputs[0]);
        Tensor tensor = ctx.GetTensor(node.Inputs[1]);
        int n = seq.Count;
        long? posArg = SeqOps.OptionalIntScalar(node, ctx, 2);
        int pos = posArg is null ? n : SeqOps.ResolvePosition(posArg.Value, n, forInsert: true, node);

        var items = new List<Tensor>(n + 1);
        items.AddRange(seq.Tensors);
        items.Insert(pos, tensor);
        ctx.SetSeq(node.Outputs[0], SeqValue.Sequence(items));
    }
}

/// <summary>
/// ONNX <c>SequenceErase</c>: returns a new sequence with the tensor at <c>position</c> removed
/// (default = remove the last element; negative positions count from the end).
/// </summary>
public sealed class SequenceEraseKernel : IKernel
{
    public string OpType => "SequenceErase";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        SeqValue seq = ctx.GetSeq(node.Inputs[0]);
        int n = seq.Count;
        if (n == 0)
            throw new ModelSharpException(
                $"SequenceErase node '{node.Name}': cannot erase from an empty sequence.");
        long? posArg = SeqOps.OptionalIntScalar(node, ctx, 1);
        int pos = posArg is null ? n - 1 : SeqOps.ResolvePosition(posArg.Value, n, forInsert: false, node);

        var items = new List<Tensor>(n - 1);
        items.AddRange(seq.Tensors);
        items.RemoveAt(pos);
        ctx.SetSeq(node.Outputs[0], SeqValue.Sequence(items));
    }
}

/// <summary>
/// ONNX <c>SequenceAt</c>: returns the tensor at <c>position</c> (negative positions count from the
/// end). The result is a plain tensor.
/// </summary>
public sealed class SequenceAtKernel : IKernel
{
    public string OpType => "SequenceAt";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        SeqValue seq = ctx.GetSeq(node.Inputs[0]);
        int n = seq.Count;
        long posArg = SeqOps.OptionalIntScalar(node, ctx, 1)
            ?? throw new ModelSharpException($"SequenceAt node '{node.Name}': missing position input.");
        int pos = SeqOps.ResolvePosition(posArg, n, forInsert: false, node);
        ctx.Set(node.Outputs[0], seq.Tensors[pos]);
    }
}

/// <summary>
/// ONNX <c>SequenceLength</c>: returns the number of tensors in the sequence as an Int64 scalar.
/// </summary>
public sealed class SequenceLengthKernel : IKernel
{
    public string OpType => "SequenceLength";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        SeqValue seq = ctx.GetSeq(node.Inputs[0]);
        ctx.Set(node.Outputs[0], new Tensor<long>(new TensorShape(), new long[] { seq.Count }));
    }
}

/// <summary>
/// ONNX <c>SplitToSequence</c>: splits <c>input</c> along <c>axis</c> into a sequence of tensors.
/// Split sizes come from the optional <c>split</c> input — a scalar (each chunk that size, last
/// shrinks), or a 1-D tensor of explicit per-chunk sizes. With <c>split</c> omitted the axis is
/// cut into length-1 chunks; the <c>keepdims</c> attribute (default 1) then controls whether the
/// split axis is retained (size 1) or squeezed out. Dtype-preserving.
/// </summary>
public sealed class SplitToSequenceKernel : IKernel
{
    public string OpType => "SplitToSequence";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        int rank = data.Shape.Rank;
        long axisAttr = Attr.Int(node, "axis", 0);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        int axisDim = data.Shape.Dimensions[axis];

        bool hasSplit = node.Inputs.Count > 1 && node.Inputs[1].Length > 0;
        bool keepdims = Attr.Int(node, "keepdims", 1) != 0;

        int[] sizes;
        bool squeeze = false;
        if (hasSplit)
        {
            Tensor splitT = ctx.GetTensor(node.Inputs[1]);
            long[] sp = TensorInts.Read(splitT);
            if (splitT.Shape.Rank == 0)
            {
                // Scalar: chunk size; last chunk takes the remainder.
                int chunk = (int)sp[0];
                if (chunk <= 0) throw new ModelSharpException("SplitToSequence: split size must be positive.");
                int count = (axisDim + chunk - 1) / chunk;
                sizes = new int[count];
                int rem = axisDim;
                for (int i = 0; i < count; i++) { int c = Math.Min(chunk, rem); sizes[i] = c; rem -= c; }
            }
            else
            {
                sizes = new int[sp.Length];
                for (int i = 0; i < sp.Length; i++) sizes[i] = (int)sp[i];
            }
        }
        else
        {
            // No split: length-1 chunks. keepdims governs whether the axis is squeezed out.
            sizes = new int[axisDim];
            for (int i = 0; i < axisDim; i++) sizes[i] = 1;
            squeeze = !keepdims;
        }

        IReadOnlyList<Tensor> parts = data.Dtype switch
        {
            ElementType.Int64 => SplitParts<long>(data.AsInt64(), axis, sizes, squeeze),
            ElementType.Int32 => SplitParts<int>(data.AsInt32(), axis, sizes, squeeze),
            ElementType.Boolean => SplitParts<bool>(data.AsBool(), axis, sizes, squeeze),
            _ => SplitParts<float>(data.AsFloat(), axis, sizes, squeeze),
        };
        ctx.SetSeq(node.Outputs[0], SeqValue.Sequence(parts));
    }

    private static IReadOnlyList<Tensor> SplitParts<T>(Tensor<T> x, int axis, int[] sizes, bool squeeze)
        where T : unmanaged
    {
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        int axisDim = dims[axis];
        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dims[i];
        Span<T> xs = x.Span;

        var result = new List<Tensor>(sizes.Length);
        int offset = 0;
        foreach (int sz in sizes)
        {
            int[] outDims;
            if (squeeze && sz == 1)
            {
                outDims = new int[rank - 1];
                for (int i = 0, j = 0; i < rank; i++) if (i != axis) outDims[j++] = dims[i];
            }
            else
            {
                outDims = dims.ToArray();
                outDims[axis] = sz;
            }
            var y = new Tensor<T>(new TensorShape(outDims));
            Span<T> ys = y.Span;
            int block = sz * inner;
            for (int o = 0; o < outer; o++)
            {
                int src = (o * axisDim + offset) * inner;
                xs.Slice(src, block).CopyTo(ys.Slice(o * block, block));
            }
            offset += sz;
            result.Add(y);
        }
        return result;
    }
}

/// <summary>
/// ONNX <c>ConcatFromSequence</c>: concatenates a sequence's tensors along <c>axis</c> (when
/// <c>new_axis</c> = 0, the default) or stacks them along a brand-new axis (when <c>new_axis</c> = 1).
/// The inverse of <c>SplitToSequence</c>. Dtype-preserving.
/// </summary>
public sealed class ConcatFromSequenceKernel : IKernel
{
    public string OpType => "ConcatFromSequence";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        SeqValue seq = ctx.GetSeq(node.Inputs[0]);
        if (seq.Count == 0)
            throw new ModelSharpException(
                $"ConcatFromSequence node '{node.Name}': cannot concat an empty sequence.");
        Tensor[] tensors = new Tensor[seq.Count];
        for (int i = 0; i < tensors.Length; i++) tensors[i] = seq.Tensors[i];

        bool newAxis = Attr.Int(node, "new_axis", 0) != 0;
        int refRank = tensors[0].Shape.Rank;
        long axisAttr = Attr.Int(node, "axis", 0);
        // For new_axis the valid range includes the appended axis position [-(r+1), r].
        int axisBound = newAxis ? refRank + 1 : refRank;
        int axis = (int)(axisAttr < 0 ? axisAttr + axisBound : axisAttr);

        Tensor result = tensors[0].Dtype switch
        {
            ElementType.Int64 => Build<long>(tensors, axis, newAxis),
            ElementType.Int32 => Build<int>(tensors, axis, newAxis),
            ElementType.Boolean => Build<bool>(tensors, axis, newAxis),
            _ => Build<float>(tensors, axis, newAxis),
        };
        ctx.Set(node.Outputs[0], result);
    }

    private static Tensor<T> Build<T>(Tensor[] tensors, int axis, bool newAxis) where T : unmanaged
    {
        ReadOnlySpan<int> dims0 = tensors[0].Shape.Dimensions;
        int rank0 = dims0.Length;

        if (newAxis)
        {
            // Stack: every element shares shape dims0; insert a new axis of size = count.
            int count = tensors.Length;
            int[] outDims = new int[rank0 + 1];
            for (int i = 0, j = 0; i < outDims.Length; i++)
                outDims[i] = i == axis ? count : dims0[j++];

            var y = new Tensor<T>(new TensorShape(outDims));
            Span<T> ys = y.Span;
            int outer = 1; for (int i = 0; i < axis; i++) outer *= outDims[i];
            int inner = 1; for (int i = axis + 1; i < outDims.Length; i++) inner *= outDims[i];
            // For each element t, its data fills slots where the new-axis index == element index.
            for (int e = 0; e < count; e++)
            {
                Span<T> ts = ((Tensor<T>)tensors[e]).Span;
                int block = inner;   // each element contributes a contiguous "inner" run per outer
                for (int o = 0; o < outer; o++)
                {
                    int src = o * block;
                    int dst = (o * count + e) * inner;
                    ts.Slice(src, block).CopyTo(ys.Slice(dst, block));
                }
            }
            return y;
        }
        else
        {
            int outAxis = 0;
            foreach (Tensor t in tensors) outAxis += t.Shape.Dimensions[axis];
            int[] outDims = dims0.ToArray();
            outDims[axis] = outAxis;
            var y = new Tensor<T>(new TensorShape(outDims));
            Span<T> ys = y.Span;

            int outer = 1; for (int i = 0; i < axis; i++) outer *= outDims[i];
            int inner = 1; for (int i = axis + 1; i < rank0; i++) inner *= outDims[i];

            int axisOffset = 0;
            foreach (Tensor tb in tensors)
            {
                var t = (Tensor<T>)tb;
                int ta = t.Shape.Dimensions[axis];
                Span<T> ts = t.Span;
                int block = ta * inner;
                for (int o = 0; o < outer; o++)
                {
                    int src = o * block;
                    int dst = (o * outAxis + axisOffset) * inner;
                    ts.Slice(src, block).CopyTo(ys.Slice(dst, block));
                }
                axisOffset += ta;
            }
            return y;
        }
    }
}
