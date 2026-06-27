using System;
using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.ControlFlow;

/// <summary>
/// Dtype-generic tensor utilities shared by the <c>Loop</c> and <c>Scan</c> kernels:
/// stacking per-iteration tensors into a scan output, and slicing a scan input along an axis.
/// </summary>
internal static class CfTensorOps
{
    /// <summary>
    /// Stacks a list of identically-shaped tensors into a single tensor with a new axis of
    /// length <c>slices.Count</c> inserted at <paramref name="axis"/>. Mirrors numpy
    /// <c>stack(slices, axis)</c>. Empty input yields a tensor whose stacked axis is 0.
    /// </summary>
    public static Tensor Stack(IReadOnlyList<Tensor> slices, int axis)
    {
        if (slices.Count == 0)
            return new Tensor<float>(new TensorShape(0));

        Tensor first = slices[0];
        ReadOnlySpan<int> sliceDims = first.Shape.Dimensions;
        int sliceRank = sliceDims.Length;
        if (axis < 0) axis += sliceRank + 1;

        int[] outDims = new int[sliceRank + 1];
        for (int i = 0, o = 0; o < outDims.Length; o++)
            outDims[o] = o == axis ? slices.Count : sliceDims[i++];

        var outShape = new TensorShape(outDims);

        return first.Dtype switch
        {
            ElementType.Int64 => StackTyped<long>(slices, axis, outShape, sliceDims),
            ElementType.Int32 => StackTyped<int>(slices, axis, outShape, sliceDims),
            ElementType.Boolean => StackTyped<bool>(slices, axis, outShape, sliceDims),
            ElementType.Float64 => StackTyped<double>(slices, axis, outShape, sliceDims),
            _ => StackTyped<float>(slices, axis, outShape, sliceDims),
        };
    }

    private static Tensor<T> StackTyped<T>(
        IReadOnlyList<Tensor> slices, int axis, TensorShape outShape, ReadOnlySpan<int> sliceDims)
        where T : unmanaged
    {
        // outer = product of dims before `axis` in the *slice* (== dims before axis in output);
        // inner = product of dims from `axis` onward in the slice.
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= sliceDims[i];
        int inner = 1;
        for (int i = axis; i < sliceDims.Length; i++) inner *= sliceDims[i];

        int n = slices.Count;
        var outBuf = new T[checked((int)outShape.Length)];
        Span<T> dst = outBuf;

        // Output layout: [outer][n][inner]. For each slice s, copy its [outer][inner]
        // contiguous block into the matching n-stripe.
        for (int s = 0; s < n; s++)
        {
            Span<T> src = ((Tensor<T>)slices[s]).Span;
            for (int o = 0; o < outer; o++)
            {
                int srcOff = o * inner;
                int dstOff = (o * n + s) * inner;
                src.Slice(srcOff, inner).CopyTo(dst.Slice(dstOff, inner));
            }
        }
        return new Tensor<T>(outShape, outBuf);
    }

    /// <summary>
    /// Returns slice <paramref name="index"/> taken along <paramref name="axis"/>, with that
    /// axis removed (rank drops by one). Mirrors numpy <c>take(index, axis)</c> for a single index.
    /// </summary>
    public static Tensor SliceAlongAxis(Tensor t, int axis, int index)
    {
        ReadOnlySpan<int> dims = t.Shape.Dimensions;
        int rank = dims.Length;
        if (axis < 0) axis += rank;

        int[] outDims = new int[rank - 1];
        for (int i = 0, o = 0; i < rank; i++)
            if (i != axis) outDims[o++] = dims[i];
        var outShape = new TensorShape(outDims);

        return t.Dtype switch
        {
            ElementType.Int64 => SliceTyped<long>(t, axis, index, outShape),
            ElementType.Int32 => SliceTyped<int>(t, axis, index, outShape),
            ElementType.Boolean => SliceTyped<bool>(t, axis, index, outShape),
            ElementType.Float64 => SliceTyped<double>(t, axis, index, outShape),
            _ => SliceTyped<float>(t, axis, index, outShape),
        };
    }

    private static Tensor<T> SliceTyped<T>(Tensor t, int axis, int index, TensorShape outShape)
        where T : unmanaged
    {
        ReadOnlySpan<int> dims = t.Shape.Dimensions;
        int axisLen = dims[axis];
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1;
        for (int i = axis + 1; i < dims.Length; i++) inner *= dims[i];

        Span<T> src = ((Tensor<T>)t).Span;
        var outBuf = new T[checked((int)outShape.Length)];
        Span<T> dst = outBuf;

        for (int o = 0; o < outer; o++)
        {
            int srcOff = (o * axisLen + index) * inner;
            int dstOff = o * inner;
            src.Slice(srcOff, inner).CopyTo(dst.Slice(dstOff, inner));
        }
        return new Tensor<T>(outShape, outBuf);
    }

    /// <summary>Reads a scalar long from an INT64 (or INT32) tensor; used for Loop's trip count.</summary>
    public static long ReadScalarInt(Tensor t)
    {
        return t.Dtype switch
        {
            ElementType.Int64 => t.AsInt64().Span[0],
            ElementType.Int32 => t.AsInt32().Span[0],
            _ => throw new ModelSharpException(
                $"Expected an integer scalar, got dtype {t.Dtype}."),
        };
    }

    /// <summary>Builds a 0-D INT64 scalar tensor (used for the Loop iteration counter).</summary>
    public static Tensor<long> Int64Scalar(long value)
        => new Tensor<long>(new TensorShape(), new[] { value });

    /// <summary>Builds a 0-D Boolean scalar tensor.</summary>
    public static Tensor<bool> BoolScalar(bool value)
        => new Tensor<bool>(new TensorShape(), new[] { value });
}
