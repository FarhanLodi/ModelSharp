using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>GatherND</c>: gathers slices of <c>data</c> addressed by the trailing dimension of
/// <c>indices</c> (length <c>k</c>), supporting <c>batch_dims</c> (default 0). Output shape is
/// <c>indices.shape[:-1] + data.shape[batch_dims + k:]</c>. Dtype-preserving for <c>data</c>.
/// </summary>
public sealed class GatherNDKernel : IKernel
{
    public string OpType => "GatherND";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        Tensor indicesT = ctx.GetTensor(node.Inputs[1]);
        long[] idx = TensorInts.Read(indicesT);
        int[] iDims = indicesT.Shape.Dimensions.ToArray();
        int batchDims = (int)Attr.Int(node, "batch_dims", 0);

        ctx.Set(node.Outputs[0], data.Dtype switch
        {
            ElementType.Int64 => Gather<long>(data.AsInt64(), idx, iDims, batchDims),
            ElementType.Int32 => Gather<int>(data.AsInt32(), idx, iDims, batchDims),
            ElementType.Boolean => Gather<bool>(data.AsBool(), idx, iDims, batchDims),
            _ => Gather<float>(data.AsFloat(), idx, iDims, batchDims),
        });
    }

    private static Tensor<T> Gather<T>(Tensor<T> data, long[] idx, int[] iDims, int b) where T : unmanaged
    {
        ReadOnlySpan<int> dDims = data.Shape.Dimensions;
        int r = dDims.Length;
        int q = iDims.Length;
        int k = iDims[q - 1];
        int[] dStrides = Nd.Strides(dDims);
        int[] iStrides = Nd.Strides(iDims);

        int midLen = q - 1 - b;       // index-iteration dims between batch and the trailing k
        int innerLen = r - (b + k);   // trailing data dims copied verbatim
        int outRank = b + midLen + innerLen;

        var outDims = new int[outRank];
        int pos = 0;
        for (int i = 0; i < b; i++) outDims[pos++] = dDims[i];
        for (int m = 0; m < midLen; m++) outDims[pos++] = iDims[b + m];
        for (int inn = 0; inn < innerLen; inn++) outDims[pos++] = dDims[b + k + inn];

        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> ds = data.Span, ys = y.Span;
        int n = (int)y.Shape.Length;
        var coord = new int[outRank];

        for (int outIdx = 0; outIdx < n; outIdx++)
        {
            int iOff = 0, dOff = 0;
            for (int i = 0; i < b; i++) { int c = coord[i]; iOff += c * iStrides[i]; dOff += c * dStrides[i]; }
            for (int m = 0; m < midLen; m++) iOff += coord[b + m] * iStrides[b + m];
            for (int j = 0; j < k; j++)
            {
                int g = (int)idx[iOff + j * iStrides[q - 1]];
                int dimj = dDims[b + j];
                if (g < 0) g += dimj;
                dOff += g * dStrides[b + j];
            }
            for (int inn = 0; inn < innerLen; inn++) dOff += coord[b + midLen + inn] * dStrides[b + k + inn];
            ys[outIdx] = ds[dOff];

            for (int ax = outRank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }
        return y;
    }
}

/// <summary>
/// ONNX <c>ScatterND</c>: copies <c>data</c>, then writes each <c>updates</c> slice into the location
/// addressed by the matching length-<c>k</c> index tuple in <c>indices</c> (negative-normalized).
/// Supports <c>reduction</c> none/add/mul/min/max for Float32; integer/bool data support
/// <c>reduction=none</c> only. <c>batch_dims=0</c> only. Dtype-preserving for <c>data</c>.
/// </summary>
public sealed class ScatterNDKernel : IKernel
{
    public string OpType => "ScatterND";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        Tensor indicesT = ctx.GetTensor(node.Inputs[1]);
        Tensor updates = ctx.GetTensor(node.Inputs[2]);
        long[] idx = TensorInts.Read(indicesT);
        int[] iDims = indicesT.Shape.Dimensions.ToArray();
        string reduction = Attr.Str(node, "reduction", "none");

        switch (data.Dtype)
        {
            case ElementType.Float32:
                ctx.Set(node.Outputs[0], Scatter(data.AsFloat(), idx, iDims, updates.AsFloat(), ScatterElementsKernel.FloatReducer(reduction)));
                break;
            case ElementType.Int64:
                ctx.Set(node.Outputs[0], Scatter(data.AsInt64(), idx, iDims, updates.AsInt64(), ScatterElementsKernel.AssignOnly<long>(reduction)));
                break;
            case ElementType.Int32:
                ctx.Set(node.Outputs[0], Scatter(data.AsInt32(), idx, iDims, updates.AsInt32(), ScatterElementsKernel.AssignOnly<int>(reduction)));
                break;
            case ElementType.Boolean:
                ctx.Set(node.Outputs[0], Scatter(data.AsBool(), idx, iDims, updates.AsBool(), ScatterElementsKernel.AssignOnly<bool>(reduction)));
                break;
            default:
                throw new ModelSharpException($"ScatterND does not support dtype {data.Dtype}.");
        }
    }

    private static Tensor<T> Scatter<T>(Tensor<T> data, long[] idx, int[] iDims, Tensor<T> updates, Func<T, T, T> combine)
        where T : unmanaged
    {
        ReadOnlySpan<int> dDims = data.Shape.Dimensions;
        int r = dDims.Length;
        int q = iDims.Length;
        int k = iDims[q - 1];
        int[] dStrides = Nd.Strides(dDims);

        int sliceLen = 1;
        for (int i = k; i < r; i++) sliceLen *= dDims[i];
        int numTuples = k == 0 ? 0 : idx.Length / k;

        var y = new Tensor<T>(data.Shape);
        data.Span.CopyTo(y.Span);
        Span<T> ys = y.Span, us = updates.Span;

        for (int t = 0; t < numTuples; t++)
        {
            int baseOff = 0;
            for (int j = 0; j < k; j++)
            {
                int g = (int)idx[t * k + j];
                if (g < 0) g += dDims[j];
                baseOff += g * dStrides[j];
            }
            int uBase = t * sliceLen;
            for (int s = 0; s < sliceLen; s++) ys[baseOff + s] = combine(ys[baseOff + s], us[uBase + s]);
        }
        return y;
    }
}
