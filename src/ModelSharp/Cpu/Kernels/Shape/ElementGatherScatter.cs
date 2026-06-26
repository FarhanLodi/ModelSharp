using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>GatherElements</c>: <c>indices</c> has the same rank as <c>data</c>; the output (shaped like
/// <c>indices</c>) takes, at each position, the <c>data</c> element whose <c>axis</c> coordinate is the
/// index value (negative-normalized) and whose other coordinates match the output position.
/// Dtype-preserving for <c>data</c>.
/// </summary>
public sealed class GatherElementsKernel : IKernel
{
    public string OpType => "GatherElements";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        Tensor indicesT = ctx.GetTensor(node.Inputs[1]);
        long[] idx = TensorInts.Read(indicesT);
        int[] idxDims = indicesT.Shape.Dimensions.ToArray();
        int axis = (int)Attr.Int(node, "axis", 0);

        ctx.Set(node.Outputs[0], data.Dtype switch
        {
            ElementType.Int64 => Gather<long>(data.AsInt64(), idx, idxDims, axis),
            ElementType.Int32 => Gather<int>(data.AsInt32(), idx, idxDims, axis),
            ElementType.Boolean => Gather<bool>(data.AsBool(), idx, idxDims, axis),
            _ => Gather<float>(data.AsFloat(), idx, idxDims, axis),
        });
    }

    private static Tensor<T> Gather<T>(Tensor<T> data, long[] idx, int[] idxDims, int axis) where T : unmanaged
    {
        ReadOnlySpan<int> dDims = data.Shape.Dimensions;
        int rank = dDims.Length;
        if (axis < 0) axis += rank;
        int axisDim = dDims[axis];
        int[] dStrides = Nd.Strides(dDims);
        int[] idxStrides = Nd.Strides(idxDims);

        var y = new Tensor<T>(new TensorShape(idxDims));
        Span<T> ds = data.Span, ys = y.Span;
        int n = idx.Length;
        for (int p = 0; p < n; p++)
        {
            int src = 0;
            for (int i = 0; i < rank; i++)
            {
                int c = (p / idxStrides[i]) % idxDims[i];
                if (i == axis)
                {
                    int gi = (int)idx[p];
                    if (gi < 0) gi += axisDim;
                    src += gi * dStrides[i];
                }
                else src += c * dStrides[i];
            }
            ys[p] = ds[src];
        }
        return y;
    }
}

/// <summary>
/// ONNX <c>ScatterElements</c>: copies <c>data</c>, then writes each <c>updates</c> element into the
/// position whose <c>axis</c> coordinate is the matching <c>indices</c> value (negative-normalized).
/// Supports <c>reduction</c> none/add/mul/min/max for Float32; integer/bool data support
/// <c>reduction=none</c> only. Dtype-preserving for <c>data</c>.
/// </summary>
public sealed class ScatterElementsKernel : IKernel
{
    public string OpType => "ScatterElements";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        Tensor indicesT = ctx.GetTensor(node.Inputs[1]);
        Tensor updates = ctx.GetTensor(node.Inputs[2]);
        long[] idx = TensorInts.Read(indicesT);
        int[] idxDims = indicesT.Shape.Dimensions.ToArray();
        int axis = (int)Attr.Int(node, "axis", 0);
        string reduction = Attr.Str(node, "reduction", "none");

        switch (data.Dtype)
        {
            case ElementType.Float32:
                ctx.Set(node.Outputs[0], Scatter(data.AsFloat(), idx, idxDims, updates.AsFloat(), axis, FloatReducer(reduction)));
                break;
            case ElementType.Int64:
                ctx.Set(node.Outputs[0], Scatter(data.AsInt64(), idx, idxDims, updates.AsInt64(), axis, AssignOnly<long>(reduction)));
                break;
            case ElementType.Int32:
                ctx.Set(node.Outputs[0], Scatter(data.AsInt32(), idx, idxDims, updates.AsInt32(), axis, AssignOnly<int>(reduction)));
                break;
            case ElementType.Boolean:
                ctx.Set(node.Outputs[0], Scatter(data.AsBool(), idx, idxDims, updates.AsBool(), axis, AssignOnly<bool>(reduction)));
                break;
            default:
                throw new ModelSharpException($"ScatterElements does not support dtype {data.Dtype}.");
        }
    }

    internal static Func<float, float, float> FloatReducer(string reduction) => reduction switch
    {
        "none" => (_, u) => u,
        "add" => (a, u) => a + u,
        "mul" => (a, u) => a * u,
        "min" => (a, u) => MathF.Min(a, u),
        "max" => (a, u) => MathF.Max(a, u),
        _ => throw new ModelSharpException($"ScatterElements: unsupported reduction '{reduction}'."),
    };

    internal static Func<T, T, T> AssignOnly<T>(string reduction)
    {
        if (reduction != "none")
            throw new ModelSharpException($"ScatterElements reduction '{reduction}' is only supported for Float32.");
        return (_, u) => u;
    }

    private static Tensor<T> Scatter<T>(Tensor<T> data, long[] idx, int[] idxDims, Tensor<T> updates, int axis, Func<T, T, T> combine)
        where T : unmanaged
    {
        ReadOnlySpan<int> dDims = data.Shape.Dimensions;
        int rank = dDims.Length;
        if (axis < 0) axis += rank;
        int axisDim = dDims[axis];
        int[] dStrides = Nd.Strides(dDims);
        int[] idxStrides = Nd.Strides(idxDims);

        var y = new Tensor<T>(data.Shape);
        data.Span.CopyTo(y.Span);
        Span<T> ys = y.Span, us = updates.Span;
        int n = idx.Length;
        for (int p = 0; p < n; p++)
        {
            int dst = 0;
            for (int i = 0; i < rank; i++)
            {
                int c = (p / idxStrides[i]) % idxDims[i];
                if (i == axis)
                {
                    int gi = (int)idx[p];
                    if (gi < 0) gi += axisDim;
                    dst += gi * dStrides[i];
                }
                else dst += c * dStrides[i];
            }
            ys[dst] = combine(ys[dst], us[p]);
        }
        return y;
    }
}
