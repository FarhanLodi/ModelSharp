using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>Permute axes by <c>perm</c> (default reverse). Dtype-preserving.</summary>
public sealed class TransposeKernel : IKernel
{
    public string OpType => "Transpose";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor x = ctx.GetTensor(node.Inputs[0]);
        int[]? perm = Attr.Ints(node, "perm");
        ctx.Set(node.Outputs[0], x.Dtype switch
        {
            ElementType.Int64 => Transpose<long>(x.AsInt64(), perm),
            ElementType.Int32 => Transpose<int>(x.AsInt32(), perm),
            ElementType.Boolean => Transpose<bool>(x.AsBool(), perm),
            _ => Transpose<float>(x.AsFloat(), perm),
        });
    }

    private static Tensor<T> Transpose<T>(Tensor<T> x, int[]? perm) where T : unmanaged
    {
        System.ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        if (perm is null) { perm = new int[rank]; for (int i = 0; i < rank; i++) perm[i] = rank - 1 - i; }

        var outDims = new int[rank];
        for (int i = 0; i < rank; i++) outDims[i] = dims[perm[i]];
        int[] inStrides = Nd.Strides(dims);

        var y = new Tensor<T>(new TensorShape(outDims));
        System.Span<T> xs = x.Span, ys = y.Span;
        int n = (int)x.Length;
        var coord = new int[rank];
        for (int idx = 0; idx < n; idx++)
        {
            int src = 0;
            for (int i = 0; i < rank; i++) src += coord[i] * inStrides[perm[i]];
            ys[idx] = xs[src];
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }
        return y;
    }
}

/// <summary>Gather slices of <c>data</c> along <c>axis</c> by integer indices. Dtype-preserving for data.</summary>
public sealed class GatherKernel : IKernel
{
    public string OpType => "Gather";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        Tensor indicesT = ctx.GetTensor(node.Inputs[1]);
        long[] indices = TensorInts.Read(indicesT);
        int[] idims = indicesT.Shape.Dimensions.ToArray();
        int axis = (int)Attr.Int(node, "axis", 0);

        ctx.Set(node.Outputs[0], data.Dtype switch
        {
            ElementType.Int64 => Gather<long>(data.AsInt64(), indices, idims, axis),
            ElementType.Int32 => Gather<int>(data.AsInt32(), indices, idims, axis),
            ElementType.Boolean => Gather<bool>(data.AsBool(), indices, idims, axis),
            ElementType.UInt8 => Gather<byte>((Tensor<byte>)data, indices, idims, axis),
            ElementType.Int8 => Gather<sbyte>((Tensor<sbyte>)data, indices, idims, axis),
            ElementType.Float64 => Gather<double>((Tensor<double>)data, indices, idims, axis),
            _ => Gather<float>(data.AsFloat(), indices, idims, axis),
        });
    }

    private static Tensor<T> Gather<T>(Tensor<T> data, long[] idx, int[] idims, int axis) where T : unmanaged
    {
        System.ReadOnlySpan<int> dd = data.Shape.Dimensions;
        int rank = dd.Length;
        if (axis < 0) axis += rank;
        int q = idx.Length;
        int axisDim = dd[axis];
        int outer = 1; for (int i = 0; i < axis; i++) outer *= dd[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dd[i];

        var outDims = new int[axis + idims.Length + (rank - axis - 1)];
        int p = 0;
        for (int i = 0; i < axis; i++) outDims[p++] = dd[i];
        for (int i = 0; i < idims.Length; i++) outDims[p++] = idims[i];
        for (int i = axis + 1; i < rank; i++) outDims[p++] = dd[i];

        var y = new Tensor<T>(new TensorShape(outDims));
        System.Span<T> ds = data.Span, ys = y.Span;
        int outPos = 0;
        for (int o = 0; o < outer; o++)
        for (int k = 0; k < q; k++)
        {
            int gi = (int)idx[k];
            if (gi < 0) gi += axisDim;
            int srcBase = (o * axisDim + gi) * inner;
            ds.Slice(srcBase, inner).CopyTo(ys.Slice(outPos, inner));
            outPos += inner;
        }
        return y;
    }
}

/// <summary>Insert size-1 axes (axes from attribute or second input). Dtype-preserving view.</summary>
public sealed class UnsqueezeKernel : IKernel
{
    public string OpType => "Unsqueeze";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor x = ctx.GetTensor(node.Inputs[0]);
        int[]? axes = Attr.Ints(node, "axes");
        if (axes is null && node.Inputs.Count > 1)
        {
            long[] a = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
            axes = new int[a.Length];
            for (int i = 0; i < a.Length; i++) axes[i] = (int)a[i];
        }
        axes ??= System.Array.Empty<int>();

        System.ReadOnlySpan<int> inDims = x.Shape.Dimensions;
        int outRank = inDims.Length + axes.Length;
        var inserted = new HashSet<int>();
        foreach (int ax in axes) inserted.Add(ax < 0 ? ax + outRank : ax);

        var outDims = new int[outRank];
        int j = 0;
        for (int i = 0; i < outRank; i++) outDims[i] = inserted.Contains(i) ? 1 : inDims[j++];
        ctx.Set(node.Outputs[0], x.WithShape(new TensorShape(outDims)));
    }
}

/// <summary>Remove size-1 axes (specified axes, or all size-1 axes). Dtype-preserving view.</summary>
public sealed class SqueezeKernel : IKernel
{
    public string OpType => "Squeeze";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor x = ctx.GetTensor(node.Inputs[0]);
        System.ReadOnlySpan<int> inDims = x.Shape.Dimensions;
        int rank = inDims.Length;

        int[]? axes = Attr.Ints(node, "axes");
        if (axes is null && node.Inputs.Count > 1)
        {
            long[] a = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
            axes = new int[a.Length];
            for (int i = 0; i < a.Length; i++) axes[i] = (int)a[i];
        }

        var kept = new List<int>();
        if (axes is null)
        {
            for (int i = 0; i < rank; i++) if (inDims[i] != 1) kept.Add(inDims[i]);
        }
        else
        {
            var remove = new HashSet<int>();
            foreach (int ax in axes) remove.Add(ax < 0 ? ax + rank : ax);
            for (int i = 0; i < rank; i++) if (!remove.Contains(i)) kept.Add(inDims[i]);
        }
        ctx.Set(node.Outputs[0], x.WithShape(new TensorShape(kept.ToArray())));
    }
}
