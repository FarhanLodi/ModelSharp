using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Pad</c>: pads (or crops, for negative pads) <c>data</c>. <c>pads</c> is the
/// <c>[begin_0..begin_{r-1}, end_0..end_{r-1}]</c> int vector from the second input (opset 11+)
/// or the <c>pads</c> attribute (opset 2). The optional <c>axes</c> input/attr (opset 18) restricts
/// padding to selected axes. Modes <c>constant</c> (default), <c>edge</c>, and <c>reflect</c> are
/// supported; the constant fill comes from the optional third input or the <c>value</c> attribute.
/// Dtype-preserving (Float32 / Int64 / Int32 / Boolean).
/// </summary>
public sealed class PadKernel : IKernel
{
    public string OpType => "Pad";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        int rank = data.Shape.Rank;
        string mode = Attr.Str(node, "mode", "constant");

        long[] padsRaw;
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
            padsRaw = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        else
        {
            int[]? pa = Attr.Ints(node, "pads");
            padsRaw = pa is null ? Array.Empty<long>() : Array.ConvertAll(pa, v => (long)v);
        }

        long[]? axes = null;
        if (node.Inputs.Count > 3 && !string.IsNullOrEmpty(node.Inputs[3]))
            axes = TensorInts.Read(ctx.GetTensor(node.Inputs[3]));

        var begin = new int[rank];
        var end = new int[rank];
        if (axes is null)
        {
            for (int i = 0; i < rank; i++) { begin[i] = (int)padsRaw[i]; end[i] = (int)padsRaw[rank + i]; }
        }
        else
        {
            int na = axes.Length;
            for (int j = 0; j < na; j++)
            {
                int ax = (int)axes[j];
                if (ax < 0) ax += rank;
                begin[ax] = (int)padsRaw[j];
                end[ax] = (int)padsRaw[na + j];
            }
        }

        switch (data.Dtype)
        {
            case ElementType.Int64:
                ctx.Set(node.Outputs[0], Pad<long>(data.AsInt64(), begin, end, ReadFillI64(node, ctx), mode));
                break;
            case ElementType.Int32:
                ctx.Set(node.Outputs[0], Pad<int>(data.AsInt32(), begin, end, (int)ReadFillI64(node, ctx), mode));
                break;
            case ElementType.Boolean:
                ctx.Set(node.Outputs[0], Pad<bool>(data.AsBool(), begin, end, ReadFillI64(node, ctx) != 0, mode));
                break;
            default:
                ctx.Set(node.Outputs[0], Pad<float>(data.AsFloat(), begin, end, ReadFillF32(node, ctx), mode));
                break;
        }
    }

    private static float ReadFillF32(GraphNode node, GraphContext ctx)
    {
        if (node.Inputs.Count > 2 && !string.IsNullOrEmpty(node.Inputs[2]))
            return ctx.GetTensor(node.Inputs[2]).AsFloat().Span[0];
        return Attr.Float(node, "value", 0f);
    }

    private static long ReadFillI64(GraphNode node, GraphContext ctx)
    {
        if (node.Inputs.Count > 2 && !string.IsNullOrEmpty(node.Inputs[2]))
            return TensorInts.Read(ctx.GetTensor(node.Inputs[2]))[0];
        return (long)Attr.Float(node, "value", 0f);
    }

    private static Tensor<T> Pad<T>(Tensor<T> x, int[] begin, int[] end, T fill, string mode) where T : unmanaged
    {
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        int[] inStrides = Nd.Strides(dims);

        var outDims = new int[rank];
        for (int i = 0; i < rank; i++) outDims[i] = dims[i] + begin[i] + end[i];

        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> xs = x.Span, ys = y.Span;
        int n = (int)y.Shape.Length;
        bool constant = mode == "constant";
        var coord = new int[rank];

        for (int idx = 0; idx < n; idx++)
        {
            bool inside = true;
            int src = 0;
            for (int i = 0; i < rank; i++)
            {
                int sc = coord[i] - begin[i];
                int d = dims[i];
                if (sc < 0 || sc >= d)
                {
                    if (constant) { inside = false; break; }
                    sc = mode == "edge" ? (sc < 0 ? 0 : d - 1) : Reflect(sc, d);
                }
                src += sc * inStrides[i];
            }
            ys[idx] = inside ? xs[src] : fill;
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }
        return y;
    }

    /// <summary>NumPy-style 'reflect' mapping for an out-of-range index (edge not repeated).</summary>
    private static int Reflect(int i, int d)
    {
        if (d == 1) return 0;
        int period = 2 * (d - 1);
        int m = ((i % period) + period) % period;
        return m < d ? m : period - m;
    }
}
