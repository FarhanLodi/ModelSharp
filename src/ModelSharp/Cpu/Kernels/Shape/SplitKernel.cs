using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Split</c>: splits <c>data</c> along <c>axis</c> (default 0, negative-normalized) into the
/// node's outputs. Split sizes come from the optional second input (opset 13+) or the <c>split</c>
/// attribute (opset &lt; 13); otherwise the axis is divided as evenly as possible across
/// <c>num_outputs</c> (opset 18) — or, lacking that, the number of declared outputs — with any
/// remainder shrinking the final chunk. Dtype-preserving.
/// </summary>
public sealed class SplitKernel : IKernel
{
    public string OpType => "Split";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        int rank = data.Shape.Rank;
        long axisAttr = Attr.Int(node, "axis", 0);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        int axisDim = data.Shape.Dimensions[axis];

        int[] sizes;
        int[]? splitAttr = Attr.Ints(node, "split");
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
        {
            long[] sp = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
            sizes = new int[sp.Length];
            for (int i = 0; i < sp.Length; i++) sizes[i] = (int)sp[i];
        }
        else if (splitAttr is not null)
        {
            sizes = splitAttr;
        }
        else
        {
            int no = (int)Attr.Int(node, "num_outputs", node.Outputs.Count);
            if (no <= 0) no = node.Outputs.Count;
            int chunk = (axisDim + no - 1) / no;
            sizes = new int[no];
            int rem = axisDim;
            for (int i = 0; i < no; i++) { int c = System.Math.Min(chunk, System.Math.Max(0, rem)); sizes[i] = c; rem -= c; }
        }

        switch (data.Dtype)
        {
            case ElementType.Int64: Split<long>(data.AsInt64(), axis, sizes, node, ctx); break;
            case ElementType.Int32: Split<int>(data.AsInt32(), axis, sizes, node, ctx); break;
            case ElementType.Boolean: Split<bool>(data.AsBool(), axis, sizes, node, ctx); break;
            default: Split<float>(data.AsFloat(), axis, sizes, node, ctx); break;
        }
    }

    private static void Split<T>(Tensor<T> x, int axis, int[] sizes, GraphNode node, GraphContext ctx)
        where T : unmanaged
    {
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        int axisDim = dims[axis];
        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dims[i];
        Span<T> xs = x.Span;

        int offset = 0;
        for (int p = 0; p < sizes.Length; p++)
        {
            int sz = sizes[p];
            int[] outDims = dims.ToArray();
            outDims[axis] = sz;
            var y = new Tensor<T>(new TensorShape(outDims));
            Span<T> ys = y.Span;
            int block = sz * inner;
            for (int o = 0; o < outer; o++)
            {
                int src = (o * axisDim + offset) * inner;
                xs.Slice(src, block).CopyTo(ys.Slice(o * block, block));
            }
            offset += sz;
            ctx.Set(node.Outputs[p], y);
        }
    }
}
