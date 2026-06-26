using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>Concatenate inputs along <c>axis</c>. Dtype-generic.</summary>
public sealed class ConcatKernel : IKernel
{
    public string OpType => "Concat";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        var tensors = new Tensor[node.Inputs.Count];
        for (int i = 0; i < tensors.Length; i++) tensors[i] = ctx.GetTensor(node.Inputs[i]);

        int rank = tensors[0].Shape.Rank;
        long axisAttr = Attr.Int(node, "axis", 0);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);

        ctx.Set(node.Outputs[0], tensors[0].Dtype switch
        {
            ElementType.Int64 => Concat<long>(tensors, axis),
            ElementType.Int32 => Concat<int>(tensors, axis),
            ElementType.Boolean => Concat<bool>(tensors, axis),
            _ => Concat<float>(tensors, axis),
        });
    }

    private static Tensor<T> Concat<T>(Tensor[] tensors, int axis) where T : unmanaged
    {
        System.ReadOnlySpan<int> dims0 = tensors[0].Shape.Dimensions;
        int rank = dims0.Length;

        int outAxis = 0;
        foreach (Tensor t in tensors) outAxis += t.Shape.Dimensions[axis];

        int[] outDims = dims0.ToArray();
        outDims[axis] = outAxis;
        var y = new Tensor<T>(new TensorShape(outDims));
        System.Span<T> ys = y.Span;

        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= outDims[i];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= outDims[i];

        int axisOffset = 0;
        foreach (Tensor tb in tensors)
        {
            var t = (Tensor<T>)tb;
            int ta = t.Shape.Dimensions[axis];
            System.Span<T> ts = t.Span;
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
