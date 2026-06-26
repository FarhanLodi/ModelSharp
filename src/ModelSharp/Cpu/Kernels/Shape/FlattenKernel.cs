using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>Flatten to 2-D about <c>axis</c> (default 1): [d0..axis) × [axis..end). Dtype-preserving view.</summary>
public sealed class FlattenKernel : IKernel
{
    public string OpType => "Flatten";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor x = ctx.GetTensor(node.Inputs[0]);
        System.ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        long axisAttr = Attr.Int(node, "axis", 1);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);

        int d0 = 1;
        for (int i = 0; i < axis; i++) d0 *= dims[i];
        int d1 = 1;
        for (int i = axis; i < rank; i++) d1 *= dims[i];

        ctx.Set(node.Outputs[0], x.WithShape(new TensorShape(d0, d1)));
    }
}
