using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>CastLike</c>: casts <c>input</c> (input 0) to the element type of the
/// <c>target_type</c> tensor (input 1), preserving <c>input</c>'s shape. Semantically
/// equivalent to <c>Cast</c> with <c>to</c> = dtype(target). Reuses the same numeric
/// conversion table as <see cref="CastKernel"/> (Float32/Float64/Int32/Int64/Bool).
/// </summary>
public sealed class CastLikeKernel : IKernel
{
    public string OpType => "CastLike";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        Tensor target = ctx.GetTensor(node.Inputs[1]);
        ctx.Set(node.Outputs[0], CastKernel.CastTo(input, target.Dtype));
    }
}
