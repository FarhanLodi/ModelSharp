using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Size</c>: returns the total number of elements of the input as a
/// scalar Int64 tensor. Dtype-agnostic — only the shape is read.
/// </summary>
public sealed class SizeKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Size";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        ctx.Set(node.Outputs[0], new Tensor<long>(new TensorShape(), new[] { input.Shape.Length }));
    }
}
