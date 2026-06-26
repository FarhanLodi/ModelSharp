using ModelSharp.Graph;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Activations;

/// <summary>ReLU: max(0, x), elementwise. (Scalar loop now; TensorPrimitives/intrinsics later.)</summary>
public sealed class ReluKernel : IKernel
{
    public string OpType => "Relu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        var y = new Tensor<float>(x.Shape);

        System.Span<float> xs = x.Span;
        System.Span<float> ys = y.Span;
        for (int i = 0; i < xs.Length; i++)
            ys[i] = xs[i] > 0f ? xs[i] : 0f;

        ctx.Set(node.Outputs[0], y);
    }
}
