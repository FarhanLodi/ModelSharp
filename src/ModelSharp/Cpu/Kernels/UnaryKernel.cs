using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels;

/// <summary>Shared machinery for elementwise unary ops (Tanh, Sqrt, Erf, ...).</summary>
public abstract class UnaryKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The scalar operation applied elementwise.</summary>
    protected abstract float Apply(float x);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        var y = new Tensor<float>(x.Shape);
        System.Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = Apply(xs[i]);
        ctx.Set(node.Outputs[0], y);
    }
}
