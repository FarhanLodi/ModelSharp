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

    /// <summary>
    /// When true, an integer (Int64/Int32) input is passed through unchanged instead of being
    /// coerced to float. Correct for round-to-integer ops (Ceil/Floor/Round) whose result on an
    /// already-integer tensor is the input itself — integer shape/size math in CNN exporters routes
    /// through these. Default false: genuinely float-only ops (Sin, Exp, Sqrt, ...) keep throwing on
    /// integer dtypes, matching ONNX type constraints.
    /// </summary>
    protected virtual bool IntegerIdentity => false;

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor xt = ctx.GetTensor(node.Inputs[0]);
        if (IntegerIdentity && (xt.Dtype == ElementType.Int64 || xt.Dtype == ElementType.Int32))
        {
            ctx.Set(node.Outputs[0], xt);   // ceil/floor/round of an integer is the integer itself
            return;
        }

        Tensor<float> x = xt.ToFloat32();
        var y = new Tensor<float>(x.Shape);
        System.Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = Apply(xs[i]);
        ctx.Set(node.Outputs[0], y);
    }
}
