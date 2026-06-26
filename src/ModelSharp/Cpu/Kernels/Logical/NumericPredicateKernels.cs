using System;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Logical;

/// <summary>
/// Shared machinery for unary float-input/Boolean-output predicates
/// (<c>IsNaN</c>, <c>IsInf</c>). Reads a Float32 tensor and writes a Boolean tensor
/// of the same shape.
/// </summary>
public abstract class FloatPredicateKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The predicate evaluated per element.</summary>
    protected abstract bool Apply(float x);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        var y = new Tensor<bool>(x.Shape);
        Span<float> xs = x.Span;
        Span<bool> ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = Apply(xs[i]);
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>ONNX <c>IsNaN</c>: elementwise test for NaN, producing a Boolean tensor.</summary>
public sealed class IsNaNKernel : FloatPredicateKernel
{
    public override string OpType => "IsNaN";
    protected override bool Apply(float x) => float.IsNaN(x);
}

/// <summary>ONNX <c>IsInf</c>: elementwise test for +/- infinity, producing a Boolean tensor.</summary>
public sealed class IsInfKernel : FloatPredicateKernel
{
    public override string OpType => "IsInf";
    protected override bool Apply(float x) => float.IsInfinity(x);
}
