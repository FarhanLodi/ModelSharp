using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.MathOps;

/// <summary>
/// Shared machinery for variadic elementwise ops (Min / Max / Sum / Mean): folds an arbitrary
/// number of inputs together with NumPy-style broadcasting to a common output shape.
/// </summary>
public abstract class VariadicElementwiseKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The accumulator's initial value before any input is folded in.</summary>
    protected abstract float Init { get; }

    /// <summary>Folds one input value into the running accumulator.</summary>
    protected abstract float Combine(float acc, float x);

    /// <summary>Post-processes the accumulator given the input count (default: identity).</summary>
    protected virtual float Postprocess(float acc, int count) => acc;

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        int m = node.Inputs.Count;
        var tensors = new Tensor<float>[m];
        for (int i = 0; i < m; i++) tensors[i] = ctx.Get(node.Inputs[i]);

        int[] outd = tensors[0].Shape.Dimensions.ToArray();
        for (int i = 1; i < m; i++) outd = Nd.BroadcastShape(outd, tensors[i].Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int n = (int)outShape.Length;

        var acc = new float[n];
        for (int i = 0; i < n; i++) acc[i] = Init;

        for (int i = 0; i < m; i++)
        {
            Span<float> sp = tensors[i].Span;
            int[] st = Nd.BroadcastStrides(tensors[i].Shape.Dimensions, rank);
            var coord = new int[rank];
            int off = 0;
            for (int idx = 0; idx < n; idx++)
            {
                acc[idx] = Combine(acc[idx], sp[off]);
                for (int ax = rank - 1; ax >= 0; ax--)
                {
                    coord[ax]++;
                    off += st[ax];
                    if (coord[ax] < outd[ax]) break;
                    coord[ax] = 0;
                    off -= st[ax] * outd[ax];
                }
            }
        }

        var y = new Tensor<float>(outShape, acc);
        Span<float> ys = y.Span;
        for (int i = 0; i < n; i++) ys[i] = Postprocess(acc[i], m);
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>Elementwise minimum across a variadic input list (ONNX <c>Min</c>).</summary>
public sealed class MinKernel : VariadicElementwiseKernel
{
    public override string OpType => "Min";
    protected override float Init => float.PositiveInfinity;
    protected override float Combine(float acc, float x) => x < acc ? x : acc;
}

/// <summary>Elementwise maximum across a variadic input list (ONNX <c>Max</c>).</summary>
public sealed class MaxKernel : VariadicElementwiseKernel
{
    public override string OpType => "Max";
    protected override float Init => float.NegativeInfinity;
    protected override float Combine(float acc, float x) => x > acc ? x : acc;
}

/// <summary>Elementwise sum across a variadic input list (ONNX <c>Sum</c>).</summary>
public sealed class SumKernel : VariadicElementwiseKernel
{
    public override string OpType => "Sum";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x;
}

/// <summary>Elementwise mean across a variadic input list (ONNX <c>Mean</c>).</summary>
public sealed class MeanKernel : VariadicElementwiseKernel
{
    public override string OpType => "Mean";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x;
    protected override float Postprocess(float acc, int count) => acc / count;
}
