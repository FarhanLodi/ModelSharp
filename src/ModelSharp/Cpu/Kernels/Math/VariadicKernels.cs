using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.MathOps;

/// <summary>
/// Shared machinery for variadic elementwise ops (Min / Max / Sum / Mean): folds an arbitrary
/// number of inputs together with NumPy-style broadcasting to a common output shape.
///
/// <para>Dtype-aware (like the broadcast binary kernel): float32/fp16 operands fold via the
/// float <see cref="Combine"/>; integer operands (int64/int32 — e.g. Min/Max clamping a Gathered
/// dynamic dimension against an int64 constant in detection / table-decoder Loop bodies) fold via
/// <see cref="CombineInt64"/> and PRESERVE their integer dtype, so a downstream Reshape/Slice/Gather
/// still receives integers. Without this, integer Min/Max threw "expected Float32".</para>
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

    /// <summary>Integer accumulator seed (overridden by Min/Max; defaults work for Sum/Mean).</summary>
    protected virtual long InitInt64 => 0L;

    /// <summary>Folds one integer input into the running integer accumulator.</summary>
    protected virtual long CombineInt64(long acc, long x) => (long)Combine(acc, x);

    /// <summary>Post-processes the integer accumulator (default: identity; Mean overrides with integer divide).</summary>
    protected virtual long PostprocessInt64(long acc, int count) => acc;

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        int m = node.Inputs.Count;
        var raw = new Tensor[m];
        for (int i = 0; i < m; i++) raw[i] = ctx.GetTensor(node.Inputs[i]);

        // The governing dtype is the first operand's; ONNX type constraints require all to match.
        switch (raw[0].Dtype)
        {
            case ElementType.Int64:
            {
                var ts = new Tensor<long>[m];
                for (int i = 0; i < m; i++) ts[i] = raw[i].AsInt64();
                ctx.Set(node.Outputs[0], Fold(ts, InitInt64, CombineInt64, PostprocessInt64));
                break;
            }
            case ElementType.Int32:
            {
                var ts = new Tensor<int>[m];
                for (int i = 0; i < m; i++) ts[i] = raw[i].AsInt32();
                ctx.Set(node.Outputs[0], Fold(ts, (int)InitInt64,
                    (a, b) => (int)CombineInt64(a, b), (a, c) => (int)PostprocessInt64(a, c)));
                break;
            }
            default:
            {
                var ts = new Tensor<float>[m];
                for (int i = 0; i < m; i++) ts[i] = raw[i].ToFloat32();
                ctx.Set(node.Outputs[0], Fold(ts, Init, Combine, Postprocess));
                break;
            }
        }
    }

    /// <summary>
    /// Variadic broadcast fold, generic over the element type so float32, int64 and int32 share the
    /// same broadcast bookkeeping. <paramref name="post"/> receives (accumulator, input count).
    /// </summary>
    private static Tensor<T> Fold<T>(Tensor<T>[] tensors, T init, Func<T, T, T> combine, Func<T, int, T> post)
        where T : unmanaged
    {
        int m = tensors.Length;
        int[] outd = tensors[0].Shape.Dimensions.ToArray();
        for (int i = 1; i < m; i++) outd = Nd.BroadcastShape(outd, tensors[i].Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int n = (int)outShape.Length;

        var acc = new T[n];
        for (int i = 0; i < n; i++) acc[i] = init;

        for (int i = 0; i < m; i++)
        {
            Span<T> sp = tensors[i].Span;
            int[] st = Nd.BroadcastStrides(tensors[i].Shape.Dimensions, rank);
            var coord = new int[rank];
            int off = 0;
            for (int idx = 0; idx < n; idx++)
            {
                acc[idx] = combine(acc[idx], sp[off]);
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

        var y = new Tensor<T>(outShape);
        Span<T> ys = y.Span;
        for (int i = 0; i < n; i++) ys[i] = post(acc[i], m);
        return y;
    }
}

/// <summary>Elementwise minimum across a variadic input list (ONNX <c>Min</c>).</summary>
public sealed class MinKernel : VariadicElementwiseKernel
{
    public override string OpType => "Min";
    protected override float Init => float.PositiveInfinity;
    protected override float Combine(float acc, float x) => x < acc ? x : acc;
    protected override long InitInt64 => long.MaxValue;
    protected override long CombineInt64(long acc, long x) => x < acc ? x : acc;
}

/// <summary>Elementwise maximum across a variadic input list (ONNX <c>Max</c>).</summary>
public sealed class MaxKernel : VariadicElementwiseKernel
{
    public override string OpType => "Max";
    protected override float Init => float.NegativeInfinity;
    protected override float Combine(float acc, float x) => x > acc ? x : acc;
    protected override long InitInt64 => long.MinValue;
    protected override long CombineInt64(long acc, long x) => x > acc ? x : acc;
}

/// <summary>Elementwise sum across a variadic input list (ONNX <c>Sum</c>).</summary>
public sealed class SumKernel : VariadicElementwiseKernel
{
    public override string OpType => "Sum";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x;
    protected override long InitInt64 => 0L;
    protected override long CombineInt64(long acc, long x) => acc + x;
}

/// <summary>Elementwise mean across a variadic input list (ONNX <c>Mean</c>).</summary>
public sealed class MeanKernel : VariadicElementwiseKernel
{
    public override string OpType => "Mean";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x;
    protected override float Postprocess(float acc, int count) => acc / count;
    protected override long InitInt64 => 0L;
    protected override long CombineInt64(long acc, long x) => acc + x;
    protected override long PostprocessInt64(long acc, int count) => acc / count;
}
