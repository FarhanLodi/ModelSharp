using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Logical;

/// <summary>
/// Shared machinery for the order-comparison ops (<c>GreaterOrEqual</c>,
/// <c>LessOrEqual</c>) that take two same-dtype numeric tensors and produce a
/// Boolean tensor with NumPy-style broadcasting. Floats follow IEEE comparison
/// semantics (NaN comparisons are false); integer inputs compare in their own domain.
/// </summary>
public abstract class OrderingKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>True when the float comparison holds.</summary>
    protected abstract bool ApplyF(float a, float b);

    /// <summary>True when the int64 comparison holds.</summary>
    protected abstract bool ApplyL(long a, long b);

    /// <summary>True when the int32 comparison holds.</summary>
    protected abstract bool ApplyI(int a, int b);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor a = ctx.GetTensor(node.Inputs[0]);
        Tensor b = ctx.GetTensor(node.Inputs[1]);

        Tensor<bool> y = a.Dtype switch
        {
            ElementType.Int64 => Compare(a.AsInt64(), b.AsInt64(), ApplyL),
            ElementType.Int32 => Compare(a.AsInt32(), b.AsInt32(), ApplyI),
            ElementType.Float32 => Compare(a.AsFloat(), b.AsFloat(), ApplyF),
            _ => throw new ModelSharpException($"{OpType} does not support input dtype {a.Dtype}."),
        };
        ctx.Set(node.Outputs[0], y);
    }

    private static Tensor<bool> Compare<T>(Tensor<T> a, Tensor<T> b, Func<T, T, bool> op)
        where T : unmanaged
    {
        Span<T> sa = a.Span;
        Span<T> sb = b.Span;

        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new Tensor<bool>(a.Shape);
            Span<bool> sy = yEqual.Span;
            for (int i = 0; i < sy.Length; i++) sy[i] = op(sa[i], sb[i]);
            return yEqual;
        }

        int[] outd = Nd.BroadcastShape(a.Shape.Dimensions, b.Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = Nd.BroadcastStrides(a.Shape.Dimensions, rank);
        int[] strideB = Nd.BroadcastStrides(b.Shape.Dimensions, rank);

        var y = new Tensor<bool>(outShape);
        Span<bool> dy = y.Span;
        int n = (int)outShape.Length;
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            dy[idx] = op(sa[aOff], sb[bOff]);
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                aOff += strideA[ax];
                bOff += strideB[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                aOff -= strideA[ax] * outd[ax];
                bOff -= strideB[ax] * outd[ax];
            }
        }
        return y;
    }
}

/// <summary>ONNX <c>GreaterOrEqual</c>: elementwise <c>A &gt;= B</c> (broadcasting, Boolean output).</summary>
public sealed class GreaterOrEqualKernel : OrderingKernel
{
    public override string OpType => "GreaterOrEqual";
    protected override bool ApplyF(float a, float b) => a >= b;
    protected override bool ApplyL(long a, long b) => a >= b;
    protected override bool ApplyI(int a, int b) => a >= b;
}

/// <summary>ONNX <c>LessOrEqual</c>: elementwise <c>A &lt;= B</c> (broadcasting, Boolean output).</summary>
public sealed class LessOrEqualKernel : OrderingKernel
{
    public override string OpType => "LessOrEqual";
    protected override bool ApplyF(float a, float b) => a <= b;
    protected override bool ApplyL(long a, long b) => a <= b;
    protected override bool ApplyI(int a, int b) => a <= b;
}
