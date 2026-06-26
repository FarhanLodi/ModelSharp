using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Shared elementwise-with-broadcasting machinery for binary ops (Add, Mul, ...).
/// Fast path for equal shapes; NumPy-style broadcasting otherwise.
///
/// <para>Dtype-aware: float32 operands use <see cref="Apply(float, float)"/>; integer operands
/// (int64/int32 — e.g. position-id / attention-mask arithmetic in transformer graphs) use
/// <see cref="ApplyInt64"/> and PRESERVE their integer dtype, so a downstream Gather/Slice that
/// needs int indices still receives integers. Without this, integer Add/Sub/Mul/Div threw
/// "Tensor dtype is Int64; expected Float32" when a real LLM graph did index math.</para>
/// </summary>
public abstract class BroadcastBinaryKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The scalar operation applied elementwise to float operands.</summary>
    protected abstract float Apply(float a, float b);

    /// <summary>
    /// The scalar operation applied elementwise to integer operands. Defaults to routing through
    /// the float <see cref="Apply(float, float)"/> and truncating; the arithmetic kernels override
    /// this with exact integer semantics (notably Div, which must be integer division, not a float
    /// divide-then-truncate that loses precision on large magnitudes).
    /// </summary>
    protected virtual long ApplyInt64(long a, long b) => (long)Apply(a, b);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor ta = ctx.GetTensor(node.Inputs[0]);
        Tensor tb = ctx.GetTensor(node.Inputs[1]);

        // The governing dtype is the first operand's; ONNX type constraints require both to match.
        switch (ta.Dtype)
        {
            case ElementType.Int64:
                ctx.Set(node.Outputs[0], Broadcast(ta.AsInt64(), tb.AsInt64(), ApplyInt64));
                break;
            case ElementType.Int32:
                ctx.Set(node.Outputs[0], Broadcast(ta.AsInt32(), tb.AsInt32(), (a, b) => (int)ApplyInt64(a, b)));
                break;
            default:
                ctx.Set(node.Outputs[0], Broadcast(ta.AsFloat(), tb.AsFloat(), Apply));
                break;
        }
    }

    /// <summary>
    /// Elementwise binary apply with NumPy-style broadcasting, generic over the element type so the
    /// same logic serves float32, int64 and int32 without duplicating the broadcast bookkeeping.
    /// </summary>
    private static Tensor<T> Broadcast<T>(Tensor<T> a, Tensor<T> b, Func<T, T, T> op) where T : unmanaged
    {
        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new Tensor<T>(a.Shape);
            Span<T> sa = a.Span, sb = b.Span, sy = yEqual.Span;
            for (int i = 0; i < sy.Length; i++) sy[i] = op(sa[i], sb[i]);
            return yEqual;
        }

        int[] outd = Nd.BroadcastShape(a.Shape.Dimensions, b.Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = Nd.BroadcastStrides(a.Shape.Dimensions, rank);
        int[] strideB = Nd.BroadcastStrides(b.Shape.Dimensions, rank);

        var y = new Tensor<T>(outShape);
        Span<T> da = a.Span, db = b.Span, dy = y.Span;
        int n = (int)outShape.Length;
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            dy[idx] = op(da[aOff], db[bOff]);
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
