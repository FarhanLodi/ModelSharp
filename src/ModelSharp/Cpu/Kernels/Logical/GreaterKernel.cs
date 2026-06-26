using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Logical;

/// <summary>
/// Elementwise <c>A &gt; B</c> with NumPy-style broadcasting. Both inputs must share
/// the same numeric dtype (Float32/Int64/Int32); the output is always Boolean.
/// </summary>
public sealed class GreaterKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Greater";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor a = ctx.GetTensor(node.Inputs[0]);
        Tensor b = ctx.GetTensor(node.Inputs[1]);

        // ONNX Greater requires both operands to be the same type T; branch on A's
        // dtype and read B through the matching typed view. Use the native '>' so
        // float NaN comparisons follow IEEE semantics (always false).
        Tensor<bool> y = a.Dtype switch
        {
            ElementType.Int64 => Compare(a.AsInt64(), b.AsInt64(), static (x, z) => x > z),
            ElementType.Int32 => Compare(a.AsInt32(), b.AsInt32(), static (x, z) => x > z),
            _ => Compare(a.AsFloat(), b.AsFloat(), static (x, z) => x > z),
        };
        ctx.Set(node.Outputs[0], y);
    }

    /// <summary>
    /// Applies <paramref name="gt"/> elementwise to <paramref name="a"/> and
    /// <paramref name="b"/>, producing a Boolean tensor. Fast path for equal shapes;
    /// NumPy-style broadcasting otherwise.
    /// </summary>
    private static Tensor<bool> Compare<T>(Tensor<T> a, Tensor<T> b, Func<T, T, bool> gt)
        where T : unmanaged
    {
        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new Tensor<bool>(a.Shape);
            System.Span<T> sa = a.Span, sb = b.Span;
            System.Span<bool> sy = yEqual.Span;
            for (int i = 0; i < sy.Length; i++) sy[i] = gt(sa[i], sb[i]);
            return yEqual;
        }

        int[] outd = Nd.BroadcastShape(a.Shape.Dimensions, b.Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = Nd.BroadcastStrides(a.Shape.Dimensions, rank);
        int[] strideB = Nd.BroadcastStrides(b.Shape.Dimensions, rank);

        var y = new Tensor<bool>(outShape);
        System.Span<T> da = a.Span, db = b.Span;
        System.Span<bool> dy = y.Span;
        int n = (int)outShape.Length;
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            dy[idx] = gt(da[aOff], db[bOff]);
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
