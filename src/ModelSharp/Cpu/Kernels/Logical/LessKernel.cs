using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Logical;

/// <summary>
/// ONNX <c>Less</c>: elementwise <c>A &lt; B</c> with NumPy-style broadcasting,
/// producing a Boolean tensor. Floats are compared as float32; integer inputs
/// (Int64/Int32) are compared in their own integer domain.
/// </summary>
public sealed class LessKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Less";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor a = ctx.GetTensor(node.Inputs[0]);
        Tensor b = ctx.GetTensor(node.Inputs[1]);

        Tensor<bool> y = a.Dtype switch
        {
            ElementType.Int64 => Compare(a.AsInt64(), b.AsInt64(), static (x, z) => x < z),
            ElementType.Int32 => Compare(a.AsInt32(), b.AsInt32(), static (x, z) => x < z),
            ElementType.Float32 => Compare(a.AsFloat(), b.AsFloat(), static (x, z) => x < z),
            _ => throw new ModelSharpException($"Less does not support input dtype {a.Dtype}."),
        };

        ctx.Set(node.Outputs[0], y);
    }

    /// <summary>
    /// Applies <paramref name="less"/> elementwise over <paramref name="a"/> and
    /// <paramref name="b"/> (equal-shape fast path, NumPy broadcasting otherwise),
    /// writing the result into a fresh Boolean tensor.
    /// </summary>
    private static Tensor<bool> Compare<T>(Tensor<T> a, Tensor<T> b, Func<T, T, bool> less)
        where T : unmanaged
    {
        Span<T> sa = a.Span;
        Span<T> sb = b.Span;

        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new Tensor<bool>(a.Shape);
            Span<bool> sy = yEqual.Span;
            for (int i = 0; i < sy.Length; i++) sy[i] = less(sa[i], sb[i]);
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
            dy[idx] = less(sa[aOff], sb[bOff]);
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
