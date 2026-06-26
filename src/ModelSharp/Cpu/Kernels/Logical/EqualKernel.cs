using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Logical;

/// <summary>
/// ONNX <c>Equal</c>: elementwise <c>A == B</c> with NumPy-style broadcasting.
/// Both inputs must share a dtype; the output is always a Boolean tensor.
/// Supports Float32, Int64, Int32 and Boolean inputs.
/// </summary>
public sealed class EqualKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Equal";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor a = ctx.GetTensor(node.Inputs[0]);
        Tensor b = ctx.GetTensor(node.Inputs[1]);

        if (a.Dtype != b.Dtype)
            throw new ModelSharpException(
                $"Equal requires both inputs to share a dtype, got {a.Dtype} and {b.Dtype}.");

        switch (a.Dtype)
        {
            case ElementType.Float32:
                Compare(a.AsFloat(), b.AsFloat(), node, ctx);
                break;
            case ElementType.Int64:
                Compare(a.AsInt64(), b.AsInt64(), node, ctx);
                break;
            case ElementType.Int32:
                Compare(a.AsInt32(), b.AsInt32(), node, ctx);
                break;
            case ElementType.Boolean:
                Compare(a.AsBool(), b.AsBool(), node, ctx);
                break;
            default:
                throw new ModelSharpException($"Equal does not support dtype {a.Dtype}.");
        }
    }

    /// <summary>
    /// Compares two same-dtype tensors elementwise, broadcasting when shapes differ,
    /// and writes a Boolean result. Fast path for equal shapes.
    /// </summary>
    private static void Compare<T>(Tensor<T> a, Tensor<T> b, GraphNode node, GraphContext ctx)
        where T : unmanaged, IEquatable<T>
    {
        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new bool[(int)a.Shape.Length];
            System.Span<T> sa = a.Span, sb = b.Span;
            for (int i = 0; i < yEqual.Length; i++) yEqual[i] = sa[i].Equals(sb[i]);
            ctx.Set(node.Outputs[0], new Tensor<bool>(a.Shape, yEqual));
            return;
        }

        int[] outd = Nd.BroadcastShape(a.Shape.Dimensions, b.Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = Nd.BroadcastStrides(a.Shape.Dimensions, rank);
        int[] strideB = Nd.BroadcastStrides(b.Shape.Dimensions, rank);

        var buf = new bool[(int)outShape.Length];
        System.Span<T> da = a.Span, db = b.Span;
        int n = buf.Length;
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            buf[idx] = da[aOff].Equals(db[bOff]);
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
        ctx.Set(node.Outputs[0], new Tensor<bool>(outShape, buf));
    }
}
