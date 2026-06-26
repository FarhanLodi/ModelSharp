using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Logical;

/// <summary>
/// ONNX <c>Where</c>: elementwise select <c>condition ? X : Y</c> with NumPy-style
/// broadcasting over all three inputs. The output preserves the dtype of X/Y
/// (Float32, Int64, Int32, or Boolean are supported).
/// </summary>
public sealed class WhereKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Where";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor cond = ctx.GetTensor(node.Inputs[0]);
        Tensor x = ctx.GetTensor(node.Inputs[1]);
        Tensor y = ctx.GetTensor(node.Inputs[2]);

        if (x.Dtype != y.Dtype)
            throw new ModelSharpException(
                $"Where: X and Y must share a dtype, got {x.Dtype} and {y.Dtype}.");

        Span<bool> c = cond.AsBool().Span;

        switch (x.Dtype)
        {
            case ElementType.Float32:
                ctx.Set(node.Outputs[0],
                    Select<float>(c, cond.Shape, x.AsFloat().Span, x.Shape, y.AsFloat().Span, y.Shape));
                break;
            case ElementType.Int64:
                ctx.Set(node.Outputs[0],
                    Select<long>(c, cond.Shape, x.AsInt64().Span, x.Shape, y.AsInt64().Span, y.Shape));
                break;
            case ElementType.Int32:
                ctx.Set(node.Outputs[0],
                    Select<int>(c, cond.Shape, x.AsInt32().Span, x.Shape, y.AsInt32().Span, y.Shape));
                break;
            case ElementType.Boolean:
                ctx.Set(node.Outputs[0],
                    Select<bool>(c, cond.Shape, x.AsBool().Span, x.Shape, y.AsBool().Span, y.Shape));
                break;
            default:
                throw new ModelSharpException($"Where: unsupported value dtype {x.Dtype}.");
        }
    }

    /// <summary>Broadcasts <paramref name="cond"/>/<paramref name="x"/>/<paramref name="y"/> to a common
    /// shape and selects <c>cond ? x : y</c> elementwise, preserving element type <typeparamref name="T"/>.</summary>
    private static Tensor<T> Select<T>(
        Span<bool> cond, TensorShape condShape,
        Span<T> x, TensorShape xShape,
        Span<T> y, TensorShape yShape) where T : unmanaged
    {
        int[] outd = Nd.BroadcastShape(
            Nd.BroadcastShape(condShape.Dimensions, xShape.Dimensions),
            yShape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] sc = Nd.BroadcastStrides(condShape.Dimensions, rank);
        int[] sx = Nd.BroadcastStrides(xShape.Dimensions, rank);
        int[] sy = Nd.BroadcastStrides(yShape.Dimensions, rank);

        var outBuf = new T[checked((int)outShape.Length)];
        int n = outBuf.Length;
        var coord = new int[rank];
        int cOff = 0, xOff = 0, yOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            outBuf[idx] = cond[cOff] ? x[xOff] : y[yOff];
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                cOff += sc[ax];
                xOff += sx[ax];
                yOff += sy[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                cOff -= sc[ax] * outd[ax];
                xOff -= sx[ax] * outd[ax];
                yOff -= sy[ax] * outd[ax];
            }
        }
        return new Tensor<T>(outShape, outBuf);
    }
}
