using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.MathOps;

/// <summary>
/// ONNX <c>CumSum</c>: cumulative sum along the axis given by the (scalar) second input.
/// Honors the <c>exclusive</c> (default 0) and <c>reverse</c> (default 0) attributes.
/// Negative axis is normalized. Float32 data.
/// </summary>
public sealed class CumSumKernel : IKernel
{
    public string OpType => "CumSum";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        int axis = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[1]))[0];
        if (axis < 0) axis += rank;

        bool exclusive = Attr.Int(node, "exclusive", 0) != 0;
        bool reverse = Attr.Int(node, "reverse", 0) != 0;

        int axisDim = dims[axis];
        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int o = 0; o < outer; o++)
        for (int q = 0; q < inner; q++)
        {
            int baseIdx = o * axisDim * inner + q;
            float running = 0f;
            if (!reverse)
            {
                for (int s = 0; s < axisDim; s++)
                {
                    int idx = baseIdx + s * inner;
                    if (exclusive) { ys[idx] = running; running += xs[idx]; }
                    else { running += xs[idx]; ys[idx] = running; }
                }
            }
            else
            {
                for (int s = axisDim - 1; s >= 0; s--)
                {
                    int idx = baseIdx + s * inner;
                    if (exclusive) { ys[idx] = running; running += xs[idx]; }
                    else { running += xs[idx]; ys[idx] = running; }
                }
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}
