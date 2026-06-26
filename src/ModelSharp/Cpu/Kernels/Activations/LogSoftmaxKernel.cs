using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Activations;

/// <summary>
/// LogSoftmax along an axis (default last): <c>x - max - log(sum(exp(x - max)))</c>,
/// numerically stabilized by max-subtraction. Negative axis is normalized.
/// </summary>
public sealed class LogSoftmaxKernel : IKernel
{
    public string OpType => "LogSoftmax";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        long axisAttr = Attr.Int(node, "axis", -1);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);

        int axisSize = dims[axis];
        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int o = 0; o < outer; o++)
        for (int q = 0; q < inner; q++)
        {
            int baseIdx = o * axisSize * inner + q;
            float mx = float.NegativeInfinity;
            for (int s = 0; s < axisSize; s++) { float v = xs[baseIdx + s * inner]; if (v > mx) mx = v; }
            float sum = 0f;
            for (int s = 0; s < axisSize; s++) sum += MathF.Exp(xs[baseIdx + s * inner] - mx);
            float logSum = MathF.Log(sum);
            for (int s = 0; s < axisSize; s++) ys[baseIdx + s * inner] = xs[baseIdx + s * inner] - mx - logSum;
        }

        ctx.Set(node.Outputs[0], y);
    }
}
