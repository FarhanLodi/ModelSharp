using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Activations;

/// <summary>Softmax along an axis (default last), numerically stabilized by max-subtraction.</summary>
public sealed class SoftmaxKernel : IKernel
{
    public string OpType => "Softmax";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        long axisAttr = Attr.Int(node, "axis", -1);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);

        int axisSize = dims[axis];
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        var y = new Tensor<float>(x.Shape);
        System.Span<float> xs = x.Span, ys = y.Span;
        for (int o = 0; o < outer; o++)
        for (int q = 0; q < inner; q++)
        {
            int baseIdx = o * axisSize * inner + q;
            float mx = float.NegativeInfinity;
            for (int s = 0; s < axisSize; s++) { float v = xs[baseIdx + s * inner]; if (v > mx) mx = v; }
            float sum = 0f;
            for (int s = 0; s < axisSize; s++) { float e = MathF.Exp(xs[baseIdx + s * inner] - mx); ys[baseIdx + s * inner] = e; sum += e; }
            for (int s = 0; s < axisSize; s++) ys[baseIdx + s * inner] /= sum;
        }

        ctx.Set(node.Outputs[0], y);
    }
}
