using System;
using ModelSharp.Graph;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Activations;

/// <summary>Logistic sigmoid: 1 / (1 + e^-x), elementwise.</summary>
public sealed class SigmoidKernel : IKernel
{
    public string OpType => "Sigmoid";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        var y = new Tensor<float>(x.Shape);

        Span<float> xs = x.Span;
        Span<float> ys = y.Span;
        for (int i = 0; i < xs.Length; i++)
            ys[i] = 1f / (1f + MathF.Exp(-xs[i]));

        ctx.Set(node.Outputs[0], y);
    }
}
