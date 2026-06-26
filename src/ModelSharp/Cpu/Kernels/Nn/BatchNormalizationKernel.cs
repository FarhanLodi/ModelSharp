using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>Batch normalization (inference): y = scale·(x−mean)/√(var+ε) + B, per channel.</summary>
public sealed class BatchNormalizationKernel : IKernel
{
    public string OpType => "BatchNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> scale = ctx.Get(node.Inputs[1]);
        Tensor<float> bvec = ctx.Get(node.Inputs[2]);
        Tensor<float> mean = ctx.Get(node.Inputs[3]);
        Tensor<float> variance = ctx.Get(node.Inputs[4]);
        float eps = Attr.Float(node, "epsilon", 1e-5f);

        System.ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int N = dims[0], C = dims[1];
        int spatial = 1;
        for (int i = 2; i < dims.Length; i++) spatial *= dims[i];

        var y = new Tensor<float>(x.Shape);
        System.Span<float> xs = x.Span, ys = y.Span;
        System.Span<float> sc = scale.Span, bb = bvec.Span, mu = mean.Span, vr = variance.Span;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            float a = sc[c] / MathF.Sqrt(vr[c] + eps);
            float shift = bb[c] - mu[c] * a;
            int baseI = (n * C + c) * spatial;
            for (int s = 0; s < spatial; s++) ys[baseI + s] = xs[baseI + s] * a + shift;
        }

        ctx.Set(node.Outputs[0], y);
    }
}
