using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// Layer normalization over the axes from <c>axis</c> to the end:
/// y = (x − mean)/√(var + ε) · scale + bias. The core transformer normalization.
/// </summary>
public sealed class LayerNormalizationKernel : IKernel
{
    public string OpType => "LayerNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> scale = ctx.Get(node.Inputs[1]);
        bool hasBias = node.Inputs.Count > 2 && node.Inputs[2].Length > 0;
        Tensor<float>? bias = hasBias ? ctx.Get(node.Inputs[2]) : null;

        System.ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        int axis = (int)Attr.Int(node, "axis", -1);
        if (axis < 0) axis += rank;
        float eps = Attr.Float(node, "epsilon", 1e-5f);

        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int norm = 1; for (int i = axis; i < rank; i++) norm *= dims[i];

        var y = new Tensor<float>(x.Shape);
        System.Span<float> xs = x.Span, ys = y.Span, sc = scale.Span;
        System.Span<float> bs = bias is null ? default : bias.Span;

        for (int o = 0; o < outer; o++)
        {
            int b = o * norm;
            float mean = 0f;
            for (int i = 0; i < norm; i++) mean += xs[b + i];
            mean /= norm;

            float varSum = 0f;
            for (int i = 0; i < norm; i++) { float d = xs[b + i] - mean; varSum += d * d; }
            float inv = 1f / MathF.Sqrt(varSum / norm + eps);

            for (int i = 0; i < norm; i++)
            {
                float nval = (xs[b + i] - mean) * inv;
                ys[b + i] = nval * sc[i] + (bias is null ? 0f : bs[i]);
            }
        }
        ctx.Set(node.Outputs[0], y);
    }
}
