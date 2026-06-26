using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Llm;

/// <summary>
/// Root-mean-square layer normalization (RMSNorm), as emitted by HuggingFace /
/// ONNXRuntime LLM exports under the name <c>SimplifiedLayerNormalization</c>.
/// Unlike <c>LayerNormalization</c> it subtracts no mean and has no bias:
/// <c>y = x / √(mean(x²) + ε) · scale</c>, where the mean of squares is taken
/// over the axes from <c>axis</c> to the end. An optional second output receives
/// the per-row inverse standard deviation (<c>1/√(mean(x²)+ε)</c>).
/// </summary>
public sealed class SimplifiedLayerNormalizationKernel : IKernel
{
    public string OpType => "SimplifiedLayerNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> scale = ctx.Get(node.Inputs[1]);

        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        int axis = (int)Attr.Int(node, "axis", -1);
        if (axis < 0) axis += rank;
        float eps = Attr.Float(node, "epsilon", 1e-5f);

        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int norm = 1; for (int i = axis; i < rank; i++) norm *= dims[i];

        var y = new Tensor<float>(x.Shape);
        bool wantInv = node.Outputs.Count > 1 && node.Outputs[1].Length > 0;
        var invStd = wantInv ? new Tensor<float>(new TensorShape(outer)) : null;

        Span<float> xs = x.Span, ys = y.Span, sc = scale.Span;
        Span<float> invs = invStd is null ? default : invStd.Span;

        for (int o = 0; o < outer; o++)
        {
            int b = o * norm;
            float sumSq = 0f;
            for (int i = 0; i < norm; i++) { float v = xs[b + i]; sumSq += v * v; }
            float inv = 1f / MathF.Sqrt(sumSq / norm + eps);
            if (invStd is not null) invs[o] = inv;

            for (int i = 0; i < norm; i++)
                ys[b + i] = xs[b + i] * inv * sc[i];
        }

        ctx.Set(node.Outputs[0], y);
        if (invStd is not null) ctx.Set(node.Outputs[1], invStd);
    }
}
