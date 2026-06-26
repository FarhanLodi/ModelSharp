using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Llm;

/// <summary>
/// Fused skip-connection + RMSNorm, the ONNXRuntime contrib op
/// <c>SkipSimplifiedLayerNormalization</c>. It first forms the residual sum
/// <c>t = input + skip (+ bias)</c>, then applies RMSNorm scaled by <c>gamma</c>
/// (plus optional <c>beta</c>): <c>output = t / √(mean(t²)+ε) · gamma (+ beta)</c>.
/// Normalization is over the last dimension (the gamma/scale length).
/// </summary>
/// <remarks>
/// Inputs: <c>input</c>, <c>skip</c>, <c>gamma</c>, optional <c>beta</c>, optional <c>bias</c>.
/// Outputs (all but the first optional): <c>output</c>, <c>mean</c> (unused by RMSNorm,
/// left zero), <c>inv_std_dev</c>, and <c>input_skip_bias_sum</c> = the residual sum
/// <c>t</c>. Only outputs that are actually present (non-empty) are written.
/// </remarks>
public sealed class SkipSimplifiedLayerNormalizationKernel : IKernel
{
    public string OpType => "SkipSimplifiedLayerNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> input = ctx.Get(node.Inputs[0]);
        Tensor<float> skip = ctx.Get(node.Inputs[1]);
        Tensor<float> gamma = ctx.Get(node.Inputs[2]);

        bool hasBeta = node.Inputs.Count > 3 && node.Inputs[3].Length > 0;
        bool hasBias = node.Inputs.Count > 4 && node.Inputs[4].Length > 0;
        Tensor<float>? beta = hasBeta ? ctx.Get(node.Inputs[3]) : null;
        Tensor<float>? bias = hasBias ? ctx.Get(node.Inputs[4]) : null;

        float eps = Attr.Float(node, "epsilon", 1e-5f);

        int norm = gamma.Span.Length;
        int total = (int)input.Length;
        int outer = norm == 0 ? 0 : total / norm;

        var output = new Tensor<float>(input.Shape);
        var sum = new Tensor<float>(input.Shape);

        bool wantInv = node.Outputs.Count > 2 && node.Outputs[2].Length > 0;
        var invStd = wantInv ? new Tensor<float>(new TensorShape(outer)) : null;

        Span<float> inS = input.Span, skS = skip.Span, g = gamma.Span;
        Span<float> b = beta is null ? default : beta.Span;
        Span<float> bi = bias is null ? default : bias.Span;
        Span<float> outS = output.Span, sumS = sum.Span;
        Span<float> invs = invStd is null ? default : invStd.Span;

        for (int o = 0; o < outer; o++)
        {
            int off = o * norm;
            float sumSq = 0f;
            for (int i = 0; i < norm; i++)
            {
                float t = inS[off + i] + skS[off + i] + (bias is null ? 0f : bi[i]);
                sumS[off + i] = t;
                sumSq += t * t;
            }

            float inv = 1f / MathF.Sqrt(sumSq / norm + eps);
            if (invStd is not null) invs[o] = inv;

            for (int i = 0; i < norm; i++)
                outS[off + i] = sumS[off + i] * inv * g[i] + (beta is null ? 0f : b[i]);
        }

        ctx.Set(node.Outputs[0], output);
        // Output 1 = mean: undefined for RMSNorm; emit zeros if requested.
        if (node.Outputs.Count > 1 && node.Outputs[1].Length > 0)
            ctx.Set(node.Outputs[1], new Tensor<float>(new TensorShape(outer)));
        if (invStd is not null)
            ctx.Set(node.Outputs[2], invStd);
        // Output 3 = input + skip (+ bias) residual sum.
        if (node.Outputs.Count > 3 && node.Outputs[3].Length > 0)
            ctx.Set(node.Outputs[3], sum);
    }
}
