using System;
using System.Threading.Tasks;
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

    /// <summary>Below this many elements (rows × norm) the row loop stays serial.</summary>
    private const long NormThreshold = 1L << 16;

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

        float[] inS = KernelSimd.Array(input);
        float[] skS = KernelSimd.Array(skip);
        float[] g = KernelSimd.Array(gamma);
        float[]? b = beta is null ? null : KernelSimd.Array(beta);
        float[]? bi = bias is null ? null : KernelSimd.Array(bias);
        float[] outS = KernelSimd.Array(output);
        float[] sumS = KernelSimd.Array(sum);
        float[]? invs = invStd is null ? null : KernelSimd.Array(invStd);

        void Row(int o)
        {
            int off = o * norm;
            // t = input + skip (+ bias), written to the residual-sum output.
            KernelSimd.Add3(sumS, off, inS, off, skS, off, bi, norm);
            float sumSq = KernelSimd.SumSquares(sumS, off, norm);
            float inv = 1f / MathF.Sqrt(sumSq / norm + eps);
            if (invs is not null) invs[o] = inv;
            // out = t * inv * gamma (+ beta).
            KernelSimd.NormApply(outS, off, sumS, off, inv, g, b, norm);
        }

        if ((long)outer * norm >= NormThreshold)
            Parallel.For(0, outer, Row);
        else
            for (int o = 0; o < outer; o++) Row(o);

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
