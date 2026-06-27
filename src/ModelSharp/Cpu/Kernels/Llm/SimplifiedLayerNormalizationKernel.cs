using System;
using System.Threading.Tasks;
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

    /// <summary>Below this many elements (rows × norm) the row loop stays serial.</summary>
    private const long NormThreshold = 1L << 16;

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

        // Flat backing arrays so each row can be handled in a Parallel.For body.
        float[] xs = KernelSimd.Array(x);
        float[] ys = KernelSimd.Array(y);
        float[] sc = KernelSimd.Array(scale);
        float[]? invs = invStd is null ? null : KernelSimd.Array(invStd);

        void Row(int o)
        {
            int b = o * norm;
            float sumSq = KernelSimd.SumSquares(xs, b, norm);
            float inv = 1f / MathF.Sqrt(sumSq / norm + eps);
            if (invs is not null) invs[o] = inv;
            KernelSimd.NormApply(ys, b, xs, b, inv, sc, null, norm);
        }

        // Parallelize across the independent rows when the work is large enough.
        if ((long)outer * norm >= NormThreshold)
            Parallel.For(0, outer, Row);
        else
            for (int o = 0; o < outer; o++) Row(o);

        ctx.Set(node.Outputs[0], y);
        if (invStd is not null) ctx.Set(node.Outputs[1], invStd);
    }
}
