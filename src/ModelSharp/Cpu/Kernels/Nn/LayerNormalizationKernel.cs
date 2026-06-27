using System;
using System.Threading.Tasks;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Cpu.Kernels.Llm;
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

    /// <summary>Below this many elements (rows × norm) the row loop stays serial.</summary>
    private const long NormThreshold = 1L << 16;

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
        float[] xs = KernelSimd.Array(x);
        float[] ys = KernelSimd.Array(y);
        float[] sc = KernelSimd.Array(scale);
        float[]? bs = bias is null ? null : KernelSimd.Array(bias);

        void Row(int o)
        {
            int b = o * norm;
            float mean = KernelSimd.Sum(xs, b, norm) / norm;
            float varSum = KernelSimd.SumSquaresCentered(xs, b, mean, norm);
            float inv = 1f / MathF.Sqrt(varSum / norm + eps);
            KernelSimd.NormApplyCentered(ys, b, xs, b, mean, inv, sc, bs, norm);
        }

        if ((long)outer * norm >= NormThreshold)
            Parallel.For(0, outer, Row);
        else
            for (int o = 0; o < outer; o++) Row(o);

        ctx.Set(node.Outputs[0], y);
    }
}
