using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Generators;

/// <summary>
/// ONNX <c>Multinomial</c>: draws class indices from a batch of categorical distributions. Input is
/// <c>[batch, num_classes]</c> of unnormalized log-probabilities (logits). For each batch row,
/// <c>sample_size</c> (default 1) draws are taken with replacement, producing an output of shape
/// <c>[batch, sample_size]</c> in the integer <c>dtype</c> (default Int32, TensorProto 6, or Int64
/// 7). The optional <c>seed</c> attribute makes draws reproducible.
/// </summary>
public sealed class MultinomialKernel : IKernel
{
    public string OpType => "Multinomial";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> logits = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> d = logits.Shape.Dimensions;
        if (d.Length != 2) throw new ModelSharpException("Multinomial: input must be [batch, num_classes].");
        int batch = d[0], classes = d[1];
        int sampleSize = (int)Attr.Int(node, "sample_size", 1);
        long dtype = Attr.Int(node, "dtype", 6);

        Random rng = RandomGen.MakeRng(node);
        Span<float> ls = logits.Span;
        var outIdx = new long[batch * sampleSize];

        var cum = new double[classes];
        for (int b = 0; b < batch; b++)
        {
            // Softmax -> cumulative distribution.
            int baseOff = b * classes;
            double max = double.NegativeInfinity;
            for (int c = 0; c < classes; c++) max = Math.Max(max, ls[baseOff + c]);
            double sum = 0;
            for (int c = 0; c < classes; c++) { sum += Math.Exp(ls[baseOff + c] - max); cum[c] = sum; }

            for (int s = 0; s < sampleSize; s++)
            {
                double u = rng.NextDouble() * sum;
                int pick = classes - 1;
                for (int c = 0; c < classes; c++) { if (u < cum[c]) { pick = c; break; } }
                outIdx[b * sampleSize + s] = pick;
            }
        }

        var shape = new TensorShape(batch, sampleSize);
        ctx.Set(node.Outputs[0], dtype == 7
            ? Tensor<long>.FromArray(shape, outIdx)
            : Tensor<int>.FromArray(shape, Array.ConvertAll(outIdx, v => (int)v)));
    }
}
