using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Signal;

/// <summary>
/// ONNX <c>MelWeightMatrix</c> (opset 17). Builds the triangular mel-scale filterbank that maps a
/// linear-frequency DFT spectrum onto mel bins.
///
/// Inputs (all scalars): <c>num_mel_bins</c>, <c>dft_length</c>, <c>sample_rate</c>,
/// <c>lower_edge_hertz</c>, <c>upper_edge_hertz</c>. Attribute: <c>output_datatype</c>
/// (default 1 = FLOAT). Output shape is <c>[floor(dft_length/2)+1, num_mel_bins]</c>.
///
/// hz↔mel use the standard <c>mel = 2595*log10(1 + hz/700)</c> conversion; <c>num_mel_bins+2</c>
/// edge points are spaced uniformly in mel space, converted back to hz, then to fractional DFT bins,
/// and each mel column is a triangle rising to 1 at its center bin.
/// </summary>
public sealed class MelWeightMatrixKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "MelWeightMatrix";

    private static double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
    private static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        int numMelBins = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[0]))[0];
        int dftLength = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[1]))[0];
        int sampleRate = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[2]))[0];
        float lowerEdgeHertz = ctx.Get(node.Inputs[3]).Span[0];
        float upperEdgeHertz = ctx.Get(node.Inputs[4]).Span[0];

        if (numMelBins <= 0) throw new ModelSharpException($"MelWeightMatrix: num_mel_bins must be positive; got {numMelBins}.");
        if (dftLength <= 0) throw new ModelSharpException($"MelWeightMatrix: dft_length must be positive; got {dftLength}.");

        long dtype = Attr.Int(node, "output_datatype", 1);

        int numSpectrogramBins = dftLength / 2 + 1;

        // num_mel_bins + 2 edges spaced uniformly on the mel scale.
        double lowMel = HzToMel(lowerEdgeHertz);
        double highMel = HzToMel(upperEdgeHertz);
        double melStep = (highMel - lowMel) / (numMelBins + 1);

        // Fractional DFT-bin index for each mel edge point: hz * dft_length / sample_rate.
        var binPoints = new double[numMelBins + 2];
        for (int i = 0; i < numMelBins + 2; i++)
        {
            double hz = MelToHz(lowMel + i * melStep);
            binPoints[i] = hz * dftLength / sampleRate;
        }

        // Output matrix [numSpectrogramBins, numMelBins].
        var outShape = new TensorShape(numSpectrogramBins, numMelBins);
        var w = new double[numSpectrogramBins * numMelBins];

        for (int m = 0; m < numMelBins; m++)
        {
            double lower = binPoints[m];
            double center = binPoints[m + 1];
            double upper = binPoints[m + 2];
            for (int bin = 0; bin < numSpectrogramBins; bin++)
            {
                double weight;
                if (bin < lower || bin > upper)
                {
                    weight = 0.0;
                }
                else if (bin <= center)
                {
                    weight = center > lower ? (bin - lower) / (center - lower) : 0.0;
                }
                else
                {
                    weight = upper > center ? (upper - bin) / (upper - center) : 0.0;
                }
                if (weight < 0.0) weight = 0.0;
                w[bin * numMelBins + m] = weight;
            }
        }

        ctx.Set(node.Outputs[0], Materialize(outShape, w, dtype));
    }

    private static Tensor Materialize(TensorShape shape, double[] w, long dtype)
    {
        switch (dtype)
        {
            case 1: // FLOAT
            {
                var b = new float[w.Length];
                for (int i = 0; i < w.Length; i++) b[i] = (float)w[i];
                return new Tensor<float>(shape, b);
            }
            case 11: // DOUBLE
                return new Tensor<double>(shape, (double[])w.Clone());
            default:
                throw new ModelSharpException(
                    $"MelWeightMatrix output_datatype {dtype} is not supported (use 1=FLOAT or 11=DOUBLE).");
        }
    }
}
