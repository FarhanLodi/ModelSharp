using System;

namespace ModelSharp.Cpu.Kernels.Signal;

/// <summary>
/// Shared DFT primitives used by the signal-processing kernels (<c>DFT</c>, <c>STFT</c>).
/// A straightforward O(N^2) naive DFT is used for correctness; a radix-2 path handles
/// power-of-two lengths for speed. Conventions match ONNX (opset 17):
/// forward uses exponent <c>-2*pi*i*k*n/N</c>, inverse uses <c>+2*pi*i*k*n/N</c> scaled by 1/N.
/// </summary>
internal static class DftCore
{
    /// <summary>
    /// Computes a 1-D DFT of a single complex frame of length <paramref name="n"/>.
    /// <paramref name="reIn"/>/<paramref name="imIn"/> hold the input; results are written to
    /// <paramref name="reOut"/>/<paramref name="imOut"/> which must each have length
    /// <paramref name="outLen"/> (the number of output bins to keep, e.g. floor(N/2)+1 for onesided).
    /// </summary>
    public static void Dft1d(
        ReadOnlySpan<double> reIn, ReadOnlySpan<double> imIn,
        Span<double> reOut, Span<double> imOut,
        int n, int outLen, bool inverse)
    {
        // Sign of the exponent: forward -1, inverse +1.
        double sign = inverse ? +1.0 : -1.0;
        double scale = inverse ? 1.0 / n : 1.0;

        for (int k = 0; k < outLen; k++)
        {
            double sumRe = 0.0, sumIm = 0.0;
            for (int t = 0; t < n; t++)
            {
                double xr = reIn[t];
                double xi = imIn.IsEmpty ? 0.0 : imIn[t];
                double ang = sign * 2.0 * Math.PI * k * t / n;
                double c = Math.Cos(ang);
                double s = Math.Sin(ang);
                // (xr + i*xi) * (c + i*s)
                sumRe += xr * c - xi * s;
                sumIm += xr * s + xi * c;
            }
            reOut[k] = sumRe * scale;
            imOut[k] = sumIm * scale;
        }
    }

    /// <summary>Number of bins kept along the transformed axis.</summary>
    public static int OutputBins(int n, bool onesided) => onesided ? n / 2 + 1 : n;
}
