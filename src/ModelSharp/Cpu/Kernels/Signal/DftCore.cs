using System;

namespace ModelSharp.Cpu.Kernels.Signal;

/// <summary>
/// Shared DFT primitives used by the signal-processing kernels (<c>DFT</c>, <c>STFT</c>).
/// The public <see cref="Dft1d"/> primitive computes the full length-N spectrum with an FFT
/// fast path (iterative radix-2 Cooley–Tukey for power-of-two N, Bluestein/chirp-z for arbitrary
/// N — which itself pads to a power-of-two and reuses the radix-2 FFT) and then keeps the
/// requested number of output bins. A naive O(N^2) routine (<see cref="Dft1dNaive"/>) is retained
/// as a reference/fallback and for cross-checking.
/// Conventions match ONNX (opset 17/20): forward uses exponent <c>-2*pi*i*k*n/N</c>, inverse uses
/// <c>+2*pi*i*k*n/N</c> scaled by 1/N.
/// </summary>
internal static class DftCore
{
    /// <summary>
    /// Computes a 1-D DFT of a single complex frame of length <paramref name="n"/>.
    /// <paramref name="reIn"/>/<paramref name="imIn"/> hold the input; results are written to
    /// <paramref name="reOut"/>/<paramref name="imOut"/> which must each have length
    /// <paramref name="outLen"/> (the number of leading output bins to keep, e.g. floor(N/2)+1 for
    /// onesided, or N for two-sided). Uses an FFT fast path internally.
    /// </summary>
    public static void Dft1d(
        ReadOnlySpan<double> reIn, ReadOnlySpan<double> imIn,
        Span<double> reOut, Span<double> imOut,
        int n, int outLen, bool inverse)
    {
        if (n <= 0)
            return;

        // Compute the full length-N spectrum into scratch buffers, then copy the kept bins.
        // For tiny transforms the naive routine is just as fast and avoids allocation; the FFT
        // path matters for the large transforms produced by real models.
        double[] re = new double[n];
        double[] im = new double[n];
        for (int t = 0; t < n; t++)
        {
            re[t] = reIn[t];
            im[t] = imIn.IsEmpty ? 0.0 : imIn[t];
        }

        Transform(re, im, inverse);

        double scale = inverse ? 1.0 / n : 1.0;
        int keep = Math.Min(outLen, n);
        for (int k = 0; k < keep; k++)
        {
            reOut[k] = re[k] * scale;
            imOut[k] = im[k] * scale;
        }
        // Defensive: if more bins were requested than exist, zero the remainder.
        for (int k = keep; k < outLen; k++)
        {
            reOut[k] = 0.0;
            imOut[k] = 0.0;
        }
    }

    /// <summary>
    /// In-place full length-N transform of complex data held in <paramref name="re"/>/<paramref name="im"/>.
    /// Forward when <paramref name="inverse"/> is false (exponent <c>-2*pi*i*k*n/N</c>), un-normalized
    /// inverse otherwise (exponent <c>+2*pi*i*k*n/N</c>, with NO 1/N scaling applied here — the caller
    /// applies the scale). Dispatches to radix-2 FFT for power-of-two N, Bluestein otherwise.
    /// </summary>
    public static void Transform(double[] re, double[] im, bool inverse)
    {
        int n = re.Length;
        if (n <= 1)
            return;

        if (IsPowerOfTwo(n))
            Fft2(re, im, inverse);
        else
            Bluestein(re, im, inverse);
    }

    /// <summary>
    /// Iterative in-place radix-2 Cooley–Tukey FFT. <paramref name="re"/>/<paramref name="im"/> length
    /// MUST be a power of two. No normalization is applied (the inverse is un-scaled).
    /// </summary>
    private static void Fft2(double[] re, double[] im, bool inverse)
    {
        int n = re.Length;
        if (n <= 1)
            return;

        // Bit-reversal permutation.
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        // Butterflies. Forward exponent is -2*pi*i/len; inverse flips the sign.
        double sign = inverse ? +1.0 : -1.0;
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = sign * 2.0 * Math.PI / len;
            double wRe = Math.Cos(ang);
            double wIm = Math.Sin(ang);
            int half = len >> 1;
            for (int i = 0; i < n; i += len)
            {
                double curRe = 1.0, curIm = 0.0;
                for (int k = 0; k < half; k++)
                {
                    int a = i + k;
                    int b = i + k + half;
                    double uRe = re[a], uIm = im[a];
                    double vRe = re[b] * curRe - im[b] * curIm;
                    double vIm = re[b] * curIm + im[b] * curRe;
                    re[a] = uRe + vRe;
                    im[a] = uIm + vIm;
                    re[b] = uRe - vRe;
                    im[b] = uIm - vIm;
                    // Advance the twiddle factor.
                    double nextRe = curRe * wRe - curIm * wIm;
                    double nextIm = curRe * wIm + curIm * wRe;
                    curRe = nextRe;
                    curIm = nextIm;
                }
            }
        }
    }

    /// <summary>
    /// Bluestein (chirp-z) transform for arbitrary length N. Re-expresses the DFT as a convolution
    /// that is evaluated with a power-of-two radix-2 FFT on a padded length M >= 2N-1.
    /// In-place; no normalization is applied.
    /// </summary>
    private static void Bluestein(double[] re, double[] im, bool inverse)
    {
        int n = re.Length;

        // Chirp: w[k] = exp(sign * i * pi * k^2 / n), sign = -1 forward, +1 inverse.
        double sign = inverse ? +1.0 : -1.0;
        var chirpRe = new double[n];
        var chirpIm = new double[n];
        for (int k = 0; k < n; k++)
        {
            // (k^2 mod 2n) keeps the angle accurate for large k.
            long kk = ((long)k * k) % (2L * n);
            double ang = sign * Math.PI * kk / n;
            chirpRe[k] = Math.Cos(ang);
            chirpIm[k] = Math.Sin(ang);
        }

        // Smallest power of two >= 2n - 1.
        int m = 1;
        while (m < 2 * n - 1)
            m <<= 1;

        // a[k] = x[k] * chirp[k], zero-padded to length m.
        var aRe = new double[m];
        var aIm = new double[m];
        for (int k = 0; k < n; k++)
        {
            aRe[k] = re[k] * chirpRe[k] - im[k] * chirpIm[k];
            aIm[k] = re[k] * chirpIm[k] + im[k] * chirpRe[k];
        }

        // b[k] = conj(chirp[k]) for k in [0,n), and mirrored into the tail so the cyclic
        // convolution over length m yields the linear convolution we need.
        var bRe = new double[m];
        var bIm = new double[m];
        bRe[0] = chirpRe[0];
        bIm[0] = -chirpIm[0];
        for (int k = 1; k < n; k++)
        {
            double cRe = chirpRe[k];
            double cIm = -chirpIm[k];
            bRe[k] = cRe;
            bIm[k] = cIm;
            bRe[m - k] = cRe;
            bIm[m - k] = cIm;
        }

        // Convolution via FFT: c = IFFT(FFT(a) .* FFT(b)). The chirp sign cancels in this inner
        // pair, so we use forward/inverse plain (un-normalized) radix-2 FFTs and divide by m.
        Fft2(aRe, aIm, inverse: false);
        Fft2(bRe, bIm, inverse: false);
        for (int k = 0; k < m; k++)
        {
            double pr = aRe[k] * bRe[k] - aIm[k] * bIm[k];
            double pi = aRe[k] * bIm[k] + aIm[k] * bRe[k];
            aRe[k] = pr;
            aIm[k] = pi;
        }
        Fft2(aRe, aIm, inverse: true);
        double inv = 1.0 / m;

        // X[k] = chirp[k] * c[k].
        for (int k = 0; k < n; k++)
        {
            double cr = aRe[k] * inv;
            double ci = aIm[k] * inv;
            re[k] = cr * chirpRe[k] - ci * chirpIm[k];
            im[k] = cr * chirpIm[k] + ci * chirpRe[k];
        }
    }

    /// <summary>
    /// Reference naive O(N^2) DFT. Retained for cross-checking the FFT path; not used on the hot
    /// path. Same conventions and arguments as <see cref="Dft1d"/>.
    /// </summary>
    public static void Dft1dNaive(
        ReadOnlySpan<double> reIn, ReadOnlySpan<double> imIn,
        Span<double> reOut, Span<double> imOut,
        int n, int outLen, bool inverse)
    {
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
                sumRe += xr * c - xi * s;
                sumIm += xr * s + xi * c;
            }
            reOut[k] = sumRe * scale;
            imOut[k] = sumIm * scale;
        }
    }

    private static bool IsPowerOfTwo(int x) => x > 0 && (x & (x - 1)) == 0;

    /// <summary>Number of bins kept along the transformed axis.</summary>
    public static int OutputBins(int n, bool onesided) => onesided ? n / 2 + 1 : n;
}
