using System;
using System.Numerics;

namespace ModelSharp.Audio;

/// <summary>
/// Arbitrary-length DFT via Bluestein's algorithm (the chirp-z transform). Computes the genuine
/// <c>N</c>-point DFT for any <c>N</c> by expressing it as a convolution that is evaluated with the
/// power-of-two radix-2 <see cref="Fft"/>. This lets Whisper use a true <c>n_fft = 400</c> DFT so bin
/// <c>k</c> corresponds to frequency <c>k · sampleRate / N</c> Hz — unlike zero-padding 400 → 512, which
/// resamples the spectrum onto the 512-point grid and mis-aligns the frequency bins.
///
/// <para>For an <c>N</c>-point transform: bin <c>k = Σ_n x[n] · exp(-2πi·k·n/N)</c>. Bluestein rewrites
/// <c>k·n = (k² + n² − (k−n)²)/2</c>, turning the sum into <c>a[n] = x[n]·w^{n²/2}</c> convolved with
/// <c>b[m] = w^{−m²/2}</c> (where <c>w = exp(-2πi/N)</c>), then post-multiplied by <c>w^{k²/2}</c>. The
/// linear convolution is computed with two forward FFTs and one inverse FFT on a power-of-two length
/// <c>M ≥ 2N − 1</c>.</para>
/// </summary>
public sealed class BluesteinDft
{
    private readonly int _n;             // logical DFT length
    private readonly int _m;             // power-of-two convolution length (>= 2N-1)
    private readonly Complex[] _wPow;    // chirp w^{k^2/2}, k = 0..N-1  (w = exp(-2πi/N))
    private readonly Complex[] _bFft;    // FFT of the zero-padded b[m] = w^{-m^2/2} kernel

    /// <summary>Builds a reusable Bluestein planner for an <paramref name="n"/>-point DFT.</summary>
    public BluesteinDft(int n)
    {
        if (n < 1) throw new ArgumentOutOfRangeException(nameof(n), n, "DFT length must be >= 1.");
        _n = n;
        _m = Fft.NextPow2(2 * n - 1);

        // Chirp factors. Use exponent (n^2 mod 2n) to keep the angle small and accurate for large n:
        // w^{k^2/2} = exp(-iπ k^2 / N); reduce k^2 modulo 2N before forming the angle.
        _wPow = new Complex[n];
        for (int k = 0; k < n; k++)
        {
            long kk = (long)k * k % (2L * n);
            double ang = -Math.PI * kk / n;          // -π k^2 / N
            _wPow[k] = new Complex(Math.Cos(ang), Math.Sin(ang));
        }

        // Convolution kernel b[m] = conj(chirp) = w^{-m^2/2}, placed at 0 and wrapped to the tail so the
        // linear convolution picks up indices (k - n) in [-(N-1), N-1].
        var b = new Complex[_m];
        b[0] = Complex.One;
        for (int k = 1; k < n; k++)
        {
            Complex v = Complex.Conjugate(_wPow[k]); // conj(chirp) = exp(+iπ k^2 / N)
            b[k] = v;
            b[_m - k] = v;                           // wrap negative lag (k - n) into [-(N-1), N-1]
        }
        Fft.Forward(b);
        _bFft = b;
    }

    /// <summary>The DFT length this planner computes.</summary>
    public int Length => _n;

    /// <summary>
    /// Computes the full <c>N</c>-point DFT of <paramref name="input"/> (length <c>N</c>) into
    /// <paramref name="output"/> (length <c>N</c>). Input and output may not alias.
    /// </summary>
    public void Forward(ReadOnlySpan<Complex> input, Span<Complex> output)
    {
        if (input.Length != _n) throw new ArgumentException($"input length must be {_n}.", nameof(input));
        if (output.Length != _n) throw new ArgumentException($"output length must be {_n}.", nameof(output));

        // a[n] = x[n] * chirp[n], zero-padded to M.
        var a = new Complex[_m];
        for (int i = 0; i < _n; i++) a[i] = input[i] * _wPow[i];
        Fft.Forward(a);

        // Pointwise multiply in the frequency domain, then inverse transform → linear convolution.
        for (int i = 0; i < _m; i++) a[i] *= _bFft[i];
        InverseInPlace(a);

        // Post-multiply by the chirp to recover the DFT bins.
        for (int k = 0; k < _n; k++) output[k] = a[k] * _wPow[k];
    }

    /// <summary>Convenience overload for real input.</summary>
    public void ForwardReal(ReadOnlySpan<float> input, Span<Complex> output)
    {
        if (input.Length != _n) throw new ArgumentException($"input length must be {_n}.", nameof(input));
        var buf = new Complex[_n];
        for (int i = 0; i < _n; i++) buf[i] = new Complex(input[i], 0);
        Forward(buf, output);
    }

    /// <summary>Inverse FFT via conjugation: ifft(x) = conj(fft(conj(x))) / M.</summary>
    private void InverseInPlace(Complex[] x)
    {
        for (int i = 0; i < _m; i++) x[i] = Complex.Conjugate(x[i]);
        Fft.Forward(x);
        double inv = 1.0 / _m;
        for (int i = 0; i < _m; i++) x[i] = Complex.Conjugate(x[i]) * inv;
    }
}
