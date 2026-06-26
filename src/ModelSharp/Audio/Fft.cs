using System;
using System.Numerics;

namespace ModelSharp.Audio;

/// <summary>Radix-2 Cooley–Tukey FFT — pure managed, no native dependency. Lengths are powers of two.</summary>
public static class Fft
{
    /// <summary>In-place forward FFT. <paramref name="data"/> length must be a power of two.</summary>
    public static void Forward(Complex[] data)
    {
        int n = data.Length;
        if (n <= 1) return;
        if ((n & (n - 1)) != 0) throw new ArgumentException("FFT length must be a power of two.", nameof(data));

        // Bit-reversal permutation.
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (data[i], data[j]) = (data[j], data[i]);
        }

        // Butterflies.
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = -2.0 * Math.PI / len;
            var wLen = new Complex(Math.Cos(ang), Math.Sin(ang));
            for (int i = 0; i < n; i += len)
            {
                Complex w = Complex.One;
                int half = len / 2;
                for (int k = 0; k < half; k++)
                {
                    Complex u = data[i + k];
                    Complex v = data[i + k + half] * w;
                    data[i + k] = u + v;
                    data[i + k + half] = u - v;
                    w *= wLen;
                }
            }
        }
    }

    /// <summary>Magnitude spectrum (bins 0..n/2) of a real signal, zero-padded to the next power of two.</summary>
    public static float[] MagnitudeSpectrum(ReadOnlySpan<float> signal)
    {
        int n = NextPow2(signal.Length);
        var data = new Complex[n];
        for (int i = 0; i < signal.Length; i++) data[i] = new Complex(signal[i], 0);
        Forward(data);

        var mag = new float[n / 2 + 1];
        for (int i = 0; i < mag.Length; i++) mag[i] = (float)data[i].Magnitude;
        return mag;
    }

    /// <summary>Smallest power of two ≥ <paramref name="n"/>.</summary>
    public static int NextPow2(int n)
    {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }
}
