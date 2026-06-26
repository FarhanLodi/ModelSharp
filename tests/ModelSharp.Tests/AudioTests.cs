using System;
using System.Numerics;
using ModelSharp.Audio;
using Xunit;

namespace ModelSharp.Tests;

public class AudioTests
{
    [Fact]
    public void Fft_Of_Dc_Signal_Has_Energy_Only_In_Bin0()
    {
        var data = new[] { Complex.One, Complex.One, Complex.One, Complex.One };
        Fft.Forward(data);
        Assert.Equal(4.0, data[0].Real, 6);
        for (int i = 1; i < data.Length; i++) Assert.True(data[i].Magnitude < 1e-9);
    }

    [Fact]
    public void Fft_Of_Impulse_Is_Flat()
    {
        var data = new[] { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero };
        Fft.Forward(data);
        foreach (Complex c in data) Assert.Equal(1.0, c.Magnitude, 6);
    }

    [Fact]
    public void MagnitudeSpectrum_Detects_Single_Tone()
    {
        // A pure cosine at k cycles over N samples peaks at bin k.
        const int n = 64, k = 8;
        var signal = new float[n];
        for (int i = 0; i < n; i++) signal[i] = MathF.Cos(2f * MathF.PI * k * i / n);

        float[] mag = Fft.MagnitudeSpectrum(signal);
        int peak = 0;
        for (int i = 1; i < mag.Length; i++) if (mag[i] > mag[peak]) peak = i;
        Assert.Equal(k, peak);
    }

    [Fact]
    public void LogMel_Has_Expected_Shape_And_Is_Finite()
    {
        const int sr = 16000;
        var signal = new float[sr / 10];   // 0.1 s
        for (int i = 0; i < signal.Length; i++) signal[i] = MathF.Sin(2f * MathF.PI * 440f * i / sr);

        float[,] mel = MelSpectrogram.LogMel(signal, sr, nFft: 400, hop: 160, nMels: 80);
        Assert.Equal(80, mel.GetLength(1));
        Assert.True(mel.GetLength(0) > 1);
        foreach (float v in mel) Assert.True(float.IsFinite(v));
    }
}
