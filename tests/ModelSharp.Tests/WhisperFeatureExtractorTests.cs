using System;
using ModelSharp.Audio;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Deterministic tests for <see cref="WhisperFeatureExtractor"/> — shape, normalization range, finiteness,
/// the 128-mel mode, and a couple of hand-checkable mel-scale / silence values. No real model required.
/// </summary>
public class WhisperFeatureExtractorTests
{
    private static float[] Sine(int samples, float hz, int sampleRate = WhisperFeatureExtractor.SampleRate)
    {
        var w = new float[samples];
        for (int i = 0; i < samples; i++) w[i] = MathF.Sin(2f * MathF.PI * hz * i / sampleRate);
        return w;
    }

    [Fact]
    public void Extract_Produces_1x80x3000_FiniteTensor()
    {
        var fx = new WhisperFeatureExtractor(80);
        // 1 second of a 440 Hz tone; the extractor pads to 30 s internally.
        Tensor<float> t = fx.Extract(Sine(WhisperFeatureExtractor.SampleRate, 440f));

        Assert.Equal(new[] { 1, 80, 3000 }, t.Shape.Dimensions.ToArray());
        foreach (float v in t.Span) Assert.True(float.IsFinite(v), "feature value must be finite");
    }

    [Fact]
    public void Extract_128MelMode_HasExpectedShape()
    {
        var fx = new WhisperFeatureExtractor(128);
        Assert.Equal(128, fx.NumMels);
        Tensor<float> t = fx.Extract(Sine(WhisperFeatureExtractor.SampleRate, 440f));
        Assert.Equal(new[] { 1, 128, 3000 }, t.Shape.Dimensions.ToArray());
    }

    [Fact]
    public void Normalization_StaysWithinWhisperRange()
    {
        var fx = new WhisperFeatureExtractor(80);
        float[,] mel = fx.LogMel(Sine(2 * WhisperFeatureExtractor.SampleRate, 1000f));

        // Whisper normalization: clamp(log10(mel), max-8) then (x+4)/4. The clamp guarantees the spread
        // never exceeds 8 dex, i.e. (max-min) <= 8/4 = 2.0 after /4. A real tone has near-silent (top)
        // mel bands that hit the clamp floor exactly, so the spread is exactly 2.0 here.
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (float v in mel) { if (v < min) min = v; if (v > max) max = v; }

        Assert.True(float.IsFinite(min) && float.IsFinite(max));
        Assert.True(max - min <= 2.0f + 1e-3f, $"normalized spread {max - min} must not exceed the clamp window of 2.0");
        Assert.Equal(2.0f, max - min, 3); // a tone reaches the floor in its quiet bands
        // Note: Whisper features are NOT bounded above by 1.0 — a loud tone concentrates energy in a
        // few mel bins whose normalized value exceeds 1.0. Only the dynamic range (clamp window) is bounded.
    }

    [Fact]
    public void Silence_Produces_ConstantFlooredFeatures()
    {
        var fx = new WhisperFeatureExtractor(80);
        float[,] mel = fx.LogMel(new float[WhisperFeatureExtractor.SampleRate]); // pure silence

        // For all-zero input every mel power is the 1e-10 floor → log10 = -10 → clamp window collapses
        // (max == every value) → (-10 + 4)/4 = -1.5 everywhere.
        for (int m = 0; m < 80; m++)
            for (int f = 0; f < WhisperFeatureExtractor.NumFrames; f++)
                Assert.Equal(-1.5f, mel[m, f], 3);
    }

    [Fact]
    public void Tone_ConcentratesEnergy_NearItsMelBand()
    {
        // A 1 kHz tone should light up mel bins whose center frequency is near 1 kHz far more than the
        // very-high-frequency bins. We compare average energy of the low-mid bins vs the top bins.
        var fx = new WhisperFeatureExtractor(80);
        float[,] mel = fx.LogMel(Sine(2 * WhisperFeatureExtractor.SampleRate, 1000f));

        double low = 0, high = 0;
        int frames = WhisperFeatureExtractor.NumFrames;
        for (int f = 0; f < frames; f++)
        {
            for (int m = 10; m < 25; m++) low += mel[m, f];   // ~ sub-2 kHz region
            for (int m = 65; m < 80; m++) high += mel[m, f];  // top mel bands (~6-8 kHz)
        }
        low /= 15.0 * frames;
        high /= 15.0 * frames;
        Assert.True(low > high, $"low-band energy {low} should exceed top-band energy {high} for a 1 kHz tone");
    }

    [Fact]
    public void MelScale_IsMonotonic_AndRoundTrips()
    {
        // Slaney mel scale: strictly increasing, and Hz→mel→Hz is the identity.
        double prev = double.NegativeInfinity;
        for (double hz = 0; hz <= 8000; hz += 250)
        {
            double mel = WhisperFeatureExtractor.HzToMelSlaney(hz);
            Assert.True(mel > prev, "mel scale must be strictly increasing");
            prev = mel;
            Assert.Equal(hz, WhisperFeatureExtractor.MelToHzSlaney(mel), 2);
        }
    }

    [Fact]
    public void Filterbank_RowsAreNonNegative_AndCoverBins()
    {
        float[][] fb = WhisperFeatureExtractor.BuildMelFilters(
            WhisperFeatureExtractor.SampleRate, WhisperFeatureExtractor.NFft, nMels: 80);

        Assert.Equal(80, fb.Length);
        Assert.Equal(WhisperFeatureExtractor.NFft / 2 + 1, fb[0].Length);

        bool anyPositive = false;
        foreach (float[] row in fb)
            foreach (float w in row)
            {
                Assert.True(w >= 0f, "Slaney filter weights are non-negative");
                if (w > 0f) anyPositive = true;
            }
        Assert.True(anyPositive, "filterbank must have positive weights");
    }
}
