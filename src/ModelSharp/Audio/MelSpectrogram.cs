using System;

namespace ModelSharp.Audio;

/// <summary>
/// Triangular mel filterbank and log-mel spectrogram — the standard front end that turns
/// a raw waveform into the input tensor a speech model (e.g. Whisper) expects. Pure managed.
/// </summary>
public static class MelSpectrogram
{
    public static float HzToMel(float hz) => 2595f * MathF.Log10(1f + hz / 700f);

    public static float MelToHz(float mel) => 700f * (MathF.Pow(10f, mel / 2595f) - 1f);

    /// <summary>Builds <paramref name="nMels"/> triangular filters over <c>nFft/2 + 1</c> frequency bins.</summary>
    public static float[][] Filterbank(int nMels, int nFft, int sampleRate, float fMin = 0f, float fMax = 0f)
    {
        if (fMax <= 0f) fMax = sampleRate / 2f;
        int nBins = nFft / 2 + 1;
        float melMin = HzToMel(fMin), melMax = HzToMel(fMax);

        var binOf = new float[nMels + 2];
        for (int i = 0; i < binOf.Length; i++)
        {
            float mel = melMin + (melMax - melMin) * i / (nMels + 1);
            binOf[i] = MelToHz(mel) * nFft / sampleRate;   // fractional bin index
        }

        var fb = new float[nMels][];
        for (int m = 0; m < nMels; m++)
        {
            fb[m] = new float[nBins];
            float left = binOf[m], center = binOf[m + 1], right = binOf[m + 2];
            for (int b = 0; b < nBins; b++)
            {
                float w = 0f;
                if (b >= left && b <= center && center > left) w = (b - left) / (center - left);
                else if (b > center && b <= right && right > center) w = (right - b) / (right - center);
                fb[m][b] = w;
            }
        }
        return fb;
    }

    /// <summary>Computes a [frames × nMels] log-mel spectrogram from a mono signal.</summary>
    public static float[,] LogMel(ReadOnlySpan<float> signal, int sampleRate, int nFft = 400, int hop = 160, int nMels = 80)
    {
        int fftN = Fft.NextPow2(nFft);
        int nBins = fftN / 2 + 1;
        float[][] fb = Filterbank(nMels, fftN, sampleRate);
        float[] window = HannWindow(nFft);

        int frames = signal.Length < nFft ? 1 : 1 + (signal.Length - nFft) / hop;
        var result = new float[frames, nMels];
        var frame = new float[nFft];

        for (int f = 0; f < frames; f++)
        {
            int start = f * hop;
            for (int i = 0; i < nFft; i++)
            {
                int idx = start + i;
                frame[i] = idx < signal.Length ? signal[idx] * window[i] : 0f;
            }

            float[] mag = Fft.MagnitudeSpectrum(frame);   // length fftN/2 + 1 == nBins
            int lim = Math.Min(nBins, mag.Length);
            for (int m = 0; m < nMels; m++)
            {
                float[] filt = fb[m];
                float sum = 0f;
                for (int b = 0; b < lim; b++) sum += filt[b] * mag[b] * mag[b];   // power
                result[f, m] = MathF.Log(MathF.Max(sum, 1e-10f));
            }
        }
        return result;
    }

    private static float[] HannWindow(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / (n - 1)));
        return w;
    }
}
