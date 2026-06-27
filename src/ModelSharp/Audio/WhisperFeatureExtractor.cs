using System;
using System.Numerics;
using ModelSharp.Tensors;

namespace ModelSharp.Audio;

/// <summary>
/// Produces OpenAI Whisper's log-mel spectrogram from a mono PCM waveform — the exact front end the
/// Whisper audio encoder expects. Pure managed (reuses <see cref="Fft"/>); independent of the generic
/// <see cref="MelSpectrogram"/> because Whisper's mel filterbank, padding and normalization differ from
/// the librosa/HTK defaults that file uses.
///
/// <para>Conventions (OpenAI <c>whisper.audio.log_mel_spectrogram</c>):</para>
/// <list type="bullet">
///   <item><description>16 kHz mono input.</description></item>
///   <item><description><c>n_fft = 400</c> (25 ms), <c>hop = 160</c> (10 ms), Hann window (periodic).</description></item>
///   <item><description><c>80</c> mel bins (Whisper tiny…large-v2) or <c>128</c> (large-v3), Slaney-normalized
///   triangular filterbank over <c>0..8000 Hz</c>, Slaney mel scale.</description></item>
///   <item><description>Power spectrum, then <c>log10(max(mel, 1e-10))</c>, clamped to <c>max - 8.0</c>,
///   then <c>(x + 4) / 4</c> normalization to roughly <c>[-1, 1]</c>.</description></item>
///   <item><description>Centered framing (reflection-padded by <c>n_fft/2</c>), trimmed/padded to
///   <c>3000</c> frames (30 s). Whisper drops the final STFT frame so 30 s → exactly 3000 frames.</description></item>
/// </list>
///
/// <para>The output is the encoder input tensor of shape <c>[1, n_mels, 3000]</c> (float32), ready to feed
/// the Whisper encoder graph (typically named <c>input_features</c>).</para>
/// </summary>
public sealed class WhisperFeatureExtractor
{
    /// <summary>Whisper's fixed sample rate (Hz).</summary>
    public const int SampleRate = 16_000;

    /// <summary>STFT window length in samples (25 ms at 16 kHz).</summary>
    public const int NFft = 400;

    /// <summary>STFT hop length in samples (10 ms at 16 kHz).</summary>
    public const int HopLength = 160;

    /// <summary>Number of mel frames the encoder consumes (30 s at 10 ms hop).</summary>
    public const int NumFrames = 3000;

    /// <summary>Number of audio samples in the 30 s analysis window.</summary>
    public const int NumSamples = SampleRate * 30; // 480_000

    /// <summary>FFT size: 400 padded up to the next power of two (512) for the radix-2 transform.</summary>
    private static readonly int FftSize = Fft.NextPow2(NFft); // 512

    private readonly int _nMels;
    private readonly float[][] _filterbank; // [nMels][FftSize/2 + 1]
    private readonly float[] _window;       // periodic Hann, length NFft

    /// <summary>The number of mel bins this extractor produces (80 or 128).</summary>
    public int NumMels => _nMels;

    /// <summary>Creates an extractor for the given mel-bin count (80 for tiny…large-v2, 128 for large-v3).</summary>
    public WhisperFeatureExtractor(int nMels = 80)
    {
        if (nMels != 80 && nMels != 128)
            throw new ArgumentOutOfRangeException(nameof(nMels), nMels, "Whisper supports 80 or 128 mel bins.");
        _nMels = nMels;
        // Build the filterbank against the actual (zero-padded) FFT size so the bin frequencies the
        // radix-2 FFT produces line up with the triangular filters. Whisper's reference uses a true
        // 400-point DFT; padding to 512 yields an interpolated spectrum sampled at 512 bins, so the
        // filterbank must use the same 512-point bin spacing.
        _filterbank = BuildMelFilters(SampleRate, FftSize, nMels);
        _window = HannWindowPeriodic(NFft);
    }

    /// <summary>
    /// Extracts the Whisper log-mel features from a mono 16 kHz waveform and returns the
    /// <c>[1, n_mels, 3000]</c> encoder input tensor. The waveform is padded with zeros (or trimmed) to 30 s.
    /// </summary>
    public Tensor<float> Extract(ReadOnlySpan<float> waveform)
    {
        float[,] mel = LogMel(waveform);                 // [nMels, NumFrames]
        var data = new float[_nMels * NumFrames];
        for (int m = 0; m < _nMels; m++)
            for (int t = 0; t < NumFrames; t++)
                data[m * NumFrames + t] = mel[m, t];
        return new Tensor<float>(new TensorShape(1, _nMels, NumFrames), data);
    }

    /// <summary>
    /// Computes the <c>[n_mels, 3000]</c> normalized log-mel matrix for a waveform (the raw features
    /// before adding the batch axis). Exposed for testing / direct use.
    /// </summary>
    public float[,] LogMel(ReadOnlySpan<float> waveform)
    {
        // 1) Pad/trim to exactly 30 s so the frame count is deterministic (3000 after dropping the last frame).
        var padded = new float[NumSamples];
        int copy = Math.Min(waveform.Length, NumSamples);
        waveform.Slice(0, copy).CopyTo(padded);

        // 2) Reflection-pad by n_fft/2 on each side for centered framing (matches torch.stft(center=True)).
        int pad = NFft / 2;
        var signal = new float[padded.Length + 2 * pad];
        for (int i = 0; i < pad; i++) signal[i] = padded[pad - i];                                  // reflect left
        Array.Copy(padded, 0, signal, pad, padded.Length);
        for (int i = 0; i < pad; i++) signal[pad + padded.Length + i] = padded[padded.Length - 2 - i]; // reflect right

        int fftN = FftSize;                       // 512
        int nBins = fftN / 2 + 1;                 // 257 bins of the 512-point FFT
        // Whisper/torch.stft with center=True over a 480000-sample signal yields 1 + n/hop = 3001 frames,
        // then [..., :-1] drops the last → 3000. We compute exactly NumFrames frames.
        var result = new float[_nMels, NumFrames];

        var buffer = new Complex[fftN];
        var mag2 = new float[nBins];

        float maxLog = float.NegativeInfinity;
        // Pass 1: compute log10 power-mel, tracking the global max for the clamp.
        var logMel = new float[_nMels, NumFrames];
        for (int f = 0; f < NumFrames; f++)
        {
            int start = f * HopLength;
            for (int i = 0; i < fftN; i++)
            {
                if (i < NFft)
                {
                    int idx = start + i;
                    float s = idx < signal.Length ? signal[idx] : 0f;
                    buffer[i] = new Complex(s * _window[i], 0);
                }
                else
                {
                    buffer[i] = Complex.Zero;
                }
            }
            Fft.Forward(buffer);

            for (int b = 0; b < nBins; b++)
            {
                double re = buffer[b].Real, im = buffer[b].Imaginary;
                mag2[b] = (float)(re * re + im * im);  // power spectrum |X|^2
            }

            for (int m = 0; m < _nMels; m++)
            {
                float[] filt = _filterbank[m];
                float sum = 0f;
                for (int b = 0; b < nBins; b++) sum += filt[b] * mag2[b];
                float lg = MathF.Log10(MathF.Max(sum, 1e-10f));
                logMel[m, f] = lg;
                if (lg > maxLog) maxLog = lg;
            }
        }

        // Pass 2: clamp to max - 8 and apply (x + 4) / 4 normalization.
        float floor = maxLog - 8.0f;
        for (int m = 0; m < _nMels; m++)
            for (int f = 0; f < NumFrames; f++)
            {
                float v = MathF.Max(logMel[m, f], floor);
                result[m, f] = (v + 4.0f) / 4.0f;
            }
        return result;
    }

    // ---- Whisper mel filterbank (Slaney scale + Slaney area normalization) ----

    /// <summary>
    /// Builds Whisper's <paramref name="nMels"/> triangular mel filters over <c>nFft/2 + 1</c> bins using the
    /// Slaney mel scale and Slaney (area) normalization — identical to <c>librosa.filters.mel(htk=False,
    /// norm="slaney")</c>, which is what <c>transformers</c> ships in <c>mel_filters</c> for Whisper.
    /// </summary>
    public static float[][] BuildMelFilters(int sampleRate, int nFft, int nMels, float fMin = 0f, float fMax = 8000f)
    {
        int nBins = nFft / 2 + 1;

        // FFT bin center frequencies (Hz) for an nFft-point transform: bin i sits at i * sampleRate / nFft.
        // The extractor passes the real (zero-padded) FFT size here so the filters align with the bins the
        // radix-2 FFT actually produces.
        var fftFreqs = new double[nBins];
        for (int i = 0; i < nBins; i++) fftFreqs[i] = (double)i * sampleRate / nFft;

        // Mel band edges (nMels + 2 points) on the Slaney scale.
        double melMin = HzToMelSlaney(fMin), melMax = HzToMelSlaney(fMax);
        var melPoints = new double[nMels + 2];
        for (int i = 0; i < melPoints.Length; i++)
            melPoints[i] = MelToHzSlaney(melMin + (melMax - melMin) * i / (nMels + 1));

        var fb = new float[nMels][];
        for (int m = 0; m < nMels; m++)
        {
            fb[m] = new float[nBins];
            double left = melPoints[m], center = melPoints[m + 1], right = melPoints[m + 2];
            // Slaney area normalization factor: 2 / (f[m+2] - f[m]).
            double enorm = 2.0 / (right - left);
            for (int b = 0; b < nBins; b++)
            {
                double freq = fftFreqs[b];
                double lower = (freq - left) / (center - left);   // rising edge
                double upper = (right - freq) / (right - center); // falling edge
                double w = Math.Max(0.0, Math.Min(lower, upper));
                fb[m][b] = (float)(w * enorm);
            }
        }
        return fb;
    }

    /// <summary>Hz → mel on the Slaney (auditory toolbox) scale used by Whisper / librosa <c>htk=False</c>.</summary>
    public static double HzToMelSlaney(double hz)
    {
        const double fMin = 0.0;
        const double fSp = 200.0 / 3.0;             // 66.6667 Hz per mel below 1 kHz
        double mel = (hz - fMin) / fSp;
        const double minLogHz = 1000.0;
        double minLogMel = (minLogHz - fMin) / fSp; // 15.0
        double logstep = Math.Log(6.4) / 27.0;
        if (hz >= minLogHz) mel = minLogMel + Math.Log(hz / minLogHz) / logstep;
        return mel;
    }

    /// <summary>Mel → Hz, the inverse of <see cref="HzToMelSlaney"/>.</summary>
    public static double MelToHzSlaney(double mel)
    {
        const double fMin = 0.0;
        const double fSp = 200.0 / 3.0;
        double hz = fMin + fSp * mel;
        const double minLogHz = 1000.0;
        double minLogMel = (minLogHz - fMin) / fSp;
        double logstep = Math.Log(6.4) / 27.0;
        if (mel >= minLogMel) hz = minLogHz * Math.Exp(logstep * (mel - minLogMel));
        return hz;
    }

    /// <summary>Periodic Hann window (the <c>torch.hann_window</c> / Whisper convention, <c>2π i / n</c>).</summary>
    private static float[] HannWindowPeriodic(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / n));
        return w;
    }
}
