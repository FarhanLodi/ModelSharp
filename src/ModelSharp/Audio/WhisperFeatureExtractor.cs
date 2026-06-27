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

    /// <summary>
    /// Number of non-negative-frequency bins of the true 400-point DFT (<c>n_fft/2 + 1 = 201</c>).
    /// Bin <c>k</c> sits at exactly <c>k · SampleRate / NFft = k · 40 Hz</c>, matching OpenAI Whisper.
    /// </summary>
    public const int NumBins = NFft / 2 + 1; // 201

    private readonly int _nMels;
    private readonly float[][] _filterbank; // [nMels][NumBins]
    private readonly float[] _window;       // periodic Hann, length NFft
    private readonly BluesteinDft _dft;     // exact 400-point DFT (chirp-z)

    /// <summary>The number of mel bins this extractor produces (80 or 128).</summary>
    public int NumMels => _nMels;

    /// <summary>Creates an extractor for the given mel-bin count (80 for tiny…large-v2, 128 for large-v3).</summary>
    public WhisperFeatureExtractor(int nMels = 80)
    {
        if (nMels != 80 && nMels != 128)
            throw new ArgumentOutOfRangeException(nameof(nMels), nMels, "Whisper supports 80 or 128 mel bins.");
        _nMels = nMels;
        // Build the filterbank against the true 400-point DFT bin grid (201 bins at sr/400 = 40 Hz
        // spacing). This is Whisper-exact: OpenAI uses n_fft = 400, so the STFT yields 201 frequency
        // bins and the mel filters are defined over that grid. (The previous implementation zero-padded
        // each frame 400 -> 512 and built filters on the 257-bin/512-point grid, which resamples the
        // spectrum and slightly mis-aligns the frequency bins.)
        _filterbank = BuildMelFilters(SampleRate, NFft, nMels);
        _window = HannWindowPeriodic(NFft);
        _dft = new BluesteinDft(NFft);
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

        int nBins = NumBins;                      // 201 bins of the true 400-point DFT
        // Whisper/torch.stft with center=True over a 480000-sample signal yields 1 + n/hop = 3001 frames,
        // then [..., :-1] drops the last → 3000. We compute exactly NumFrames frames.
        var result = new float[_nMels, NumFrames];

        var windowed = new float[NFft];           // windowed 400-sample frame (real input to the DFT)
        var spectrum = new Complex[NFft];          // full 400-point DFT output
        var mag2 = new float[nBins];

        float maxLog = float.NegativeInfinity;
        // Pass 1: compute log10 power-mel, tracking the global max for the clamp.
        var logMel = new float[_nMels, NumFrames];
        for (int f = 0; f < NumFrames; f++)
        {
            int start = f * HopLength;
            for (int i = 0; i < NFft; i++)
            {
                int idx = start + i;
                float s = idx < signal.Length ? signal[idx] : 0f;
                windowed[i] = s * _window[i];
            }
            // True 400-point DFT (chirp-z): bin k is exactly k·sr/400 Hz, Whisper-accurate alignment.
            _dft.ForwardReal(windowed, spectrum);

            for (int b = 0; b < nBins; b++)
            {
                double re = spectrum[b].Real, im = spectrum[b].Imaginary;
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
        // The extractor passes the true n_fft (400) so the filters align with the 201 bins of the genuine
        // 400-point DFT (40 Hz spacing) — Whisper-exact.
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
