using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Tensors;

namespace ModelSharp.Audio;

/// <summary>
/// How the per-frame emission values should be interpreted by the decoder.
/// Greedy decoding is invariant to this (argmax is monotone) — it only matters for beam search,
/// which needs normalized probabilities to sum path mass correctly.
/// </summary>
public enum CtcInputKind
{
    /// <summary>Raw, unnormalized logits. Beam search applies log-softmax per frame.</summary>
    Logits,

    /// <summary>Already log-softmaxed log-probabilities; used as-is.</summary>
    LogProbabilities,

    /// <summary>Already softmaxed probabilities in [0, 1]; their log is taken internally.</summary>
    Probabilities,
}

/// <summary>A decoded CTC hypothesis: a labeling (token indices) and its log-probability score.</summary>
public readonly struct CtcHypothesis
{
    /// <summary>Creates a hypothesis from a token-index labeling and its log-probability.</summary>
    public CtcHypothesis(IReadOnlyList<int> tokens, float logProbability)
    {
        Tokens = tokens;
        LogProbability = logProbability;
    }

    /// <summary>The decoded labeling as token indices (blanks removed, repeats collapsed).</summary>
    public IReadOnlyList<int> Tokens { get; }

    /// <summary>The natural-log probability of the labeling (sum over all collapsing paths, in log-space).</summary>
    public float LogProbability { get; }
}

/// <summary>
/// CTC (Connectionist Temporal Classification) decoding for acoustic models whose output is a
/// per-frame token distribution of shape <c>[T frames, V vocab]</c> (e.g. wav2vec2 / DeepSpeech).
/// Pure managed and model-agnostic.
/// </summary>
/// <remarks>
/// End-to-end speech path this pairs with:
/// <list type="number">
///   <item><description>waveform → <see cref="MelSpectrogram.LogMel"/> (front end, already in this module);</description></item>
///   <item><description>features → acoustic model run via an <c>IExecutionEngine</c> elsewhere, producing an
///   emission <see cref="Tensor{T}"/> of shape <c>[T, V]</c> (logits or log-probs);</description></item>
///   <item><description>emissions → <see cref="GreedyDecode(Tensor{float}, int)"/> or
///   <see cref="BeamSearchDecode(Tensor{float}, int, int, CtcInputKind, int)"/>, then
///   <see cref="CtcVocabulary.Decode"/> to obtain text.</description></item>
/// </list>
/// The decoder never loads or requires a model file. The blank symbol index is configurable and
/// defaults to 0, the common wav2vec2 convention (blank shared with <c>&lt;pad&gt;</c>).
/// </remarks>
public static class CtcDecoder
{
    // ----------------------------------------------------------------------------------------
    // Greedy (best-path) decoding
    // ----------------------------------------------------------------------------------------

    /// <summary>
    /// Greedy (best-path) decode: take the argmax token per frame, collapse consecutive duplicates,
    /// then drop the blank symbol. Returns the resulting sequence of token indices.
    /// </summary>
    public static int[] GreedyDecode(Tensor<float> emissions, int blank = 0)
    {
        (int frames, int vocab) = Shape2D(emissions);
        return GreedyDecode(emissions.Buffer.Span, frames, vocab, blank);
    }

    /// <summary>Greedy decode then map to text via <paramref name="vocabulary"/>.</summary>
    public static string GreedyDecode(Tensor<float> emissions, CtcVocabulary vocabulary, int blank = 0)
    {
        if (vocabulary is null) throw new ArgumentNullException(nameof(vocabulary));
        return vocabulary.Decode(GreedyDecode(emissions, blank));
    }

    /// <summary>
    /// Greedy (best-path) decode over a row-major <c>[frames, vocab]</c> emission buffer:
    /// argmax per frame, collapse consecutive duplicates, then remove the blank symbol.
    /// </summary>
    public static int[] GreedyDecode(ReadOnlySpan<float> emissions, int frames, int vocab, int blank = 0)
    {
        ValidateDims(emissions.Length, frames, vocab, blank);

        var result = new List<int>();
        int prev = -1;
        for (int t = 0; t < frames; t++)
        {
            int off = t * vocab;
            int arg = 0;
            float best = emissions[off];
            for (int v = 1; v < vocab; v++)
            {
                float x = emissions[off + v];
                if (x > best) { best = x; arg = v; }
            }

            // Collapse repeats, then drop blank: emit only a new (non-repeat) non-blank symbol.
            if (arg != blank && arg != prev) result.Add(arg);
            prev = arg;
        }
        return result.ToArray();
    }

    // ----------------------------------------------------------------------------------------
    // Prefix beam search decoding
    // ----------------------------------------------------------------------------------------

    /// <summary>
    /// CTC prefix beam search. Approximates the most probable LABELING (summing over all paths that
    /// collapse to it) rather than the most probable single path. Returns the top <paramref name="topN"/>
    /// hypotheses ordered by descending log-probability (element 0 is the best).
    /// </summary>
    /// <param name="emissions">Emission tensor of shape <c>[T, V]</c>.</param>
    /// <param name="beamWidth">Maximum number of prefixes retained after each frame.</param>
    /// <param name="blank">The blank-symbol index (default 0).</param>
    /// <param name="inputKind">How <paramref name="emissions"/> values are interpreted.</param>
    /// <param name="topN">How many hypotheses to return (clamped to the surviving beam size).</param>
    public static IReadOnlyList<CtcHypothesis> BeamSearchDecode(
        Tensor<float> emissions,
        int beamWidth = 100,
        int blank = 0,
        CtcInputKind inputKind = CtcInputKind.Logits,
        int topN = 1)
    {
        (int frames, int vocab) = Shape2D(emissions);
        return BeamSearchDecode(emissions.Buffer.Span, frames, vocab, beamWidth, blank, inputKind, topN);
    }

    /// <summary>Beam-search decode then map the top hypothesis to text via <paramref name="vocabulary"/>.</summary>
    public static string BeamSearchDecode(
        Tensor<float> emissions,
        CtcVocabulary vocabulary,
        int beamWidth = 100,
        int blank = 0,
        CtcInputKind inputKind = CtcInputKind.Logits)
    {
        if (vocabulary is null) throw new ArgumentNullException(nameof(vocabulary));
        IReadOnlyList<CtcHypothesis> best = BeamSearchDecode(emissions, beamWidth, blank, inputKind, topN: 1);
        return best.Count == 0 ? string.Empty : vocabulary.Decode(best[0].Tokens);
    }

    /// <summary>
    /// CTC prefix beam search over a row-major <c>[frames, vocab]</c> emission buffer.
    /// Maintains, per prefix, separate log-probabilities for paths ending in a blank
    /// (<c>p_blank</c>) versus a non-blank (<c>p_non_blank</c>), and merges contributions in log-space.
    /// </summary>
    public static IReadOnlyList<CtcHypothesis> BeamSearchDecode(
        ReadOnlySpan<float> emissions,
        int frames,
        int vocab,
        int beamWidth = 100,
        int blank = 0,
        CtcInputKind inputKind = CtcInputKind.Logits,
        int topN = 1)
    {
        ValidateDims(emissions.Length, frames, vocab, blank);
        if (beamWidth <= 0) throw new ArgumentOutOfRangeException(nameof(beamWidth), beamWidth, "Beam width must be positive.");
        if (topN <= 0) throw new ArgumentOutOfRangeException(nameof(topN), topN, "topN must be positive.");

        // The beam: prefix → (log p_blank, log p_non_blank). Seed with the empty prefix, which ends
        // in "blank" with probability 1 (log 0) and non-blank with probability 0 (log -inf).
        var emptyPrefix = new Prefix(Array.Empty<int>());
        var beam = new List<KeyValuePair<Prefix, PrefixState>>
        {
            new(emptyPrefix, new PrefixState { LogPBlank = 0.0, LogPNonBlank = double.NegativeInfinity }),
        };

        var logp = new double[vocab];
        for (int t = 0; t < frames; t++)
        {
            FrameLogProbs(emissions, t, vocab, inputKind, logp);

            var next = new Dictionary<Prefix, PrefixState>();

            foreach (KeyValuePair<Prefix, PrefixState> kv in beam)
            {
                Prefix prefix = kv.Key;
                PrefixState st = kv.Value;
                int[] tokens = prefix.Tokens;
                int last = tokens.Length > 0 ? tokens[^1] : -1;
                double pTotal = LogSumExp(st.LogPBlank, st.LogPNonBlank);

                // (1) Extend by blank: labeling unchanged, new path ends in blank.
                PrefixState same = GetOrAdd(next, prefix);
                same.LogPBlank = LogSumExp(same.LogPBlank, logp[blank] + pTotal);

                // (2) Repeat the last symbol with no blank between: labeling unchanged (collapses),
                //     and only the previously-non-blank mass can extend this way.
                if (last >= 0)
                    same.LogPNonBlank = LogSumExp(same.LogPNonBlank, logp[last] + st.LogPNonBlank);

                // (3) Extend by each non-blank symbol c, producing the prefix (prefix + c).
                for (int c = 0; c < vocab; c++)
                {
                    if (c == blank) continue;

                    var newTokens = new int[tokens.Length + 1];
                    Array.Copy(tokens, newTokens, tokens.Length);
                    newTokens[^1] = c;
                    PrefixState ext = GetOrAdd(next, new Prefix(newTokens));

                    if (c == last)
                        // Same symbol as the prefix's tail: a real new symbol requires a blank
                        // separator, so only the blank-ending mass may extend here.
                        ext.LogPNonBlank = LogSumExp(ext.LogPNonBlank, logp[c] + st.LogPBlank);
                    else
                        ext.LogPNonBlank = LogSumExp(ext.LogPNonBlank, logp[c] + pTotal);
                }
            }

            // Prune to the beam width by total prefix probability.
            beam = next.Count <= beamWidth
                ? next.ToList()
                : next.OrderByDescending(kv => kv.Value.LogPTotal).Take(beamWidth).ToList();
        }

        List<KeyValuePair<Prefix, PrefixState>> ranked =
            beam.OrderByDescending(kv => kv.Value.LogPTotal).ToList();

        int take = Math.Min(topN, ranked.Count);
        var output = new List<CtcHypothesis>(take);
        for (int i = 0; i < take; i++)
            output.Add(new CtcHypothesis(ranked[i].Key.Tokens, (float)ranked[i].Value.LogPTotal));
        return output;
    }

    // ----------------------------------------------------------------------------------------
    // Internals
    // ----------------------------------------------------------------------------------------

    private static PrefixState GetOrAdd(Dictionary<Prefix, PrefixState> map, Prefix key)
    {
        if (!map.TryGetValue(key, out PrefixState? state))
        {
            state = new PrefixState();
            map[key] = state;
        }
        return state;
    }

    /// <summary>Fills <paramref name="dest"/> with the natural-log probabilities for frame <paramref name="t"/>.</summary>
    private static void FrameLogProbs(ReadOnlySpan<float> emissions, int t, int vocab, CtcInputKind kind, double[] dest)
    {
        int off = t * vocab;
        switch (kind)
        {
            case CtcInputKind.LogProbabilities:
                for (int v = 0; v < vocab; v++) dest[v] = emissions[off + v];
                break;

            case CtcInputKind.Probabilities:
                for (int v = 0; v < vocab; v++) dest[v] = Math.Log(Math.Max(emissions[off + v], 1e-30));
                break;

            case CtcInputKind.Logits:
            default:
                double max = double.NegativeInfinity;
                for (int v = 0; v < vocab; v++) max = Math.Max(max, emissions[off + v]);
                double sum = 0.0;
                for (int v = 0; v < vocab; v++) sum += Math.Exp(emissions[off + v] - max);
                double logZ = max + Math.Log(sum);
                for (int v = 0; v < vocab; v++) dest[v] = emissions[off + v] - logZ;
                break;
        }
    }

    /// <summary>Numerically stable <c>log(exp(a) + exp(b))</c>.</summary>
    private static double LogSumExp(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;
        double max = a > b ? a : b;
        return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
    }

    private static (int frames, int vocab) Shape2D(Tensor<float> emissions)
    {
        if (emissions is null) throw new ArgumentNullException(nameof(emissions));
        if (emissions.Shape.Rank != 2)
            throw new ArgumentException(
                $"Emissions must be rank-2 [frames, vocab]; got shape {emissions.Shape}.", nameof(emissions));
        return (emissions.Shape[0], emissions.Shape[1]);
    }

    private static void ValidateDims(long length, int frames, int vocab, int blank)
    {
        if (frames < 0) throw new ArgumentOutOfRangeException(nameof(frames), frames, "frames must be non-negative.");
        if (vocab <= 0) throw new ArgumentOutOfRangeException(nameof(vocab), vocab, "vocab must be positive.");
        if ((long)frames * vocab != length)
            throw new ArgumentException($"Buffer length {length} does not match frames*vocab = {(long)frames * vocab}.");
        if (blank < 0 || blank >= vocab)
            throw new ArgumentOutOfRangeException(nameof(blank), blank, $"Blank index must be in [0, {vocab}).");
    }

    /// <summary>An immutable token-index sequence usable as a dictionary key (structural equality + cached hash).</summary>
    private sealed class Prefix : IEquatable<Prefix>
    {
        public readonly int[] Tokens;
        private readonly int _hash;

        public Prefix(int[] tokens)
        {
            Tokens = tokens;
            var hc = new HashCode();
            foreach (int t in tokens) hc.Add(t);
            _hash = hc.ToHashCode();
        }

        public bool Equals(Prefix? other) =>
            other is not null && (ReferenceEquals(this, other) || Tokens.AsSpan().SequenceEqual(other.Tokens));

        public override bool Equals(object? obj) => obj is Prefix p && Equals(p);

        public override int GetHashCode() => _hash;
    }

    /// <summary>Per-prefix log-probabilities split by whether the path ends in a blank or a non-blank.</summary>
    private sealed class PrefixState
    {
        public double LogPBlank = double.NegativeInfinity;
        public double LogPNonBlank = double.NegativeInfinity;

        public double LogPTotal => LogSumExp(LogPBlank, LogPNonBlank);
    }
}
