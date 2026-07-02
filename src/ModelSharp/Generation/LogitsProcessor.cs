using System;
using System.Collections.Generic;

namespace ModelSharp.Generation;

/// <summary>
/// Pure, stateless logit transforms and samplers operating on a single logits row (the
/// vocabulary distribution for the last sequence position). Every method is deterministic
/// given its inputs — sampling draws randomness exclusively from the injected
/// <see cref="Random"/>, so seeding it makes generation reproducible and unit-testable.
/// All warpers mutate the supplied span in place; masked-out tokens are set to
/// <see cref="float.NegativeInfinity"/> so a subsequent softmax assigns them zero probability.
/// </summary>
public static class LogitsProcessor
{
    /// <summary>
    /// Applies a repetition penalty to every token that appears in <paramref name="contextIds"/>
    /// (HF convention: a positive logit is divided by the penalty, a negative logit is multiplied).
    /// No-op when <paramref name="penalty"/> is exactly 1.
    /// </summary>
    public static void ApplyRepetitionPenalty(Span<float> logits, IReadOnlyList<long> contextIds, float penalty)
    {
        if (contextIds is null) throw new ArgumentNullException(nameof(contextIds));
        if (penalty == 1f) return;
        if (penalty <= 0f) throw new ArgumentOutOfRangeException(nameof(penalty), penalty, "Must be > 0.");

        for (int i = 0; i < contextIds.Count; i++)
        {
            long id = contextIds[i];
            if (id < 0 || id >= logits.Length) continue;
            float v = logits[(int)id];
            logits[(int)id] = v > 0f ? v / penalty : v * penalty;
        }
    }

    /// <summary>Divides every logit by <paramref name="temperature"/>. No-op when temperature is 1.</summary>
    public static void ApplyTemperature(Span<float> logits, float temperature)
    {
        if (temperature == 1f) return;
        if (temperature <= 0f) throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Must be > 0.");
        for (int i = 0; i < logits.Length; i++) logits[i] /= temperature;
    }

    /// <summary>
    /// Top-K filtering: keeps exactly the <paramref name="topK"/> highest logits and masks the rest.
    /// No-op when <paramref name="topK"/> is &lt;= 0 or &gt;= the vocabulary size.
    /// </summary>
    /// <remarks>
    /// Runs a partial selection (quickselect) rather than a full <c>O(V·logV)</c> sort of the whole
    /// vocabulary: it partitions so the <paramref name="topK"/> survivors (ranked value-descending,
    /// index-ascending on ties — the same ordering the full sort produced) are identified in
    /// <c>O(V)</c> expected time, then masks everything else. The surviving <b>set</b> is identical to
    /// the previous full-sort implementation, so the tokens kept/masked are unchanged.
    /// </remarks>
    public static void ApplyTopK(Span<float> logits, int topK)
    {
        int n = logits.Length;
        if (topK <= 0 || topK >= n) return;

        // Indices 0..n-1 partitioned so the first topK are the highest logits under the
        // (value desc, index asc) ordering; the remainder are masked.
        int[] idx = BuildIndices(n);
        PartialSelectTop(logits, idx, topK);
        for (int rank = topK; rank < n; rank++) logits[idx[rank]] = float.NegativeInfinity;
    }

    /// <summary>
    /// Top-P (nucleus) filtering: keeps the smallest set of highest-probability tokens whose cumulative
    /// probability reaches <paramref name="topP"/> and masks the rest (always keeps at least one token).
    /// No-op when <paramref name="topP"/> &gt;= 1.
    /// </summary>
    /// <remarks>
    /// The nucleus is at most the highest-probability prefix, but its size is not known in advance, so we
    /// can't quickselect a fixed K. Instead of sorting the whole vocabulary we grow the candidate prefix
    /// in doubling chunks: quickselect the top <c>m</c> probabilities, sort just those <c>m</c>, and walk
    /// their cumulative sum; if the nucleus fits inside the first <c>m</c> we stop, otherwise we double
    /// <c>m</c> and retry. Because probabilities are dominated by their largest entries the nucleus is
    /// tiny in practice (a handful of tokens), so this touches <c>O(V)</c> for the selects plus
    /// <c>O(m·log m)</c> for a small <c>m</c> — versus the old full <c>O(V·log V)</c> sort. The cumulative
    /// walk visits candidates in the exact (probability desc, index asc) order the old sort used, so the
    /// keep/mask decision — including the floating-point threshold-crossing point — is unchanged.
    /// </remarks>
    public static void ApplyTopP(Span<float> logits, float topP)
    {
        if (topP >= 1f) return;
        if (topP <= 0f) throw new ArgumentOutOfRangeException(nameof(topP), topP, "Must be in (0, 1].");

        int n = logits.Length;
        float[] probs = Softmax(logits);
        int[] idx = BuildIndices(n);

        // Grow the examined prefix in doubling chunks until the nucleus threshold is crossed within it
        // (or we have examined the whole vocabulary). Only the prefix is ever sorted.
        int m = Math.Min(n, 16);
        int keep = n;
        while (true)
        {
            PartialSelectTop(probs, idx, m);
            SortPrefixDescending(probs, idx, m);

            float cumulative = 0f;
            bool crossed = false;
            for (int rank = 0; rank < m; rank++)
            {
                cumulative += probs[idx[rank]];
                if (cumulative >= topP) { keep = rank + 1; crossed = true; break; }
            }
            if (crossed || m >= n) break;
            m = Math.Min(n, m * 2);
        }

        for (int rank = keep; rank < n; rank++) logits[idx[rank]] = float.NegativeInfinity;
    }

    /// <summary>Returns the index of the maximum logit (ties resolved to the lowest index).</summary>
    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        if (logits.Length == 0) throw new ArgumentException("Logits row is empty.", nameof(logits));
        int best = 0;
        float bestValue = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestValue) { bestValue = logits[i]; best = i; }
        }
        return best;
    }

    /// <summary>
    /// Numerically stable softmax. Entries equal to <see cref="float.NegativeInfinity"/> map to 0.
    /// A row that is entirely <c>-inf</c> falls back to a uniform distribution.
    /// </summary>
    public static float[] Softmax(ReadOnlySpan<float> logits)
    {
        int n = logits.Length;
        var probs = new float[n];

        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (logits[i] > max) max = logits[i];

        if (float.IsNegativeInfinity(max))
        {
            float uniform = n == 0 ? 0f : 1f / n;
            for (int i = 0; i < n; i++) probs[i] = uniform;
            return probs;
        }

        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            double e = float.IsNegativeInfinity(logits[i]) ? 0.0 : Math.Exp(logits[i] - max);
            probs[i] = (float)e;
            sum += e;
        }
        if (sum <= 0)
        {
            float uniform = 1f / n;
            for (int i = 0; i < n; i++) probs[i] = uniform;
            return probs;
        }
        for (int i = 0; i < n; i++) probs[i] = (float)(probs[i] / sum);
        return probs;
    }

    /// <summary>
    /// Draws a single index from a (normalized) probability distribution using inverse-CDF sampling.
    /// Tokens with zero probability are never selected. Determined entirely by <paramref name="random"/>.
    /// </summary>
    public static int SampleMultinomial(ReadOnlySpan<float> probabilities, Random random)
    {
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (probabilities.Length == 0) throw new ArgumentException("Distribution is empty.", nameof(probabilities));

        double r = random.NextDouble();
        double cumulative = 0;
        int last = -1;
        for (int i = 0; i < probabilities.Length; i++)
        {
            float p = probabilities[i];
            if (p <= 0f) continue;
            last = i;
            cumulative += p;
            if (r < cumulative) return i;
        }
        // Floating-point slack: r landed past the last positive bucket.
        return last >= 0 ? last : 0;
    }

    /// <summary>
    /// Runs the full processing pipeline for one decoding step and returns the chosen token id.
    /// Order: repetition penalty, then (for sampling only) temperature, Top-K, Top-P, softmax and a
    /// multinomial draw. Greedy decoding returns the argmax straight after the repetition penalty.
    /// </summary>
    /// <param name="logits">The last-position logits row; mutated in place.</param>
    /// <param name="config">Decoding parameters.</param>
    /// <param name="contextIds">All ids seen so far (prompt + generated), for the repetition penalty.</param>
    /// <param name="random">RNG used when <see cref="GenerationConfig.DoSample"/> is true; may be null for greedy.</param>
    public static int SelectNextToken(Span<float> logits, GenerationConfig config, IReadOnlyList<long> contextIds, Random? random)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));

        ApplyRepetitionPenalty(logits, contextIds, config.RepetitionPenalty);

        if (!config.DoSample)
            return ArgMax(logits);

        ApplyTemperature(logits, config.Temperature);
        if (config.TopK > 0) ApplyTopK(logits, config.TopK);
        if (config.TopP < 1f) ApplyTopP(logits, config.TopP);

        float[] probs = Softmax(logits);
        return SampleMultinomial(probs, random ?? throw new InvalidOperationException(
            "Sampling requires a Random instance; none was supplied."));
    }

    /// <summary>
    /// Selects the next token directly from a slice of the engine's logits tensor, avoiding a
    /// per-step vocabulary-sized copy on the common greedy path.
    ///
    /// <para>The greedy path (<see cref="GenerationConfig.DoSample"/> false <b>and</b> no repetition
    /// penalty) scans <paramref name="logits"/> read-only and returns the argmax — nothing is copied or
    /// mutated. Any path that has to mutate the row (repetition penalty, temperature, Top-K/Top-P,
    /// sampling) first copies the slice into <paramref name="scratch"/> — a caller-owned buffer reused
    /// across steps — so the decode loop still allocates nothing here. The result is identical to
    /// copying the row out and calling <see cref="SelectNextToken(Span{float}, GenerationConfig,
    /// IReadOnlyList{long}, Random?)"/>.</para>
    /// </summary>
    /// <param name="logits">The full logits tensor buffer.</param>
    /// <param name="offset">Start of the last-position vocabulary row within <paramref name="logits"/>.</param>
    /// <param name="vocab">Vocabulary length (row width).</param>
    /// <param name="config">Decoding parameters.</param>
    /// <param name="contextIds">All ids seen so far (prompt + generated), for the repetition penalty.</param>
    /// <param name="random">RNG used when sampling; may be null for greedy.</param>
    /// <param name="scratch">Caller-owned reusable buffer; grown to <paramref name="vocab"/> as needed.</param>
    public static int SelectNextTokenFromLogits(
        ReadOnlySpan<float> logits, int offset, int vocab,
        GenerationConfig config, IReadOnlyList<long> contextIds, Random? random, ref float[]? scratch)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));

        ReadOnlySpan<float> row = logits.Slice(offset, vocab);

        // Greedy with no repetition penalty never mutates the row → scan it in place, zero-copy.
        if (!config.DoSample && config.RepetitionPenalty == 1f)
            return ArgMax(row);

        // Everything else mutates the row; copy once into the reused scratch buffer, then run the
        // standard in-place pipeline over it (identical result to SelectNextToken on a fresh copy).
        if (scratch is null || scratch.Length < vocab) scratch = new float[vocab];
        row.CopyTo(scratch);
        return SelectNextToken(scratch.AsSpan(0, vocab), config, contextIds, random);
    }

    /// <summary>Fresh identity index array 0..n-1.</summary>
    private static int[] BuildIndices(int n)
    {
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        return idx;
    }

    /// <summary>
    /// Total order used everywhere selection happens: <paramref name="ia"/> ranks strictly before
    /// <paramref name="ib"/> when its key is larger, or (on an exact key tie) when its index is smaller.
    /// This reproduces the old full-sort comparator <c>(value desc, index asc)</c> exactly.
    /// </summary>
    private static bool RanksBefore(ReadOnlySpan<float> keys, int ia, int ib)
    {
        float ka = keys[ia], kb = keys[ib];
        if (ka > kb) return true;
        if (ka < kb) return false;
        return ia < ib;
    }

    /// <summary>
    /// Partitions <paramref name="idx"/> in place (quickselect) so that its first <paramref name="k"/>
    /// entries are exactly the <c>k</c> highest-ranked elements under <see cref="RanksBefore"/> — the same
    /// set the old full sort placed in ranks <c>[0, k)</c>. The first <c>k</c> entries are not fully
    /// sorted among themselves; callers that need order call <see cref="SortPrefixDescending"/> after.
    /// Expected <c>O(n)</c>.
    /// </summary>
    private static void PartialSelectTop(ReadOnlySpan<float> keys, int[] idx, int k)
    {
        if (k <= 0 || k >= idx.Length) return;
        int lo = 0, hi = idx.Length - 1;
        // Target boundary: we want the element that would sit at rank (k-1) in its final slot, which
        // guarantees idx[0..k) is the top-k set (order within each side is irrelevant here).
        int target = k - 1;
        var rng = new Random(0); // deterministic pivot choice → reproducible partitioning.
        while (lo < hi)
        {
            int pivotIndex = lo + rng.Next(hi - lo + 1);
            int p = Partition(keys, idx, lo, hi, pivotIndex);
            if (p == target) return;
            if (p < target) lo = p + 1;
            else hi = p - 1;
        }
    }

    /// <summary>
    /// Lomuto partition around the pivot value: elements ranking before the pivot (higher-ranked) move to
    /// the left, the rest to the right. Returns the pivot's final position. Uses <see cref="RanksBefore"/>
    /// so ties break by index, matching the sort semantics.
    /// </summary>
    private static int Partition(ReadOnlySpan<float> keys, int[] idx, int lo, int hi, int pivotIndex)
    {
        int pivot = idx[pivotIndex];
        (idx[pivotIndex], idx[hi]) = (idx[hi], idx[pivotIndex]);
        int store = lo;
        for (int i = lo; i < hi; i++)
        {
            if (RanksBefore(keys, idx[i], pivot))
            {
                (idx[store], idx[i]) = (idx[i], idx[store]);
                store++;
            }
        }
        (idx[store], idx[hi]) = (idx[hi], idx[store]);
        return store;
    }

    /// <summary>
    /// Sorts just <c>idx[0..count)</c> into descending rank order (<see cref="RanksBefore"/>). Used to walk
    /// a small candidate prefix in the exact order the old full sort produced. Insertion sort — <c>count</c>
    /// is tiny (nucleus size) in practice.
    /// </summary>
    private static void SortPrefixDescending(ReadOnlySpan<float> keys, int[] idx, int count)
    {
        for (int i = 1; i < count; i++)
        {
            int cur = idx[i];
            int j = i - 1;
            while (j >= 0 && RanksBefore(keys, cur, idx[j]))
            {
                idx[j + 1] = idx[j];
                j--;
            }
            idx[j + 1] = cur;
        }
    }
}
