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
    public static void ApplyTopK(Span<float> logits, int topK)
    {
        int n = logits.Length;
        if (topK <= 0 || topK >= n) return;

        // Rank indices by logit descending (index ascending breaks ties for determinism).
        float[] values = logits.ToArray();
        int[] order = BuildOrder(values);
        for (int rank = topK; rank < n; rank++) logits[order[rank]] = float.NegativeInfinity;
    }

    /// <summary>
    /// Top-P (nucleus) filtering: keeps the smallest set of highest-probability tokens whose cumulative
    /// probability reaches <paramref name="topP"/> and masks the rest (always keeps at least one token).
    /// No-op when <paramref name="topP"/> &gt;= 1.
    /// </summary>
    public static void ApplyTopP(Span<float> logits, float topP)
    {
        if (topP >= 1f) return;
        if (topP <= 0f) throw new ArgumentOutOfRangeException(nameof(topP), topP, "Must be in (0, 1].");

        int n = logits.Length;
        float[] probs = Softmax(logits);
        int[] order = BuildOrder(probs);

        float cumulative = 0f;
        int keep = n;
        for (int rank = 0; rank < n; rank++)
        {
            cumulative += probs[order[rank]];
            if (cumulative >= topP) { keep = rank + 1; break; }
        }
        for (int rank = keep; rank < n; rank++) logits[order[rank]] = float.NegativeInfinity;
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

    /// <summary>Returns indices 0..n-1 ordered by <paramref name="keys"/> descending, index ascending on ties.</summary>
    private static int[] BuildOrder(float[] keys)
    {
        int n = keys.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int c = keys[b].CompareTo(keys[a]);
            return c != 0 ? c : a.CompareTo(b);
        });
        return order;
    }
}
