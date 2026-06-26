using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using ModelSharp.Manifest;
using ModelSharp.Pipeline;
using ModelSharp.Tensors;

namespace ModelSharp.ImageSharp;

/// <summary>A single ranked classification result.</summary>
public sealed class Classification
{
    public int Index { get; init; }
    public float Score { get; init; }
    public string? Label { get; init; }

    public override string ToString() => $"{Label ?? Index.ToString()} ({Score:P1})";
}

/// <summary>
/// Decodes a classifier's output logits into a ranked top-K list, optionally applying
/// softmax and attaching labels from the manifest.
/// <para>
/// Tunables fall back to <see cref="ModelManifest.Extra"/> (parsed with invariant culture)
/// when not given explicitly to the constructor: <c>"top_k"</c>/<c>"topk"</c> (default 5)
/// and <c>"softmax"</c> (<c>true</c>|<c>false</c>|<c>1</c>|<c>0</c>, default <see langword="true"/>).
/// Explicit constructor arguments win over <c>Extra</c>, which wins over the defaults.
/// </para>
/// <para>
/// Labels are attached from <see cref="ModelManifest.Labels"/> when present (the core
/// manifest resolver folds any sidecar/metadata <c>labels</c> hint into that list).
/// </para>
/// </summary>
public sealed class ClassificationPostprocessor : IPostprocessor
{
    /// <summary>Default number of ranked results returned.</summary>
    public const int DefaultTopK = 5;

    private readonly ModelManifest _manifest;
    private readonly int _topK;
    private readonly bool _applySoftmax;

    /// <summary>
    /// Builds a classification postprocessor. Any argument left <see langword="null"/> falls back to the
    /// matching <see cref="ModelManifest.Extra"/> hint, then to the documented default.
    /// </summary>
    public ClassificationPostprocessor(ModelManifest manifest, int? topK = null, bool? applySoftmax = null)
    {
        _manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        IReadOnlyDictionary<string, string> extra = manifest.Extra;
        _topK = topK ?? ParseInt(extra, "top_k", "topk", DefaultTopK);
        if (_topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), _topK, "top_k must be positive.");
        _applySoftmax = applySoftmax ?? ParseBool(extra, "softmax", defaultValue: true);
    }

    /// <inheritdoc />
    public object Decode(IReadOnlyDictionary<string, NamedTensor> outputs)
    {
        if (outputs is null || outputs.Count == 0)
            throw new ArgumentException("Classification expects at least one output tensor.", nameof(outputs));

        NamedTensor first = outputs.Values.First();
        long batch = LeadingBatch(first.Tensor.Shape);
        if (batch > 1)
            throw new NotSupportedException(
                $"ClassificationPostprocessor decodes a single example; got a batch of {batch} (shape {first.Tensor.Shape}).");

        float[] logits = first.Data.Span.ToArray();
        float[] scores = _applySoftmax ? Softmax(logits) : logits;
        IReadOnlyList<string>? labels = _manifest.Labels;

        return scores
            .Select((s, i) => new Classification
            {
                Index = i,
                Score = s,
                Label = labels is not null && i < labels.Count ? labels[i] : null,
            })
            .OrderByDescending(c => c.Score)
            .Take(_topK)
            .ToList();
    }

    /// <summary>Returns the leading batch dim for a rank ≥ 2 tensor; 1 for a bare [N] logits vector.</summary>
    private static long LeadingBatch(TensorShape shape) => shape.Rank >= 2 ? shape[0] : 1;

    private static float[] Softmax(float[] x)
    {
        float max = x.Max();
        double sum = 0;
        var e = new float[x.Length];
        for (int i = 0; i < x.Length; i++) { e[i] = MathF.Exp(x[i] - max); sum += e[i]; }
        for (int i = 0; i < x.Length; i++) e[i] = (float)(e[i] / sum);
        return e;
    }

    private static int ParseInt(IReadOnlyDictionary<string, string> extra, string key, string altKey, int fallback)
    {
        if ((extra.TryGetValue(key, out string? s) || extra.TryGetValue(altKey, out s))
            && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
            return v;
        return fallback;
    }

    private static bool ParseBool(IReadOnlyDictionary<string, string> extra, string key, bool defaultValue)
    {
        if (!extra.TryGetValue(key, out string? s) || string.IsNullOrWhiteSpace(s))
            return defaultValue;
        switch (s.Trim().ToLowerInvariant())
        {
            case "true":
            case "1":
            case "yes":
                return true;
            case "false":
            case "0":
            case "no":
                return false;
            default:
                return defaultValue;
        }
    }
}
