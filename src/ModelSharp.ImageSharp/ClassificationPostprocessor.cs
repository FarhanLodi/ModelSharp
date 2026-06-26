using System;
using System.Collections.Generic;
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
/// </summary>
public sealed class ClassificationPostprocessor : IPostprocessor
{
    private readonly ModelManifest _manifest;
    private readonly int _topK;
    private readonly bool _applySoftmax;

    public ClassificationPostprocessor(ModelManifest manifest, int topK = 5, bool applySoftmax = true)
    {
        _manifest = manifest;
        _topK = topK;
        _applySoftmax = applySoftmax;
    }

    /// <inheritdoc />
    public object Decode(IReadOnlyDictionary<string, NamedTensor> outputs)
    {
        NamedTensor first = outputs.Values.First();
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

    private static float[] Softmax(float[] x)
    {
        float max = x.Max();
        double sum = 0;
        var e = new float[x.Length];
        for (int i = 0; i < x.Length; i++) { e[i] = MathF.Exp(x[i] - max); sum += e[i]; }
        for (int i = 0; i < x.Length; i++) e[i] = (float)(e[i] / sum);
        return e;
    }
}
