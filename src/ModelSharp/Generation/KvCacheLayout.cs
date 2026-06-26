using System;
using System.Collections.Generic;
using ModelSharp.Engine;

namespace ModelSharp.Generation;

/// <summary>A paired past-KV input and its matching present-KV output for one cache slot (layer/branch).</summary>
internal readonly struct KvCachePair
{
    /// <summary>Engine input name carrying the past cache (e.g. <c>past_key_values.0.key</c>).</summary>
    public string PastInput { get; }

    /// <summary>Engine output name producing the updated cache (e.g. <c>present.0.key</c>).</summary>
    public string PresentOutput { get; }

    /// <summary>Static description of the past input (used to size the empty first-pass tensor).</summary>
    public TensorInfo Info { get; }

    public KvCachePair(string pastInput, string presentOutput, TensorInfo info)
    {
        PastInput = pastInput;
        PresentOutput = presentOutput;
        Info = info;
    }
}

/// <summary>
/// Result of inspecting an engine's bindings to decide whether it follows the decoder-with-past
/// (KV cache) convention, and to enumerate the past-&gt;present pairings that must be threaded.
/// Cache mode is active when at least one input name starts with the configured past prefix and a
/// correspondingly named present output exists.
/// </summary>
internal sealed class KvCacheLayout
{
    /// <summary>True when the engine exposes past-KV inputs paired with present-KV outputs.</summary>
    public bool IsCacheMode => Pairs.Count > 0;

    /// <summary>The past-&gt;present pairings, in the order the engine declared its inputs.</summary>
    public IReadOnlyList<KvCachePair> Pairs { get; }

    /// <summary>Axis along which the cached sequence grows (from <see cref="DecoderModelOptions.KvSequenceAxis"/>).</summary>
    public int SequenceAxis { get; }

    private KvCacheLayout(IReadOnlyList<KvCachePair> pairs, int sequenceAxis)
    {
        Pairs = pairs;
        SequenceAxis = sequenceAxis;
    }

    /// <summary>Detects the cache layout by matching past inputs to present outputs by name.</summary>
    public static KvCacheLayout Detect(IExecutionEngine engine, DecoderModelOptions options)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (options is null) throw new ArgumentNullException(nameof(options));

        var outputs = new HashSet<string>(StringComparer.Ordinal);
        foreach (TensorInfo o in engine.Outputs) outputs.Add(o.Name);

        string pastDot = options.PastKeyValuesPrefix + ".";
        var pairs = new List<KvCachePair>();
        foreach (TensorInfo input in engine.Inputs)
        {
            if (!input.Name.StartsWith(pastDot, StringComparison.Ordinal)) continue;

            // present name = past name with the prefix swapped, preserving the rest of the suffix
            // (handles both "...0.key" and "...0.decoder.key").
            string presentName = options.PresentPrefix + input.Name.Substring(options.PastKeyValuesPrefix.Length);
            if (outputs.Contains(presentName))
                pairs.Add(new KvCachePair(input.Name, presentName, input));
        }
        return new KvCacheLayout(pairs, options.KvSequenceAxis);
    }
}
