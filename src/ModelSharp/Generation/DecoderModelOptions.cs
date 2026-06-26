namespace ModelSharp.Generation;

/// <summary>
/// Describes the input/output binding conventions of a decoder model so the
/// <see cref="TextGenerator"/> can build feeds and thread the KV cache without hard-coding
/// names. Defaults match the standard Hugging Face / Optimum ONNX decoder-with-past export.
/// </summary>
public sealed record DecoderModelOptions
{
    /// <summary>Name of the token-ids input. Default <c>input_ids</c>.</summary>
    public string InputIdsName { get; init; } = "input_ids";

    /// <summary>Name of the attention-mask input. Fed only when the model declares it. Default <c>attention_mask</c>.</summary>
    public string AttentionMaskName { get; init; } = "attention_mask";

    /// <summary>Name of the position-ids input. Fed only when the model declares it. Default <c>position_ids</c>.</summary>
    public string PositionIdsName { get; init; } = "position_ids";

    /// <summary>
    /// Name of the boolean <c>use_cache_branch</c> input found in Optimum "merged" decoder exports,
    /// which unify the prefill (no-past) and decode (with-past) graphs into one. Fed only when the
    /// model declares it: <c>false</c> on the first/prefill pass, <c>true</c> on every cached step.
    /// Default <c>use_cache_branch</c>.
    /// </summary>
    public string UseCacheBranchName { get; init; } = "use_cache_branch";

    /// <summary>Name of the logits output. Default <c>logits</c>; if absent the first non-cache output is used.</summary>
    public string LogitsOutputName { get; init; } = "logits";

    /// <summary>
    /// Prefix of past-KV cache <em>inputs</em>. The full names are matched by
    /// <c>{PastKeyValuesPrefix}.&lt;layer&gt;[.decoder].key|value</c>. Default <c>past_key_values</c>.
    /// </summary>
    public string PastKeyValuesPrefix { get; init; } = "past_key_values";

    /// <summary>
    /// Prefix of present-KV cache <em>outputs</em>. A past input is paired with the present output
    /// whose name is the input name with <see cref="PastKeyValuesPrefix"/> replaced by this prefix
    /// (so the <c>.decoder.</c> / <c>.key</c> / <c>.value</c> suffix is preserved). Default <c>present</c>.
    /// </summary>
    public string PresentPrefix { get; init; } = "present";

    /// <summary>
    /// Axis along which the cached sequence length grows in a KV tensor. Default 2, matching the
    /// canonical <c>[batch, heads, sequence, head_dim]</c> layout.
    /// </summary>
    public int KvSequenceAxis { get; init; } = 2;

    /// <summary>
    /// Number of key/value heads, used to size the empty past tensors on the first pass when the engine
    /// does not report concrete KV input shapes. Required for such engines; ignored when shapes are known.
    /// </summary>
    public int? KvCacheNumHeads { get; init; }

    /// <summary>
    /// Per-head dimension, used together with <see cref="KvCacheNumHeads"/> to size empty past tensors
    /// when the engine does not report concrete KV input shapes.
    /// </summary>
    public int? KvCacheHeadDim { get; init; }

    /// <summary>Batch size. Currently only <c>1</c> is supported.</summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>A shared instance using all default names.</summary>
    public static DecoderModelOptions Default { get; } = new();
}
