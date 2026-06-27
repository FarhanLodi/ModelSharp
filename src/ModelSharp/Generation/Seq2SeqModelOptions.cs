namespace ModelSharp.Generation;

/// <summary>
/// Describes the input/output binding conventions of an encoder-decoder (seq2seq) model so the
/// <see cref="Seq2SeqGenerator"/> can run the encoder once, then drive the decoder autoregressively
/// without hard-coding names. Defaults match the standard Hugging Face / Optimum ONNX seq2seq export
/// (<c>encoder_model.onnx</c> + <c>decoder_model.onnx</c> / <c>decoder_with_past_model.onnx</c>, or the
/// merged <c>decoder_model_merged.onnx</c>), which is what T5 / BART / MarianMT / Whisper produce.
///
/// <para>The decoder names reuse the same conventions as <see cref="DecoderModelOptions"/>. Seq2seq adds
/// the encoder bindings, the cross-attention KV-cache naming, and the <see cref="DecoderStartTokenId"/>
/// that seeds the decoder loop.</para>
/// </summary>
public sealed record Seq2SeqModelOptions
{
    // ---- encoder ----

    /// <summary>Name of the encoder token-ids (or feature) input. Default <c>input_ids</c>.</summary>
    public string EncoderInputIdsName { get; init; } = "input_ids";

    /// <summary>
    /// Name of the encoder attention-mask input. Fed only when the encoder declares it. Default
    /// <c>attention_mask</c>. (Whisper-style audio encoders that take mel features and declare no mask
    /// simply skip this.)
    /// </summary>
    public string EncoderAttentionMaskName { get; init; } = "attention_mask";

    /// <summary>Name of the encoder hidden-states output. Default <c>last_hidden_state</c>.</summary>
    public string EncoderHiddenStatesOutputName { get; init; } = "last_hidden_state";

    // ---- decoder ----

    /// <summary>Name of the decoder token-ids input. Default <c>input_ids</c>.</summary>
    public string DecoderInputIdsName { get; init; } = "input_ids";

    /// <summary>
    /// Name of the decoder's encoder-hidden-states (cross-attention) input. Default
    /// <c>encoder_hidden_states</c>.
    /// </summary>
    public string EncoderHiddenStatesInputName { get; init; } = "encoder_hidden_states";

    /// <summary>
    /// Name of the decoder's encoder attention-mask input (used to mask cross-attention over padded
    /// source tokens). Fed only when the decoder declares it. Default <c>encoder_attention_mask</c>.
    /// </summary>
    public string EncoderAttentionMaskInputName { get; init; } = "encoder_attention_mask";

    /// <summary>
    /// Name of the decoder self-attention-mask input. Fed only when the decoder declares it. Some seq2seq
    /// decoders take an explicit decoder <c>attention_mask</c>; many rely on the causal mask alone and omit
    /// it. Default <c>decoder_attention_mask</c>.
    /// </summary>
    public string DecoderAttentionMaskName { get; init; } = "decoder_attention_mask";

    /// <summary>
    /// Name of the boolean <c>use_cache_branch</c> input found in Optimum "merged" decoder exports.
    /// Fed only when the decoder declares it: <c>false</c> on the first/prefill pass, <c>true</c> on every
    /// cached step. Default <c>use_cache_branch</c>.
    /// </summary>
    public string UseCacheBranchName { get; init; } = "use_cache_branch";

    /// <summary>Name of the logits output. Default <c>logits</c>; if absent the first non-cache output is used.</summary>
    public string LogitsOutputName { get; init; } = "logits";

    // ---- KV cache naming ----

    /// <summary>
    /// Prefix of past-KV cache <em>inputs</em> on the decoder. The full names are matched by
    /// <c>{PastKeyValuesPrefix}.&lt;layer&gt;.(decoder|encoder).key|value</c>. Default <c>past_key_values</c>.
    /// </summary>
    public string PastKeyValuesPrefix { get; init; } = "past_key_values";

    /// <summary>
    /// Prefix of present-KV cache <em>outputs</em> on the decoder. A past input is paired with the present
    /// output whose name is the input name with <see cref="PastKeyValuesPrefix"/> replaced by this prefix.
    /// Default <c>present</c>.
    /// </summary>
    public string PresentPrefix { get; init; } = "present";

    /// <summary>
    /// Infix that marks a <em>cross-attention</em> KV cache slot in the past/present names (e.g.
    /// <c>past_key_values.0.encoder.key</c>). Self-attention slots use <see cref="SelfAttentionInfix"/>.
    /// Cross-attention caches are computed once from the encoder output and stay constant across decode
    /// steps, so the generator threads the decoder's first-pass <c>present.*.encoder.*</c> outputs straight
    /// back as inputs on every later step. Default <c>encoder</c>.
    /// </summary>
    public string CrossAttentionInfix { get; init; } = "encoder";

    /// <summary>
    /// Infix that marks a <em>self-attention</em> KV cache slot (e.g. <c>past_key_values.0.decoder.key</c>),
    /// which grows by one position each decode step. Default <c>decoder</c>.
    /// </summary>
    public string SelfAttentionInfix { get; init; } = "decoder";

    /// <summary>
    /// Axis along which a self-attention cached sequence grows. Default 2, matching the canonical
    /// <c>[batch, heads, sequence, head_dim]</c> layout.
    /// </summary>
    public int KvSequenceAxis { get; init; } = 2;

    /// <summary>
    /// Number of key/value heads, used to size empty past tensors on the first pass when the engine does not
    /// report concrete KV input shapes. Ignored when shapes are known.
    /// </summary>
    public int? KvCacheNumHeads { get; init; }

    /// <summary>Per-head dimension, used together with <see cref="KvCacheNumHeads"/> to size empty past tensors.</summary>
    public int? KvCacheHeadDim { get; init; }

    // ---- decoding ----

    /// <summary>
    /// The token id the decoder starts from (HF <c>decoder_start_token_id</c>). T5 uses the pad id (0),
    /// BART/Marian use <c>&lt;/s&gt;</c> / <c>&lt;s&gt;</c> family ids, Whisper uses <c>&lt;|startoftranscript|&gt;</c>.
    /// Default 0.
    /// </summary>
    public long DecoderStartTokenId { get; init; }

    /// <summary>Batch size. Currently only <c>1</c> is supported.</summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>A shared instance using all default names and <c>decoder_start_token_id = 0</c> (the T5 default).</summary>
    public static Seq2SeqModelOptions Default { get; } = new();
}
