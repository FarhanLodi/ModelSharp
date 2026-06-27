using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Engine;
using ModelSharp.Tensors;

namespace ModelSharp.Generation;

/// <summary>
/// Engine-agnostic encoder-decoder (seq2seq) generation loop for models such as T5, BART, MarianMT and
/// Whisper. It runs the <b>encoder once</b> to produce <c>encoder_hidden_states</c>, then drives the
/// <b>decoder autoregressively</b> — feeding decoder input ids, the cached encoder hidden states (cross
/// attention), the encoder attention mask, the growing decoder self-attention KV cache and the constant
/// cross-attention KV cache — starting from <see cref="Seq2SeqModelOptions.DecoderStartTokenId"/> until an
/// EOS token or a length bound is hit.
///
/// <para>Execution modes (auto-detected from the decoder engine's bindings):</para>
/// <list type="bullet">
///   <item><description><b>KV cache</b> (decoder-with-past / merged): the decoder is fed one token at a
///   time. Self-attention <c>present.*.decoder.*</c> outputs are threaded back as <c>past_key_values.*</c>
///   and grow each step; cross-attention <c>present.*.encoder.*</c> outputs are computed once on the first
///   pass and re-fed unchanged thereafter.</description></item>
///   <item><description><b>No-cache</b> fallback: the whole growing decoder sequence is re-fed every step
///   alongside the encoder hidden states.</description></item>
/// </list>
///
/// <para>The type works purely in token-id space (decoding to text happens elsewhere) and is engine
/// agnostic: the same code drives <c>ManagedCpuEngine</c> or the GPU engine via
/// <see cref="IExecutionEngine"/>. Two engines are supplied — one for the encoder, one for the decoder —
/// which may be the same instance for a single-graph export.</para>
/// </summary>
public sealed class Seq2SeqGenerator
{
    private readonly IExecutionEngine _encoder;
    private readonly IExecutionEngine _decoder;
    private readonly Seq2SeqModelOptions _options;

    private readonly TensorInfo? _encInputInfo;
    private readonly TensorInfo? _encMaskInfo;

    private readonly TensorInfo? _decInputInfo;
    private readonly TensorInfo? _encHiddenInputInfo;
    private readonly TensorInfo? _encMaskInputInfo;
    private readonly TensorInfo? _decMaskInfo;
    private readonly TensorInfo? _useCacheBranchInfo;

    private readonly IReadOnlyList<Seq2SeqKvPair> _selfPairs;
    private readonly IReadOnlyList<Seq2SeqKvPair> _crossPairs;

    /// <summary>Creates a generator over the supplied encoder and decoder engines.</summary>
    /// <param name="encoderEngine">Engine for the encoder graph.</param>
    /// <param name="decoderEngine">Engine for the decoder graph (may equal <paramref name="encoderEngine"/> for a merged single-graph export).</param>
    /// <param name="options">IO binding conventions; defaults to the HF/Optimum seq2seq layout.</param>
    public Seq2SeqGenerator(IExecutionEngine encoderEngine, IExecutionEngine decoderEngine, Seq2SeqModelOptions? options = null)
    {
        _encoder = encoderEngine ?? throw new ArgumentNullException(nameof(encoderEngine));
        _decoder = decoderEngine ?? throw new ArgumentNullException(nameof(decoderEngine));
        _options = options ?? Seq2SeqModelOptions.Default;
        if (_options.BatchSize != 1)
            throw new NotSupportedException("Seq2SeqGenerator currently supports BatchSize == 1.");

        _encInputInfo = FindInput(_encoder, _options.EncoderInputIdsName);
        _encMaskInfo = FindInput(_encoder, _options.EncoderAttentionMaskName);
        if (_encInputInfo is null)
            throw new ModelSharpException(
                $"Encoder does not declare the input '{_options.EncoderInputIdsName}'. " +
                "Set Seq2SeqModelOptions.EncoderInputIdsName to match the model.");

        _decInputInfo = FindInput(_decoder, _options.DecoderInputIdsName);
        _encHiddenInputInfo = FindInput(_decoder, _options.EncoderHiddenStatesInputName);
        _encMaskInputInfo = FindInput(_decoder, _options.EncoderAttentionMaskInputName);
        _decMaskInfo = FindInput(_decoder, _options.DecoderAttentionMaskName);
        _useCacheBranchInfo = FindInput(_decoder, _options.UseCacheBranchName);
        if (_decInputInfo is null)
            throw new ModelSharpException(
                $"Decoder does not declare the token-ids input '{_options.DecoderInputIdsName}'. " +
                "Set Seq2SeqModelOptions.DecoderInputIdsName to match the model.");

        (_selfPairs, _crossPairs) = DetectKvPairs();
    }

    /// <summary>True when the decoder follows the decoder-with-past (KV cache) convention.</summary>
    public bool UsesKvCache => _selfPairs.Count > 0 || _crossPairs.Count > 0;

    /// <summary>Whether the decoder declares a cross-attention KV cache that is computed once and reused.</summary>
    public bool HasCrossAttentionCache => _crossPairs.Count > 0;

    /// <summary>Whether the decoder declares an explicit decoder self-attention-mask input.</summary>
    public bool FeedsDecoderAttentionMask => _decMaskInfo is not null;

    /// <summary>Whether the decoder declares a <c>use_cache_branch</c> input (merged export).</summary>
    public bool FeedsUseCacheBranch => _useCacheBranchInfo is not null;

    /// <summary>
    /// Generates target token ids for the given source ids and returns the result eagerly. The returned list
    /// does not include the decoder start token.
    /// </summary>
    /// <param name="sourceTokenIds">The encoder input as token ids (or feature ids); must be non-empty.</param>
    /// <param name="config">Decoding parameters.</param>
    /// <param name="sourceAttentionMask">Optional per-token encoder mask (1 = attend, 0 = pad); defaults to all-ones.</param>
    public IReadOnlyList<long> Generate(
        IReadOnlyList<long> sourceTokenIds, GenerationConfig config, IReadOnlyList<long>? sourceAttentionMask = null)
    {
        var result = new List<long>();
        foreach (long token in Iterate(sourceTokenIds, config, sourceAttentionMask)) result.Add(token);
        return result;
    }

    /// <summary>Lazily streams generated target token ids one at a time.</summary>
    public IEnumerable<long> GenerateStream(
        IReadOnlyList<long> sourceTokenIds, GenerationConfig config, IReadOnlyList<long>? sourceAttentionMask = null)
        => Iterate(sourceTokenIds, config, sourceAttentionMask);

    /// <summary>Generates target token ids and invokes <paramref name="onToken"/> for each produced token.</summary>
    public void Generate(
        IReadOnlyList<long> sourceTokenIds, GenerationConfig config, Action<long> onToken,
        IReadOnlyList<long>? sourceAttentionMask = null)
    {
        if (onToken is null) throw new ArgumentNullException(nameof(onToken));
        foreach (long token in Iterate(sourceTokenIds, config, sourceAttentionMask)) onToken(token);
    }

    // ---- core loop ----

    private IEnumerable<long> Iterate(
        IReadOnlyList<long> source, GenerationConfig config, IReadOnlyList<long>? sourceMask)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (config is null) throw new ArgumentNullException(nameof(config));
        config.Validate();
        if (source.Count == 0)
            throw new ArgumentException("Source sequence must contain at least one token.", nameof(source));
        if (sourceMask is not null && sourceMask.Count != source.Count)
            throw new ArgumentException("sourceAttentionMask length must match the source length.", nameof(sourceMask));

        Random? rng = config.DoSample
            ? (config.Seed.HasValue ? new Random(config.Seed.Value) : new Random())
            : null;

        // 1) Encoder runs exactly once. Capture the hidden states (and reuse the source mask for cross-attn).
        NamedTensor encoderHidden = RunEncoder(source, sourceMask);
        long[] encMask = BuildEncMaskValues(source.Count, sourceMask);

        // 2) Decoder autoregressive loop, seeded by the decoder start token (which is never emitted).
        bool cache = UsesKvCache;
        var decoderSequence = new List<long> { _options.DecoderStartTokenId };
        int generatedCount = 0;

        Dictionary<string, Tensor>? selfPast = null;   // self-attn KV, grows each step
        Dictionary<string, Tensor>? crossPast = null;  // cross-attn KV, captured once then constant

        while (true)
        {
            if (generatedCount >= config.MaxNewTokens) yield break;
            if (config.MaxLength > 0 && decoderSequence.Count - 1 >= config.MaxLength) yield break;

            // Cache mode: feed whole decoder prefix once (selfPast == null), then one token at a time.
            int chunkLength = (!cache || selfPast is null) ? decoderSequence.Count : 1;
            int startPosition = decoderSequence.Count - chunkLength;

            IReadOnlyDictionary<string, NamedTensor> feeds =
                BuildDecoderFeeds(decoderSequence, chunkLength, startPosition, encoderHidden, encMask, cache, selfPast, crossPast);
            IReadOnlyDictionary<string, NamedTensor> outputs = _decoder.Run(feeds);

            float[] logitsRow = ExtractLastPositionLogits(outputs);
            int nextId = LogitsProcessor.SelectNextToken(logitsRow, config, decoderSequence, rng);
            long next = nextId;

            decoderSequence.Add(next);
            generatedCount++;

            if (cache)
            {
                selfPast = CapturePresent(outputs, _selfPairs);
                crossPast ??= _crossPairs.Count > 0 ? CapturePresent(outputs, _crossPairs) : null;
            }

            yield return next;

            if (IsEos(next, config)) yield break;
        }
    }

    /// <summary>Runs the encoder once and returns its hidden-states output bound to the decoder's input name.</summary>
    private NamedTensor RunEncoder(IReadOnlyList<long> source, IReadOnlyList<long>? sourceMask)
    {
        var feeds = new Dictionary<string, NamedTensor>(StringComparer.Ordinal);

        var ids = new long[source.Count];
        for (int i = 0; i < source.Count; i++) ids[i] = source[i];
        feeds[_options.EncoderInputIdsName] = new NamedTensor(
            _options.EncoderInputIdsName, BuildIntTensor(_encInputInfo, new TensorShape(1, source.Count), ids));

        if (_encMaskInfo is not null)
        {
            long[] mask = BuildEncMaskValues(source.Count, sourceMask);
            feeds[_options.EncoderAttentionMaskName] = new NamedTensor(
                _options.EncoderAttentionMaskName, BuildIntTensor(_encMaskInfo, new TensorShape(1, source.Count), mask));
        }

        IReadOnlyDictionary<string, NamedTensor> outputs = _encoder.Run(feeds);

        if (!outputs.TryGetValue(_options.EncoderHiddenStatesOutputName, out NamedTensor? hidden))
            hidden = outputs.Values.FirstOrDefault();
        if (hidden is null)
            throw new ModelSharpException(
                $"Encoder produced no '{_options.EncoderHiddenStatesOutputName}' output to feed the decoder.");

        // Rebind under the decoder's encoder-hidden-states input name for the loop.
        return new NamedTensor(_options.EncoderHiddenStatesInputName, hidden.Tensor);
    }

    private static long[] BuildEncMaskValues(int length, IReadOnlyList<long>? sourceMask)
    {
        var mask = new long[length];
        if (sourceMask is null)
            for (int i = 0; i < length; i++) mask[i] = 1;
        else
            for (int i = 0; i < length; i++) mask[i] = sourceMask[i];
        return mask;
    }

    /// <summary>Builds the decoder feed dictionary for one step.</summary>
    private IReadOnlyDictionary<string, NamedTensor> BuildDecoderFeeds(
        List<long> sequence, int chunkLength, int startPosition,
        NamedTensor encoderHidden, long[] encMask,
        bool cache, Dictionary<string, Tensor>? selfPast, Dictionary<string, Tensor>? crossPast)
    {
        var feeds = new Dictionary<string, NamedTensor>(StringComparer.Ordinal);

        // decoder input_ids: the chunk processed this step.
        var ids = new long[chunkLength];
        for (int i = 0; i < chunkLength; i++) ids[i] = sequence[startPosition + i];
        feeds[_options.DecoderInputIdsName] = new NamedTensor(
            _options.DecoderInputIdsName, BuildIntTensor(_decInputInfo, new TensorShape(1, chunkLength), ids));

        // encoder_hidden_states (cross attention): same tensor every step.
        if (_encHiddenInputInfo is not null)
            feeds[_options.EncoderHiddenStatesInputName] = encoderHidden;

        // encoder_attention_mask: masks cross-attention over padded source tokens.
        if (_encMaskInputInfo is not null)
            feeds[_options.EncoderAttentionMaskInputName] = new NamedTensor(
                _options.EncoderAttentionMaskInputName,
                BuildIntTensor(_encMaskInputInfo, new TensorShape(1, encMask.Length), encMask));

        // decoder_attention_mask: ones over every decoder token attended to so far.
        if (_decMaskInfo is not null)
        {
            var mask = new long[sequence.Count];
            for (int i = 0; i < sequence.Count; i++) mask[i] = 1;
            feeds[_options.DecoderAttentionMaskName] = new NamedTensor(
                _options.DecoderAttentionMaskName, BuildIntTensor(_decMaskInfo, new TensorShape(1, sequence.Count), mask));
        }

        // use_cache_branch: false on the prefill pass, true on cached steps.
        if (_useCacheBranchInfo is not null)
            feeds[_options.UseCacheBranchName] = new NamedTensor(
                _options.UseCacheBranchName, BuildBoolTensor(_useCacheBranchInfo, selfPast is not null));

        if (cache)
        {
            // Self-attention past: empty on the first pass, otherwise the previous present.*.decoder.*.
            FeedPast(feeds, _selfPairs, selfPast);
            // Cross-attention past: empty on the first pass (the graph still computes + outputs it), then
            // the captured first-pass present.*.encoder.* re-fed unchanged every later step.
            FeedPast(feeds, _crossPairs, crossPast);
        }

        return feeds;
    }

    private void FeedPast(
        Dictionary<string, NamedTensor> feeds, IReadOnlyList<Seq2SeqKvPair> pairs,
        Dictionary<string, Tensor>? past)
    {
        if (past is null)
        {
            foreach (Seq2SeqKvPair pair in pairs)
                feeds[pair.PastInput] = new NamedTensor(pair.PastInput, CreateEmptyPast(pair.Info));
        }
        else
        {
            foreach (KeyValuePair<string, Tensor> kvp in past)
                feeds[kvp.Key] = new NamedTensor(kvp.Key, kvp.Value);
        }
    }

    /// <summary>Rebinds the engine's present.* outputs (for the given pairs) to their past input names.</summary>
    private static Dictionary<string, Tensor> CapturePresent(
        IReadOnlyDictionary<string, NamedTensor> outputs, IReadOnlyList<Seq2SeqKvPair> pairs)
    {
        var map = new Dictionary<string, Tensor>(pairs.Count, StringComparer.Ordinal);
        foreach (Seq2SeqKvPair pair in pairs)
        {
            if (!outputs.TryGetValue(pair.PresentOutput, out NamedTensor? present))
                throw new ModelSharpException(
                    $"Decoder did not produce expected KV-cache output '{pair.PresentOutput}'.");
            map[pair.PastInput] = present.Tensor;
        }
        return map;
    }

    /// <summary>Reads the vocabulary distribution at the final decoder position (rank 1-3 accepted).</summary>
    private float[] ExtractLastPositionLogits(IReadOnlyDictionary<string, NamedTensor> outputs)
    {
        if (!outputs.TryGetValue(_options.LogitsOutputName, out NamedTensor? logitsNamed))
        {
            string presentDot = _options.PresentPrefix + ".";
            foreach (KeyValuePair<string, NamedTensor> kvp in outputs)
            {
                if (kvp.Key.StartsWith(presentDot, StringComparison.Ordinal)) continue;
                logitsNamed = kvp.Value;
                break;
            }
            logitsNamed ??= outputs.Values.FirstOrDefault();
        }
        if (logitsNamed is null)
            throw new ModelSharpException("Decoder produced no outputs to read logits from.");

        Tensor<float> logits = logitsNamed.Tensor.AsFloat();
        ReadOnlySpan<int> dims = logits.Shape.Dimensions;

        int vocab;
        int offset;
        switch (dims.Length)
        {
            case 3: vocab = dims[2]; offset = (dims[1] - 1) * vocab; break; // [batch, seq, vocab]
            case 2: vocab = dims[1]; offset = (dims[0] - 1) * vocab; break; // [batch, vocab] / [seq, vocab]
            case 1: vocab = dims[0]; offset = 0; break;                     // [vocab]
            default:
                throw new ModelSharpException(
                    $"Unexpected logits rank {dims.Length} (shape {logits.Shape}); expected 1-3 dimensions.");
        }

        var row = new float[vocab];
        logits.Span.Slice(offset, vocab).CopyTo(row);
        return row;
    }

    // ---- KV layout detection ----

    /// <summary>
    /// Splits the decoder's past-&gt;present KV pairings into self-attention and cross-attention groups by the
    /// configured infixes. A pairing exists when a <c>past_key_values.*</c> input has a matching
    /// <c>present.*</c> output (same suffix). Pairs whose name carries neither infix are treated as
    /// self-attention (the simplest decoder-with-past form, e.g. <c>past_key_values.0.key</c>).
    /// </summary>
    private (IReadOnlyList<Seq2SeqKvPair> Self, IReadOnlyList<Seq2SeqKvPair> Cross) DetectKvPairs()
    {
        var outputs = new HashSet<string>(StringComparer.Ordinal);
        foreach (TensorInfo o in _decoder.Outputs) outputs.Add(o.Name);

        string pastDot = _options.PastKeyValuesPrefix + ".";
        string crossInfix = "." + _options.CrossAttentionInfix + ".";
        var self = new List<Seq2SeqKvPair>();
        var cross = new List<Seq2SeqKvPair>();

        foreach (TensorInfo input in _decoder.Inputs)
        {
            if (!input.Name.StartsWith(pastDot, StringComparison.Ordinal)) continue;
            string presentName = _options.PresentPrefix + input.Name.Substring(_options.PastKeyValuesPrefix.Length);
            if (!outputs.Contains(presentName)) continue;

            var pair = new Seq2SeqKvPair(input.Name, presentName, input);
            if (input.Name.Contains(crossInfix, StringComparison.Ordinal)) cross.Add(pair);
            else self.Add(pair);
        }
        return (self, cross);
    }

    private static TensorInfo? FindInput(IExecutionEngine engine, string name)
    {
        foreach (TensorInfo info in engine.Inputs)
            if (info.Name == name) return info;
        return null;
    }

    // ---- tensor builders (mirrors TextGenerator's conventions) ----

    private static Tensor BuildIntTensor(TensorInfo? info, TensorShape shape, long[] values)
    {
        if (info?.ElementType == ElementType.Int32)
        {
            var buffer = new int[values.Length];
            for (int i = 0; i < values.Length; i++) buffer[i] = checked((int)values[i]);
            return new Tensor<int>(shape, buffer);
        }
        return new Tensor<long>(shape, values);
    }

    private static Tensor BuildBoolTensor(TensorInfo info, bool value)
    {
        var shape = new TensorShape(1);
        return info.ElementType switch
        {
            ElementType.Int32 => new Tensor<int>(shape, new[] { value ? 1 : 0 }),
            ElementType.Int64 => new Tensor<long>(shape, new long[] { value ? 1L : 0L }),
            _ => new Tensor<bool>(shape, new[] { value }),
        };
    }

    /// <summary>Allocates a zero-length-on-the-sequence-axis past-KV tensor for the first pass.</summary>
    private Tensor CreateEmptyPast(TensorInfo info)
    {
        int batch = _options.BatchSize;
        int seqAxis = _options.KvSequenceAxis;
        IReadOnlyList<int> declared = info.Dimensions;
        int[] shape;

        if (declared.Count == 0)
        {
            if (_options.KvCacheNumHeads is int heads && _options.KvCacheHeadDim is int headDim)
                shape = new[] { batch, heads, 0, headDim };
            else
                throw new ModelSharpException(
                    $"KV-cache input '{info.Name}' has no declared shape; set Seq2SeqModelOptions.KvCacheNumHeads " +
                    "and KvCacheHeadDim (from the model config) so empty past tensors can be allocated.");
        }
        else
        {
            shape = new int[declared.Count];
            for (int axis = 0; axis < declared.Count; axis++)
            {
                int dim = declared[axis];
                if (axis == seqAxis) shape[axis] = 0;
                else if (axis == 0) shape[axis] = dim < 0 ? batch : dim;
                else if (dim >= 0) shape[axis] = dim;
                else if (declared.Count == 4 && axis == 1 && _options.KvCacheNumHeads is int h) shape[axis] = h;
                else if (declared.Count == 4 && axis == declared.Count - 1 && _options.KvCacheHeadDim is int d) shape[axis] = d;
                else
                    throw new ModelSharpException(
                        $"KV-cache input '{info.Name}' axis {axis} is dynamic and cannot be resolved; " +
                        "supply Seq2SeqModelOptions.KvCacheNumHeads / KvCacheHeadDim.");
            }
        }

        ElementType dtype = info.ElementType == ElementType.Unknown ? ElementType.Float32 : info.ElementType;
        return CreateZeroTensor(dtype, new TensorShape(shape));
    }

    private static Tensor CreateZeroTensor(ElementType dtype, TensorShape shape) => dtype switch
    {
        ElementType.Float32 => new Tensor<float>(shape),
        ElementType.Float64 => new Tensor<double>(shape),
        ElementType.Int64 => new Tensor<long>(shape),
        ElementType.Int32 => new Tensor<int>(shape),
        ElementType.Int8 => new Tensor<sbyte>(shape),
        ElementType.UInt8 => new Tensor<byte>(shape),
        ElementType.Boolean => new Tensor<bool>(shape),
        _ => new Tensor<float>(shape),
    };

    private static bool IsEos(long token, GenerationConfig config)
    {
        if (config.EosTokenIds is null) return false;
        foreach (int eos in config.EosTokenIds)
            if (eos == token) return true;
        return false;
    }
}

/// <summary>A paired past-KV decoder input and its matching present-KV output for one seq2seq cache slot.</summary>
internal readonly struct Seq2SeqKvPair
{
    /// <summary>Decoder input name carrying the past cache (e.g. <c>past_key_values.0.decoder.key</c>).</summary>
    public string PastInput { get; }

    /// <summary>Decoder output name producing the updated cache (e.g. <c>present.0.decoder.key</c>).</summary>
    public string PresentOutput { get; }

    /// <summary>Static description of the past input (used to size the empty first-pass tensor).</summary>
    public TensorInfo Info { get; }

    public Seq2SeqKvPair(string pastInput, string presentOutput, TensorInfo info)
    {
        PastInput = pastInput;
        PresentOutput = presentOutput;
        Info = info;
    }
}
