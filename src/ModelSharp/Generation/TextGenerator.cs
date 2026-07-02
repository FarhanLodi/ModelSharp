using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Engine;
using ModelSharp.Tensors;

namespace ModelSharp.Generation;

/// <summary>
/// Engine-agnostic autoregressive text generation loop. Drives any <see cref="IExecutionEngine"/>
/// that exposes a decoder model: at each step it builds the feeds, runs the engine, reads the
/// last-position logits, applies the configured <see cref="LogitsProcessor"/> pipeline, appends the
/// chosen token, and repeats until an EOS token or a length bound is hit.
///
/// <para>Two execution modes are supported and auto-detected from the engine's input names
/// (see <see cref="KvCacheLayout"/>):</para>
/// <list type="bullet">
///   <item><description><b>KV-cache</b> (decoder-with-past): the prompt is fed once in full, then each
///   subsequent step feeds only the single new token together with the previous <c>present.*</c>
///   outputs rebound as <c>past_key_values.*</c> inputs.</description></item>
///   <item><description><b>No-cache</b> fallback: the entire growing sequence is re-fed every step.</description></item>
/// </list>
///
/// <para><c>attention_mask</c> and <c>position_ids</c> are supplied automatically whenever the model
/// declares them. Token ids decode to text elsewhere — this type works purely in token-id space.</para>
/// </summary>
public sealed class TextGenerator
{
    private readonly IExecutionEngine _engine;
    private readonly DecoderModelOptions _options;
    private readonly KvCacheLayout _kvCache;
    private readonly TensorInfo? _inputIdsInfo;
    private readonly TensorInfo? _attentionMaskInfo;
    private readonly TensorInfo? _positionIdsInfo;
    private readonly TensorInfo? _useCacheBranchInfo;

    /// <summary>Creates a generator over the supplied engine.</summary>
    /// <param name="engine">The decoder execution engine to drive.</param>
    /// <param name="options">IO binding conventions; defaults to the HF/Optimum decoder layout.</param>
    public TextGenerator(IExecutionEngine engine, DecoderModelOptions? options = null)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _options = options ?? DecoderModelOptions.Default;
        if (_options.BatchSize != 1)
            throw new NotSupportedException("TextGenerator currently supports BatchSize == 1.");

        _kvCache = KvCacheLayout.Detect(engine, _options);
        _inputIdsInfo = FindInput(_options.InputIdsName);
        _attentionMaskInfo = FindInput(_options.AttentionMaskName);
        _positionIdsInfo = FindInput(_options.PositionIdsName);
        _useCacheBranchInfo = FindInput(_options.UseCacheBranchName);

        if (_inputIdsInfo is null)
            throw new ModelSharpException(
                $"Engine does not declare the token-ids input '{_options.InputIdsName}'. " +
                "Set DecoderModelOptions.InputIdsName to match the model.");
    }

    /// <summary>True when the engine follows the decoder-with-past (KV cache) convention.</summary>
    public bool UsesKvCache => _kvCache.IsCacheMode;

    /// <summary>Whether the model declares an attention-mask input that this generator will feed.</summary>
    public bool FeedsAttentionMask => _attentionMaskInfo is not null;

    /// <summary>Whether the model declares a position-ids input that this generator will feed.</summary>
    public bool FeedsPositionIds => _positionIdsInfo is not null;

    /// <summary>
    /// Whether the model declares a <c>use_cache_branch</c> input (Optimum "merged" decoder export)
    /// that this generator will feed: <c>false</c> on the prefill pass, <c>true</c> on cached steps.
    /// </summary>
    public bool FeedsUseCacheBranch => _useCacheBranchInfo is not null;

    /// <summary>
    /// Generates tokens for a prompt and returns the result eagerly.
    /// </summary>
    /// <param name="promptTokenIds">The prompt as token ids (must be non-empty).</param>
    /// <param name="config">Decoding parameters.</param>
    /// <param name="includePrompt">When true the returned list is prefixed with the prompt tokens.</param>
    /// <returns>The generated token ids (optionally including the prompt).</returns>
    public IReadOnlyList<long> Generate(IReadOnlyList<long> promptTokenIds, GenerationConfig config, bool includePrompt = false)
    {
        var result = includePrompt ? new List<long>(promptTokenIds) : new List<long>();
        foreach (long token in Iterate(promptTokenIds, config)) result.Add(token);
        return result;
    }

    /// <summary>
    /// Lazily streams generated tokens one at a time. The engine is invoked on demand as the
    /// sequence is enumerated, so a caller can decode and display tokens as they arrive.
    /// </summary>
    /// <param name="promptTokenIds">The prompt as token ids (must be non-empty).</param>
    /// <param name="config">Decoding parameters.</param>
    public IEnumerable<long> GenerateStream(IReadOnlyList<long> promptTokenIds, GenerationConfig config)
        => Iterate(promptTokenIds, config);

    /// <summary>
    /// Generates tokens and invokes <paramref name="onToken"/> for each newly produced token.
    /// </summary>
    /// <param name="promptTokenIds">The prompt as token ids (must be non-empty).</param>
    /// <param name="config">Decoding parameters.</param>
    /// <param name="onToken">Callback invoked once per generated token, in order.</param>
    public void Generate(IReadOnlyList<long> promptTokenIds, GenerationConfig config, Action<long> onToken)
    {
        if (onToken is null) throw new ArgumentNullException(nameof(onToken));
        foreach (long token in Iterate(promptTokenIds, config)) onToken(token);
    }

    /// <summary>The core autoregressive loop, shared by every public entry point.</summary>
    private IEnumerable<long> Iterate(IReadOnlyList<long> prompt, GenerationConfig config)
    {
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        if (config is null) throw new ArgumentNullException(nameof(config));
        config.Validate();
        if (prompt.Count == 0)
            throw new ArgumentException("Prompt must contain at least one token.", nameof(prompt));

        Random? rng = config.DoSample
            ? (config.Seed.HasValue ? new Random(config.Seed.Value) : new Random())
            : null;

        var sequence = new List<long>(prompt);
        int generatedCount = 0;
        bool cache = _kvCache.IsCacheMode;

        // pastInputName -> tensor, threaded from the previous step's present.* outputs. Reused across
        // steps so the threading map is allocated once, not per token.
        Dictionary<string, Tensor>? past = null;

        // Hot-loop scratch, allocated once and reused on every step to keep the decode loop
        // allocation-free apart from the engine's own outputs:
        //  - the feed dictionary is cleared + repopulated rather than reallocated;
        //  - the attention_mask buffer is filled with ones up to the prompt length and only grown
        //    (one extra "1" appended) per generated token, never reallocated unless capacity is hit;
        //  - a 1-element scratch backs the single-token input_ids / position_ids during decode.
        var feeds = new Dictionary<string, NamedTensor>(StringComparer.Ordinal);
        var maskState = _attentionMaskInfo is not null ? new MaskBuffer(sequence.Count) : null;

        // Reused across steps: the sampling pipeline needs a mutable copy of the last-position logits
        // row, but greedy decoding reads them in place. Allocated lazily on the first step that needs it.
        float[]? logitsScratch = null;

        while (true)
        {
            if (generatedCount >= config.MaxNewTokens) yield break;
            if (config.MaxLength > 0 && sequence.Count >= config.MaxLength) yield break;

            // In cache mode the prompt is fed whole once (past == null), then one token at a time.
            // In no-cache mode the whole growing sequence is fed every step.
            int chunkLength = (!cache || past is null) ? sequence.Count : 1;
            int startPosition = sequence.Count - chunkLength;

            BuildFeeds(feeds, maskState, sequence, chunkLength, startPosition, cache, past);
            IReadOnlyDictionary<string, NamedTensor> outputs = _engine.Run(feeds);

            (Tensor<float> logits, int offset, int vocab) = LocateLastPositionLogits(outputs);
            int nextId = LogitsProcessor.SelectNextTokenFromLogits(
                logits.Span, offset, vocab, config, sequence, rng, ref logitsScratch);
            long next = nextId;

            sequence.Add(next);
            generatedCount++;

            // Thread present.* -> past_key_values.* for the next step. We hold the engine's output
            // tensors by reference and feed them straight back as next-step past: no per-step copy or
            // re-tokenization of the KV cache. Reuses the same map instance across steps.
            if (cache) past = CapturePresent(outputs, past);

            yield return next;

            if (IsEos(next, config)) yield break;
        }
    }

    /// <summary>
    /// Populates the (reused) feed dictionary for one step in place. Clears the previous step's
    /// entries first, then rebinds input_ids / attention_mask / position_ids / use_cache_branch /
    /// past_key_values for the current chunk.
    /// </summary>
    private void BuildFeeds(
        Dictionary<string, NamedTensor> feeds, MaskBuffer? maskState,
        List<long> sequence, int chunkLength, int startPosition, bool cache, Dictionary<string, Tensor>? past)
    {
        feeds.Clear();

        // input_ids: the chunk being processed this step. In decode mode this is a single token.
        var ids = new long[chunkLength];
        for (int i = 0; i < chunkLength; i++) ids[i] = sequence[startPosition + i];
        feeds[_options.InputIdsName] = new NamedTensor(
            _options.InputIdsName, BuildIntTensor(_inputIdsInfo, new TensorShape(1, chunkLength), ids));

        // attention_mask: ones over every token the model attends to so far (past + current chunk).
        // Grown in place from a persistent buffer so each decode step appends a single "1" instead of
        // reallocating and refilling a full-length array.
        if (_attentionMaskInfo is not null)
        {
            int maskLength = sequence.Count;
            Memory<long> mask = maskState!.OnesUpTo(maskLength);
            feeds[_options.AttentionMaskName] = new NamedTensor(
                _options.AttentionMaskName, BuildIntTensor(_attentionMaskInfo, new TensorShape(1, maskLength), mask));
        }

        // position_ids: absolute positions of the chunk tokens.
        if (_positionIdsInfo is not null)
        {
            var positions = new long[chunkLength];
            for (int i = 0; i < chunkLength; i++) positions[i] = startPosition + i;
            feeds[_options.PositionIdsName] = new NamedTensor(
                _options.PositionIdsName, BuildIntTensor(_positionIdsInfo, new TensorShape(1, chunkLength), positions));
        }

        // use_cache_branch (Optimum "merged" exports): false selects the no-past branch on the first
        // pass, true selects the with-past branch on every subsequent cached step. The empty past
        // tensors are still fed on the first pass below so the graph's input bindings all exist.
        if (_useCacheBranchInfo is not null)
        {
            bool useCache = past is not null;
            feeds[_options.UseCacheBranchName] = new NamedTensor(
                _options.UseCacheBranchName, BuildBoolTensor(_useCacheBranchInfo, useCache));
        }

        // past_key_values: empty on the first pass, otherwise the previous present.* outputs threaded
        // straight through (no copy).
        if (cache)
        {
            if (past is null)
            {
                foreach (KvCachePair pair in _kvCache.Pairs)
                    feeds[pair.PastInput] = new NamedTensor(pair.PastInput, CreateEmptyPast(pair.Info));
            }
            else
            {
                foreach (KeyValuePair<string, Tensor> kvp in past)
                    feeds[kvp.Key] = new NamedTensor(kvp.Key, kvp.Value);
            }
        }
    }

    /// <summary>
    /// Rebinds the engine's present.* outputs to their past_key_values.* input names for the next
    /// step. The present tensors are taken by reference (no copy / no re-prefill); only the small
    /// name->tensor map is updated, reusing <paramref name="reuse"/> when one already exists.
    /// </summary>
    private Dictionary<string, Tensor> CapturePresent(
        IReadOnlyDictionary<string, NamedTensor> outputs, Dictionary<string, Tensor>? reuse)
    {
        Dictionary<string, Tensor> map = reuse ?? new Dictionary<string, Tensor>(_kvCache.Pairs.Count, StringComparer.Ordinal);
        foreach (KvCachePair pair in _kvCache.Pairs)
        {
            if (!outputs.TryGetValue(pair.PresentOutput, out NamedTensor? present))
                throw new ModelSharpException(
                    $"Engine did not produce expected KV-cache output '{pair.PresentOutput}'.");
            map[pair.PastInput] = present.Tensor;
        }
        return map;
    }

    /// <summary>
    /// Locates the vocabulary distribution at the final sequence position without copying it. Accepts
    /// logits shaped <c>[batch, seq, vocab]</c>, <c>[batch, vocab]</c> (single-position cached run, or
    /// <c>[seq, vocab]</c>), or a bare <c>[vocab]</c> row, and returns the backing tensor together with
    /// the <c>offset</c>/<c>vocab</c> of the last-position row inside it. The caller reads the row in
    /// place (greedy) or copies it into a reused scratch buffer (sampling).
    /// </summary>
    private (Tensor<float> Logits, int Offset, int Vocab) LocateLastPositionLogits(
        IReadOnlyDictionary<string, NamedTensor> outputs)
    {
        if (!outputs.TryGetValue(_options.LogitsOutputName, out NamedTensor? logitsNamed))
        {
            // Fall back to the first output that is not a present.* cache tensor.
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
            throw new ModelSharpException("Engine produced no outputs to read logits from.");

        Tensor<float> logits = logitsNamed.Tensor.AsFloat();
        ReadOnlySpan<int> dims = logits.Shape.Dimensions;

        int vocab;
        int offset;
        switch (dims.Length)
        {
            case 3: // [batch, seq, vocab] -> last position of batch 0
                vocab = dims[2];
                offset = (dims[1] - 1) * vocab;
                break;
            case 2: // [batch, vocab] or [seq, vocab] -> last row
                vocab = dims[1];
                offset = (dims[0] - 1) * vocab;
                break;
            case 1: // [vocab]
                vocab = dims[0];
                offset = 0;
                break;
            default:
                throw new ModelSharpException(
                    $"Unexpected logits rank {dims.Length} (shape {logits.Shape}); expected 1-3 dimensions.");
        }

        return (logits, offset, vocab);
    }

    private TensorInfo? FindInput(string name)
    {
        foreach (TensorInfo info in _engine.Inputs)
            if (info.Name == name) return info;
        return null;
    }

    /// <summary>
    /// Builds an integer feed tensor. Token ids/mask/positions are fed as int64 by default (the HF
    /// convention); an engine that explicitly declares the binding as <see cref="ElementType.Int32"/>
    /// receives int32 instead.
    /// </summary>
    private static Tensor BuildIntTensor(TensorInfo? info, TensorShape shape, long[] values)
        => BuildIntTensor(info, shape, (Memory<long>)values);

    /// <summary>
    /// Builds an integer feed tensor from an int64 span. When the binding is declared int64 the buffer
    /// is wrapped without a copy (so a reused backing array threads straight into the feed); int32
    /// bindings are narrowed into a fresh buffer.
    /// </summary>
    private static Tensor BuildIntTensor(TensorInfo? info, TensorShape shape, Memory<long> values)
    {
        if (info?.ElementType == ElementType.Int32)
        {
            ReadOnlySpan<long> src = values.Span;
            var buffer = new int[src.Length];
            for (int i = 0; i < src.Length; i++) buffer[i] = checked((int)src[i]);
            return new Tensor<int>(shape, buffer);
        }
        return new Tensor<long>(shape, values);
    }

    /// <summary>
    /// Builds the single-element <c>use_cache_branch</c> feed (shape <c>[1]</c>). The HF/Optimum
    /// convention declares it as <see cref="ElementType.Boolean"/>; an engine that instead declares
    /// it as Int32/Int64 receives <c>1</c>/<c>0</c> in that dtype (mirrors the int dtype-adaptation).
    /// </summary>
    private static Tensor BuildBoolTensor(TensorInfo info, bool value)
    {
        var shape = new TensorShape(1);
        switch (info.ElementType)
        {
            case ElementType.Int32:
                return new Tensor<int>(shape, new[] { value ? 1 : 0 });
            case ElementType.Int64:
                return new Tensor<long>(shape, new long[] { value ? 1L : 0L });
            default:
                return new Tensor<bool>(shape, new[] { value });
        }
    }

    /// <summary>
    /// Allocates a zero-length (sequence axis = 0) past-KV tensor for the first pass. Concrete dims are
    /// taken from the engine's declared shape; dynamic axes resolve to the batch size, 0 on the sequence
    /// axis, or — when the engine reports no shape at all — to the canonical
    /// <c>[batch, KvCacheNumHeads, 0, KvCacheHeadDim]</c> layout from <see cref="DecoderModelOptions"/>.
    /// </summary>
    private Tensor CreateEmptyPast(TensorInfo info)
    {
        int batch = _options.BatchSize;
        int seqAxis = _kvCache.SequenceAxis;
        IReadOnlyList<int> declared = info.Dimensions;
        int[] shape;

        if (declared.Count == 0)
        {
            if (_options.KvCacheNumHeads is int heads && _options.KvCacheHeadDim is int headDim)
            {
                shape = new[] { batch, heads, 0, headDim };
            }
            else
            {
                throw new ModelSharpException(
                    $"KV-cache input '{info.Name}' has no declared shape; set DecoderModelOptions.KvCacheNumHeads " +
                    "and KvCacheHeadDim (from the model config) so empty past tensors can be allocated.");
            }
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
                        "supply DecoderModelOptions.KvCacheNumHeads / KvCacheHeadDim.");
            }
        }

        ElementType dtype = info.ElementType == ElementType.Unknown ? ElementType.Float32 : info.ElementType;
        return CreateZeroTensor(dtype, new TensorShape(shape));
    }

    /// <summary>Allocates a zero-filled tensor of the requested dtype and shape.</summary>
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

    /// <summary>
    /// A grow-only buffer of all-ones int64 attention-mask values, reused across decode steps. The
    /// attention mask is always a run of <c>1</c>s of the current sequence length, so each step only
    /// needs the buffer to be at least that long; we fill any newly exposed tail with <c>1</c> once
    /// and hand back a length-bounded view, avoiding a fresh full-length allocation per token.
    /// </summary>
    private sealed class MaskBuffer
    {
        private long[] _ones;
        private int _filled;

        public MaskBuffer(int initialLength)
        {
            int cap = Math.Max(initialLength, 1);
            _ones = new long[cap];
            // _filled left at 0; OnesUpTo fills lazily on first use.
        }

        /// <summary>Returns a view of <paramref name="length"/> ones, growing/filling the buffer as needed.</summary>
        public Memory<long> OnesUpTo(int length)
        {
            if (length > _ones.Length)
            {
                int cap = _ones.Length;
                while (cap < length) cap *= 2;
                Array.Resize(ref _ones, cap);
                // Resize preserves existing (already-filled) ones; the new tail is zero-filled and
                // will be set below.
            }
            if (length > _filled)
            {
                for (int i = _filled; i < length; i++) _ones[i] = 1;
                _filled = length;
            }
            return new Memory<long>(_ones, 0, length);
        }
    }
}
