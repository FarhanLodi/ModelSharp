using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Focused tests for the optimized KV-cache decode loop in <see cref="TextGenerator"/>.
///
/// <para>The goal these tests pin down: after the one-shot prefill pass, every subsequent decode
/// step must be a true single-token step — <c>input_ids</c> of length 1, the matching single
/// <c>position_ids</c>, a full-length <c>attention_mask</c>, and the previous step's
/// <c>present.N.*</c> outputs rebound as this step's growing <c>past_key_values.N.*</c> inputs
/// (so the model never re-prefills the whole sequence).</para>
///
/// <para>The <see cref="RecordingEngine"/> snapshots every per-step feed (shapes, the actual fed
/// past tensors by reference, position ids, attention-mask length) so we can assert the
/// single-token property directly, prove the cache grows by exactly one token per step, and verify
/// the threaded past tensor on step N+1 is the very <c>present</c> tensor produced on step N (no
/// copy / no re-prefill).</para>
/// </summary>
public class KvCacheDecodeTests
{
    // ----------------------------------------------------------------------------------------
    // Recording fake engine
    // ----------------------------------------------------------------------------------------

    private sealed class StepFeeds
    {
        public string[] Names = Array.Empty<string>();
        public int InputIdsLength;
        public long[] InputIds = Array.Empty<long>();
        public int AttentionMaskLength = -1;
        public int PastSeqLength = -1;
        public long[]? Positions;
        // The actual tensor instances fed as past_key_values.0.{key,value} this step (by reference).
        public Tensor? PastKey;
        public Tensor? PastValue;
        // The present.0.{key,value} tensors this step produced (by reference).
        public Tensor? PresentKey;
        public Tensor? PresentValue;
    }

    /// <summary>
    /// A decoder-with-past engine that returns scripted logits and fabricates <c>present.*</c> KV
    /// tensors whose sequence axis is <c>pastSeq + chunkLength</c>. Every call's feeds (and the
    /// produced present tensors) are recorded for inspection.
    /// </summary>
    private sealed class RecordingEngine : IExecutionEngine
    {
        private readonly Func<int, float[]> _logits;
        private readonly int _kvHeads;
        private readonly int _headDim;
        private int _call;

        public List<StepFeeds> Steps { get; } = new();
        public IReadOnlyList<TensorInfo> Inputs { get; }
        public IReadOnlyList<TensorInfo> Outputs { get; }

        public RecordingEngine(Func<int, float[]> logits, int kvHeads = 2, int headDim = 4)
        {
            _logits = logits;
            _kvHeads = kvHeads;
            _headDim = headDim;
            Inputs = new[]
            {
                new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()),
                new TensorInfo("attention_mask", ElementType.Int64, Array.Empty<int>()),
                new TensorInfo("position_ids", ElementType.Int64, Array.Empty<int>()),
                new TensorInfo("past_key_values.0.key", ElementType.Float32, new[] { 1, kvHeads, 1, headDim }),
                new TensorInfo("past_key_values.0.value", ElementType.Float32, new[] { 1, kvHeads, 1, headDim }),
            };
            Outputs = new[]
            {
                new TensorInfo("logits", ElementType.Float32, Array.Empty<int>()),
                new TensorInfo("present.0.key", ElementType.Float32, Array.Empty<int>()),
                new TensorInfo("present.0.value", ElementType.Float32, Array.Empty<int>()),
            };
        }

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            var step = new StepFeeds
            {
                Names = feeds.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray(),
                InputIdsLength = feeds["input_ids"].Tensor.Shape.Dimensions[1],
                InputIds = feeds["input_ids"].Tensor.AsInt64().Span.ToArray(),
                AttentionMaskLength = feeds["attention_mask"].Tensor.Shape.Dimensions[1],
                Positions = feeds["position_ids"].Tensor.AsInt64().Span.ToArray(),
                PastKey = feeds["past_key_values.0.key"].Tensor,
                PastValue = feeds["past_key_values.0.value"].Tensor,
            };
            step.PastSeqLength = step.PastKey!.Shape.Dimensions[2];

            int seq = step.PastSeqLength + step.InputIdsLength;
            var presentKey = new Tensor<float>(new TensorShape(1, _kvHeads, seq, _headDim));
            var presentValue = new Tensor<float>(new TensorShape(1, _kvHeads, seq, _headDim));
            step.PresentKey = presentKey;
            step.PresentValue = presentValue;
            Steps.Add(step);

            float[] row = _logits(_call);
            _call++;

            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["logits"] = new NamedTensor("logits", Tensor<float>.FromArray(new TensorShape(1, 1, row.Length), row)),
                ["present.0.key"] = new NamedTensor("present.0.key", presentKey),
                ["present.0.value"] = new NamedTensor("present.0.value", presentValue),
            };
        }

        public void Dispose() { }
    }

    /// <summary>A no-cache reference engine: it re-feeds the whole growing sequence each step.</summary>
    private sealed class NoCacheReferenceEngine : IExecutionEngine
    {
        private readonly Func<int, float[]> _logits;
        private int _call;
        public List<int> InputIdsLengths { get; } = new();

        public NoCacheReferenceEngine(Func<int, float[]> logits) => _logits = logits;

        public IReadOnlyList<TensorInfo> Inputs { get; } =
            new[] { new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()) };
        public IReadOnlyList<TensorInfo> Outputs { get; } =
            new[] { new TensorInfo("logits", ElementType.Float32, Array.Empty<int>()) };

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            InputIdsLengths.Add(feeds["input_ids"].Tensor.Shape.Dimensions[1]);
            float[] row = _logits(_call);
            _call++;
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["logits"] = new NamedTensor("logits", Tensor<float>.FromArray(new TensorShape(1, 1, row.Length), row)),
            };
        }

        public void Dispose() { }
    }

    private static float[] OneHot(int vocab, int peak)
    {
        var row = new float[vocab];
        row[peak] = 10f;
        return row;
    }

    // ----------------------------------------------------------------------------------------
    // Single-token decode property
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void Decode_AfterPrefill_FeedsExactlyOneTokenPerStep()
    {
        var engine = new RecordingEngine(_ => OneHot(16, 7));
        var generator = new TextGenerator(engine);

        long[] prompt = { 10, 11, 12, 13 };
        generator.Generate(prompt, new GenerationConfig { MaxNewTokens = 5, DoSample = false });

        Assert.Equal(5, engine.Steps.Count);

        // Prefill: the entire prompt in one shot, with an empty past.
        Assert.Equal(prompt.Length, engine.Steps[0].InputIdsLength);
        Assert.Equal(0, engine.Steps[0].PastSeqLength);
        Assert.Equal(prompt, engine.Steps[0].InputIds);

        // Every decode step: a SINGLE new token (seq len 1) — proving no re-prefill.
        for (int s = 1; s < engine.Steps.Count; s++)
        {
            Assert.Equal(1, engine.Steps[s].InputIdsLength);
            Assert.Single(engine.Steps[s].InputIds);
        }
    }

    [Fact]
    public void Decode_PastGrowsByExactlyOneTokenPerStep()
    {
        var engine = new RecordingEngine(_ => OneHot(16, 3));
        var generator = new TextGenerator(engine);

        long[] prompt = { 1, 2, 3 };
        generator.Generate(prompt, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        // past length: 0 (prefill), then prompt(3), 4, 5 — grows by exactly 1 each decode step.
        Assert.Equal(new[] { 0, 3, 4, 5 }, engine.Steps.Select(s => s.PastSeqLength).ToArray());

        // attention_mask always spans past + current chunk (the full sequence so far).
        Assert.Equal(new[] { 3, 4, 5, 6 }, engine.Steps.Select(s => s.AttentionMaskLength).ToArray());

        // position_ids: the whole prompt on prefill, then the single next absolute position.
        Assert.Equal(new long[] { 0, 1, 2 }, engine.Steps[0].Positions);
        Assert.Equal(new long[] { 3 }, engine.Steps[1].Positions);
        Assert.Equal(new long[] { 4 }, engine.Steps[2].Positions);
        Assert.Equal(new long[] { 5 }, engine.Steps[3].Positions);
    }

    [Fact]
    public void Decode_ThreadsPresentTensorIntoNextStepPast_ByReference_NoCopy()
    {
        var engine = new RecordingEngine(_ => OneHot(16, 9));
        var generator = new TextGenerator(engine);

        generator.Generate(new long[] { 5, 6 }, new GenerationConfig { MaxNewTokens = 3, DoSample = false });

        Assert.Equal(3, engine.Steps.Count);

        // The single load-bearing optimisation: the present.N.* tensor produced on step t is the very
        // same object fed back as past_key_values.N.* on step t+1 (no per-step copy, no re-prefill).
        for (int t = 0; t < engine.Steps.Count - 1; t++)
        {
            Assert.Same(engine.Steps[t].PresentKey, engine.Steps[t + 1].PastKey);
            Assert.Same(engine.Steps[t].PresentValue, engine.Steps[t + 1].PastValue);
        }
    }

    [Fact]
    public void Decode_FedTokenMatchesScriptedLogitsArgmax()
    {
        // Scripted argmax sequence: each step's chosen token is the next step's single input token.
        int[] peaks = { 4, 8, 2, 5 };
        var engine = new RecordingEngine(call => OneHot(16, peaks[call]));
        var generator = new TextGenerator(engine);

        long[] prompt = { 0 };
        IReadOnlyList<long> generated =
            generator.Generate(prompt, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        Assert.Equal(new long[] { 4, 8, 2, 5 }, generated.ToArray());

        // Step 0 feeds the prompt; each later step feeds the token produced on the previous step.
        Assert.Equal(new long[] { 0 }, engine.Steps[0].InputIds);
        Assert.Equal(new long[] { 4 }, engine.Steps[1].InputIds);
        Assert.Equal(new long[] { 8 }, engine.Steps[2].InputIds);
        Assert.Equal(new long[] { 2 }, engine.Steps[3].InputIds);
    }

    // ----------------------------------------------------------------------------------------
    // Correctness of the optimization: cache path == no-cache reference
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void CachePath_And_NoCacheReference_ProduceIdenticalTokens()
    {
        // Argmax depends only on the (call-indexed) scripted logits, so a correct cache loop and a
        // correct no-cache loop must select exactly the same token sequence.
        int[] peaks = { 6, 1, 9, 3, 14, 0, 11, 2 };
        Func<int, float[]> logits = call => OneHot(16, peaks[call]);
        var config = new GenerationConfig { MaxNewTokens = 8, DoSample = false };
        long[] prompt = { 100, 101, 102 };

        var cacheEngine = new RecordingEngine(logits);
        long[] cacheTokens = new TextGenerator(cacheEngine).Generate(prompt, config).ToArray();

        var refEngine = new NoCacheReferenceEngine(logits);
        long[] refTokens = new TextGenerator(refEngine).Generate(prompt, config).ToArray();

        Assert.Equal(refTokens, cacheTokens);
        Assert.Equal(peaks.Select(p => (long)p).ToArray(), cacheTokens);

        // Sanity: the two engines truly took different feed paths.
        // Cache: prompt-once then single tokens.
        Assert.Equal(1, cacheEngine.Steps[1].InputIdsLength);
        // No-cache: the whole growing sequence is re-fed each step.
        Assert.Equal(new[] { 3, 4, 5, 6, 7, 8, 9, 10 }, refEngine.InputIdsLengths.ToArray());
    }

    [Fact]
    public void CachePath_And_NoCacheReference_AgreeUnderSampling_WithSameSeed()
    {
        // Uniform logits => the chosen token is driven entirely by the seeded RNG, which advances
        // once per step in both paths. Same seed => identical sequence regardless of cache mode.
        Func<int, float[]> uniform = _ => new float[20];
        var config = new GenerationConfig { MaxNewTokens = 10, DoSample = true, Seed = 1234 };
        long[] prompt = { 7, 8 };

        long[] cacheTokens = new TextGenerator(new RecordingEngine(uniform)).Generate(prompt, config).ToArray();
        long[] refTokens = new TextGenerator(new NoCacheReferenceEngine(uniform)).Generate(prompt, config).ToArray();

        Assert.Equal(refTokens, cacheTokens);
    }

    // ----------------------------------------------------------------------------------------
    // Allocation-reduction: the reused attention_mask buffer still feeds correct values
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void ReusedAttentionMaskBuffer_IsAllOnes_AtEveryStep()
    {
        // The mask buffer is grown in place across steps; assert every step still fed a full run of 1s.
        var engine = new MaskCapturingEngine(_ => OneHot(8, 1));
        var generator = new TextGenerator(engine);

        generator.Generate(new long[] { 1, 2, 3 }, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        Assert.Equal(new[] { 3, 4, 5, 6 }, engine.MaskLengths);
        foreach (long[] mask in engine.Masks)
            Assert.All(mask, v => Assert.Equal(1L, v));
    }

    /// <summary>A cache engine that captures the full attention_mask contents fed each step.</summary>
    private sealed class MaskCapturingEngine : IExecutionEngine
    {
        private readonly RecordingEngine _inner;
        public List<int> MaskLengths { get; } = new();
        public List<long[]> Masks { get; } = new();

        public MaskCapturingEngine(Func<int, float[]> logits) => _inner = new RecordingEngine(logits);
        public IReadOnlyList<TensorInfo> Inputs => _inner.Inputs;
        public IReadOnlyList<TensorInfo> Outputs => _inner.Outputs;

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            long[] mask = feeds["attention_mask"].Tensor.AsInt64().Span.ToArray();
            MaskLengths.Add(mask.Length);
            Masks.Add(mask);
            return _inner.Run(feeds);
        }

        public void Dispose() => _inner.Dispose();
    }
}
