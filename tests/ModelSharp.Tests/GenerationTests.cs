using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Deterministic tests for the autoregressive generation engine. Everything runs against an
/// in-test <see cref="FakeEngine"/> that returns scripted logits and records the feeds it sees,
/// so no real model is required and the behaviour is fully reproducible.
/// </summary>
public class GenerationTests
{
    // ----------------------------------------------------------------------------------------
    // Fake engine + helpers
    // ----------------------------------------------------------------------------------------

    /// <summary>Captured snapshot of one engine invocation, used by the KV/no-cache assertions.</summary>
    private sealed class CallRecord
    {
        public string[] FeedNames = Array.Empty<string>();
        public int InputIdsLength;
        public int AttentionMaskLength = -1;
        public int PastSequenceLength = -1;
        public long[]? Positions;
        public bool? UseCacheBranch;
    }

    /// <summary>
    /// Scripted, recording <see cref="IExecutionEngine"/>. Returns the logits supplied by a
    /// per-call function and, when configured with present.* outputs, fabricates KV tensors whose
    /// sequence axis grows by the fed chunk length each step.
    /// </summary>
    private sealed class FakeEngine : IExecutionEngine
    {
        private readonly Func<int, float[]> _logits;
        private readonly bool _logitsRank2;
        private int _call;

        public List<CallRecord> Records { get; } = new();
        public IReadOnlyList<TensorInfo> Inputs { get; }
        public IReadOnlyList<TensorInfo> Outputs { get; }

        public FakeEngine(
            IReadOnlyList<TensorInfo> inputs,
            IReadOnlyList<TensorInfo> outputs,
            Func<int, float[]> logits,
            bool logitsRank2 = false)
        {
            Inputs = inputs;
            Outputs = outputs;
            _logits = logits;
            _logitsRank2 = logitsRank2;
        }

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            var record = new CallRecord
            {
                FeedNames = feeds.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray(),
                InputIdsLength = feeds["input_ids"].Tensor.Shape.Dimensions[1],
                AttentionMaskLength = feeds.ContainsKey("attention_mask")
                    ? feeds["attention_mask"].Tensor.Shape.Dimensions[1] : -1,
                PastSequenceLength = feeds.ContainsKey("past_key_values.0.key")
                    ? feeds["past_key_values.0.key"].Tensor.Shape.Dimensions[2] : -1,
                Positions = feeds.ContainsKey("position_ids")
                    ? feeds["position_ids"].Tensor.AsInt64().Span.ToArray() : null,
                UseCacheBranch = feeds.ContainsKey("use_cache_branch")
                    ? ReadBoolFeed(feeds["use_cache_branch"].Tensor) : null,
            };
            Records.Add(record);

            float[] row = _logits(_call);
            _call++;

            var result = new Dictionary<string, NamedTensor>(StringComparer.Ordinal);
            Tensor<float> logits = _logitsRank2
                ? Tensor<float>.FromArray(new TensorShape(1, row.Length), row)
                : Tensor<float>.FromArray(new TensorShape(1, 1, row.Length), row);
            string logitsName = Outputs.Any(o => o.Name == "logits") ? "logits" : Outputs[0].Name;
            result[logitsName] = new NamedTensor(logitsName, logits);

            // Fabricate present.* KV outputs that grow along the sequence axis.
            foreach (TensorInfo o in Outputs)
            {
                if (!o.Name.StartsWith("present.", StringComparison.Ordinal)) continue;
                int pastSeq = record.PastSequenceLength < 0 ? 0 : record.PastSequenceLength;
                int seq = pastSeq + record.InputIdsLength;
                result[o.Name] = new NamedTensor(o.Name, new Tensor<float>(new TensorShape(1, 2, seq, 4)));
            }
            return result;
        }

        public void Dispose() { }
    }

    /// <summary>A fixed-draw RNG so multinomial sampling is exactly computable in tests.</summary>
    private sealed class FixedRandom : Random
    {
        private readonly double _value;
        public FixedRandom(double value) => _value = value;
        public override double NextDouble() => _value;
    }

    private static TensorInfo In(string name, ElementType dt = ElementType.Int64, int[]? dims = null) =>
        new(name, dt, dims ?? Array.Empty<int>());

    private static TensorInfo Out(string name) => new(name, ElementType.Float32, Array.Empty<int>());

    /// <summary>A logits row whose argmax is <paramref name="peak"/>.</summary>
    private static float[] OneHot(int vocab, int peak)
    {
        var row = new float[vocab];
        row[peak] = 10f;
        return row;
    }

    private static float[] Clone(float[] a) => (float[])a.Clone();

    /// <summary>Reads a single-element use_cache_branch feed as a bool, regardless of declared dtype.</summary>
    private static bool ReadBoolFeed(Tensor tensor) => tensor switch
    {
        Tensor<bool> tb => tb.Span[0],
        Tensor<int> ti => ti.Span[0] != 0,
        Tensor<long> tl => tl.Span[0] != 0,
        _ => throw new InvalidOperationException($"Unexpected use_cache_branch tensor type {tensor.GetType().Name}."),
    };

    private static FakeEngine NoCacheEngine(Func<int, float[]> logits, bool rank2 = false) =>
        new(new[] { In("input_ids") }, new[] { Out("logits") }, logits, rank2);

    // ----------------------------------------------------------------------------------------
    // Greedy decoding: argmax, EOS stop, MaxNewTokens
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void Greedy_PicksArgmax_EachStep()
    {
        int[] peaks = { 3, 1, 4, 2 };
        var engine = NoCacheEngine(call => OneHot(8, peaks[call]));
        var generator = new TextGenerator(engine);

        IReadOnlyList<long> generated = generator.Generate(
            new long[] { 0 }, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        Assert.Equal(new long[] { 3, 1, 4, 2 }, generated.ToArray());
    }

    [Fact]
    public void Greedy_StopsExactlyAtEos_IncludingTheEosToken()
    {
        int[] peaks = { 3, 1, 4, 9 }; // 4 is EOS; token 9 must never be produced
        var engine = NoCacheEngine(call => OneHot(10, peaks[call]));
        var generator = new TextGenerator(engine);

        IReadOnlyList<long> generated = generator.Generate(
            new long[] { 0 },
            new GenerationConfig { MaxNewTokens = 50, DoSample = false, EosTokenIds = new[] { 4 } });

        Assert.Equal(new long[] { 3, 1, 4 }, generated.ToArray());
        Assert.Equal(3, engine.Records.Count); // stopped right after EOS; no extra run
    }

    [Fact]
    public void Greedy_RespectsMaxNewTokens()
    {
        var engine = NoCacheEngine(call => OneHot(8, 7));
        var generator = new TextGenerator(engine);

        IReadOnlyList<long> generated = generator.Generate(
            new long[] { 0 }, new GenerationConfig { MaxNewTokens = 5, DoSample = false });

        Assert.Equal(5, generated.Count);
        Assert.All(generated, t => Assert.Equal(7, t));
    }

    [Fact]
    public void Generate_CanIncludeOrExcludePrompt()
    {
        var engine = NoCacheEngine(call => OneHot(8, 7));
        var generator = new TextGenerator(engine);
        var config = new GenerationConfig { MaxNewTokens = 2, DoSample = false };

        Assert.Equal(new long[] { 7, 7 }, generator.Generate(new long[] { 5, 6 }, config).ToArray());

        var engine2 = NoCacheEngine(call => OneHot(8, 7));
        var generator2 = new TextGenerator(engine2);
        Assert.Equal(new long[] { 5, 6, 7, 7 },
            generator2.Generate(new long[] { 5, 6 }, config, includePrompt: true).ToArray());
    }

    [Fact]
    public void MaxLength_CapsTotalSequenceLength()
    {
        var engine = NoCacheEngine(call => OneHot(8, 7));
        var generator = new TextGenerator(engine);

        // prompt length 2, MaxLength 5 => at most 3 new tokens, regardless of MaxNewTokens.
        IReadOnlyList<long> generated = generator.Generate(
            new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 100, MaxLength = 5, DoSample = false });

        Assert.Equal(3, generated.Count);
    }

    // ----------------------------------------------------------------------------------------
    // Streaming + callback variants
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void StreamingAndCallback_MatchEagerGenerate()
    {
        int[] peaks = { 3, 1, 4 };
        var config = new GenerationConfig { MaxNewTokens = 3, DoSample = false };

        var streamed = new TextGenerator(NoCacheEngine(c => OneHot(8, peaks[c])))
            .GenerateStream(new long[] { 0 }, config).ToArray();

        var collected = new List<long>();
        new TextGenerator(NoCacheEngine(c => OneHot(8, peaks[c])))
            .Generate(new long[] { 0 }, config, collected.Add);

        Assert.Equal(new long[] { 3, 1, 4 }, streamed);
        Assert.Equal(streamed, collected.ToArray());
    }

    // ----------------------------------------------------------------------------------------
    // LogitsProcessor: pure warpers
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void RepetitionPenalty_ShiftsTheArgmax()
    {
        // Without penalty argmax is token 0; penalising the already-seen token 0 shifts it to token 1.
        float[] baseLogits = { 3f, 2f, 1f, 0f };
        Assert.Equal(0, LogitsProcessor.ArgMax(baseLogits));

        var penalised = Clone(baseLogits);
        LogitsProcessor.ApplyRepetitionPenalty(penalised, new long[] { 0 }, penalty: 5f);
        Assert.Equal(1, LogitsProcessor.ArgMax(penalised));
        Assert.Equal(0.6f, penalised[0], 3); // 3 / 5
    }

    [Fact]
    public void Temperature_SharpensOrFlattensTheDistribution()
    {
        float[] logits = { 0f, 1f, 2f };
        float sharpMax = LogitsProcessor.Softmax(Apply(logits, 0.5f)).Max();
        float flatMax = LogitsProcessor.Softmax(Apply(logits, 2.0f)).Max();
        float baseMax = LogitsProcessor.Softmax(logits).Max();

        Assert.True(sharpMax > baseMax, "T<1 should sharpen (raise the peak probability).");
        Assert.True(flatMax < baseMax, "T>1 should flatten (lower the peak probability).");

        static float[] Apply(float[] l, float t) { var c = (float[])l.Clone(); LogitsProcessor.ApplyTemperature(c, t); return c; }
    }

    [Fact]
    public void TopK_KeepsExactlyKAndMasksTheRest()
    {
        float[] logits = { 1f, 2f, 3f, 4f, 5f };
        LogitsProcessor.ApplyTopK(logits, topK: 2);

        Assert.Equal(5f, logits[4]);
        Assert.Equal(4f, logits[3]);
        Assert.True(float.IsNegativeInfinity(logits[0]));
        Assert.True(float.IsNegativeInfinity(logits[1]));
        Assert.True(float.IsNegativeInfinity(logits[2]));
    }

    [Fact]
    public void TopP_KeepsTheNucleusAndMasksTheTail()
    {
        // softmax([3,2,1,0]) ~ [0.644, 0.237, 0.087, 0.032]; cumulative reaches 0.7 at token 1.
        float[] logits = { 3f, 2f, 1f, 0f };
        LogitsProcessor.ApplyTopP(logits, topP: 0.7f);

        Assert.Equal(3f, logits[0]);
        Assert.Equal(2f, logits[1]);
        Assert.True(float.IsNegativeInfinity(logits[2]));
        Assert.True(float.IsNegativeInfinity(logits[3]));
    }

    // ----------------------------------------------------------------------------------------
    // LogitsProcessor: SelectNextToken — each knob measurably changes the chosen token
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void SelectNextToken_Greedy_ReturnsArgmax()
    {
        float[] logits = { 1f, 5f, 2f };
        int chosen = LogitsProcessor.SelectNextToken(logits, new GenerationConfig { DoSample = false },
            Array.Empty<long>(), random: null);
        Assert.Equal(1, chosen);
    }

    [Fact]
    public void SelectNextToken_RepetitionPenalty_ChangesGreedyChoice()
    {
        var cfgPlain = new GenerationConfig { DoSample = false };
        var cfgPenalty = new GenerationConfig { DoSample = false, RepetitionPenalty = 5f };

        Assert.Equal(0, LogitsProcessor.SelectNextToken(new float[] { 3f, 2f, 1f, 0f }, cfgPlain, new long[] { 0 }, null));
        Assert.Equal(1, LogitsProcessor.SelectNextToken(new float[] { 3f, 2f, 1f, 0f }, cfgPenalty, new long[] { 0 }, null));
    }

    [Fact]
    public void SelectNextToken_Temperature_ChangesSampledToken()
    {
        // With a fixed draw of 0.5 and logits [3,2,1,0]: T=1 -> token 0, T=5 -> token 1.
        var ctx = Array.Empty<long>();
        int t1 = LogitsProcessor.SelectNextToken(new float[] { 3f, 2f, 1f, 0f },
            new GenerationConfig { DoSample = true, Temperature = 1f }, ctx, new FixedRandom(0.5));
        int t5 = LogitsProcessor.SelectNextToken(new float[] { 3f, 2f, 1f, 0f },
            new GenerationConfig { DoSample = true, Temperature = 5f }, ctx, new FixedRandom(0.5));

        Assert.Equal(0, t1);
        Assert.Equal(1, t5);
        Assert.NotEqual(t1, t5);
    }

    [Fact]
    public void SelectNextToken_TopKAndTopP_ChangeSampledToken()
    {
        // Fixed draw 0.95 over logits [3,2,1,0]:
        //   no filter  -> token 2
        //   Top-K = 1  -> only the argmax survives -> token 0
        //   Top-P=0.7  -> nucleus {0,1} -> token 1
        var ctx = Array.Empty<long>();
        int plain = LogitsProcessor.SelectNextToken(new float[] { 3f, 2f, 1f, 0f },
            new GenerationConfig { DoSample = true }, ctx, new FixedRandom(0.95));
        int topK = LogitsProcessor.SelectNextToken(new float[] { 3f, 2f, 1f, 0f },
            new GenerationConfig { DoSample = true, TopK = 1 }, ctx, new FixedRandom(0.95));
        int topP = LogitsProcessor.SelectNextToken(new float[] { 3f, 2f, 1f, 0f },
            new GenerationConfig { DoSample = true, TopP = 0.7f }, ctx, new FixedRandom(0.95));

        Assert.Equal(2, plain);
        Assert.Equal(0, topK);
        Assert.Equal(1, topP);
    }

    // ----------------------------------------------------------------------------------------
    // Seeded sampling reproducibility
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void SeededSampling_IsReproducible_AndSeedDependent()
    {
        // Uniform logits => the chosen token is driven entirely by the seeded RNG.
        Func<int, float[]> uniform = _ => new float[10];
        var config1 = new GenerationConfig { MaxNewTokens = 8, DoSample = true, Seed = 1 };

        var gen = new TextGenerator(NoCacheEngine(uniform));
        long[] runA = gen.Generate(new long[] { 0 }, config1).ToArray();
        long[] runB = gen.Generate(new long[] { 0 }, config1).ToArray();
        Assert.Equal(runA, runB); // same seed -> identical sequence

        long[] runDifferentSeed = new TextGenerator(NoCacheEngine(uniform))
            .Generate(new long[] { 0 }, config1 with { Seed = 2 }).ToArray();
        Assert.NotEqual(runA, runDifferentSeed); // different seed -> different sequence
    }

    // ----------------------------------------------------------------------------------------
    // KV-cache mode
    // ----------------------------------------------------------------------------------------

    private static FakeEngine KvCacheEngine(Func<int, float[]> logits)
    {
        var inputs = new[]
        {
            In("input_ids"),
            In("attention_mask"),
            In("position_ids"),
            In("past_key_values.0.key", ElementType.Float32, new[] { 1, 2, 1, 4 }),   // seq axis (2) -> 0 on first pass
            In("past_key_values.0.value", ElementType.Float32, new[] { 1, 2, 1, 4 }),
        };
        var outputs = new[] { Out("logits"), Out("present.0.key"), Out("present.0.value") };
        return new FakeEngine(inputs, outputs, logits);
    }

    [Fact]
    public void KvCache_IsDetected_FromInputNames()
    {
        Assert.True(new TextGenerator(KvCacheEngine(_ => OneHot(8, 7))).UsesKvCache);
        Assert.False(new TextGenerator(NoCacheEngine(_ => OneHot(8, 7))).UsesKvCache);
    }

    [Fact]
    public void KvCache_FeedsFullPromptThenSingleTokens_AndThreadsPresentToPast()
    {
        var engine = KvCacheEngine(_ => OneHot(8, 7));
        var generator = new TextGenerator(engine);

        generator.Generate(new long[] { 10, 11, 12 }, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        Assert.Equal(4, engine.Records.Count);

        // First pass: whole 3-token prompt, empty past (seq length 0).
        Assert.Equal(3, engine.Records[0].InputIdsLength);
        Assert.Equal(0, engine.Records[0].PastSequenceLength);

        // Subsequent passes: exactly one new token, with the threaded past growing each step.
        Assert.Equal(1, engine.Records[1].InputIdsLength);
        Assert.Equal(1, engine.Records[2].InputIdsLength);
        Assert.Equal(1, engine.Records[3].InputIdsLength);
        Assert.Equal(new[] { 0, 3, 4, 5 }, engine.Records.Select(r => r.PastSequenceLength).ToArray());

        // attention_mask covers past + current chunk every step.
        Assert.Equal(new[] { 3, 4, 5, 6 }, engine.Records.Select(r => r.AttentionMaskLength).ToArray());

        // position_ids start at 0 for the prompt then point at the single new position.
        Assert.Equal(new long[] { 0, 1, 2 }, engine.Records[0].Positions);
        Assert.Equal(new long[] { 3 }, engine.Records[1].Positions);
        Assert.Equal(new long[] { 4 }, engine.Records[2].Positions);
        Assert.Equal(new long[] { 5 }, engine.Records[3].Positions);

        // Past KV inputs are present on every step.
        Assert.All(engine.Records, r => Assert.Contains("past_key_values.0.key", r.FeedNames));
    }

    [Fact]
    public void NoCache_ReFeedsTheWholeGrowingSequence()
    {
        var engine = NoCacheEngine(_ => OneHot(8, 7), rank2: true); // also exercises [batch, vocab] logits
        var generator = new TextGenerator(engine);

        generator.Generate(new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 3, DoSample = false });

        Assert.Equal(3, engine.Records.Count);
        Assert.Equal(new[] { 2, 3, 4 }, engine.Records.Select(r => r.InputIdsLength).ToArray());
        // No attention_mask / position_ids were declared, so none were fed.
        Assert.All(engine.Records, r => Assert.Equal(new[] { "input_ids" }, r.FeedNames));
    }

    [Fact]
    public void KvCache_CanResolveEmptyPast_FromExplicitHeadDims_WhenShapeUnknown()
    {
        // Engine declares no KV shapes (like ManagedCpuEngine); head dims come from options.
        var inputs = new[]
        {
            In("input_ids"),
            In("past_key_values.0.key", ElementType.Float32),
            In("past_key_values.0.value", ElementType.Float32),
        };
        var outputs = new[] { Out("logits"), Out("present.0.key"), Out("present.0.value") };
        var engine = new FakeEngine(inputs, outputs, _ => OneHot(8, 7));

        var generator = new TextGenerator(engine,
            new DecoderModelOptions { KvCacheNumHeads = 2, KvCacheHeadDim = 4 });

        generator.Generate(new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 2, DoSample = false });

        Assert.True(generator.UsesKvCache);
        Assert.Equal(0, engine.Records[0].PastSequenceLength); // empty past built from [1,2,0,4]
        Assert.Equal(2, engine.Records[1].PastSequenceLength); // threaded present (prompt of length 2)
    }

    // ----------------------------------------------------------------------------------------
    // use_cache_branch (Optimum "merged" decoder exports)
    // ----------------------------------------------------------------------------------------

    /// <summary>
    /// A "merged" decoder engine: KV-cache inputs/outputs plus a <c>use_cache_branch</c> input of the
    /// requested dtype. The empty past is still declared so the first pass can bind it.
    /// </summary>
    private static FakeEngine MergedEngine(ElementType branchDtype, Func<int, float[]> logits)
    {
        var inputs = new[]
        {
            In("input_ids"),
            In("attention_mask"),
            In("use_cache_branch", branchDtype, new[] { 1 }),
            In("past_key_values.0.key", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.value", ElementType.Float32, new[] { 1, 2, 1, 4 }),
        };
        var outputs = new[] { Out("logits"), Out("present.0.key"), Out("present.0.value") };
        return new FakeEngine(inputs, outputs, logits);
    }

    [Fact]
    public void UseCacheBranch_IsDetected_FromInputNames()
    {
        Assert.True(new TextGenerator(MergedEngine(ElementType.Boolean, _ => OneHot(8, 7))).FeedsUseCacheBranch);
        Assert.False(new TextGenerator(KvCacheEngine(_ => OneHot(8, 7))).FeedsUseCacheBranch);
        Assert.False(new TextGenerator(NoCacheEngine(_ => OneHot(8, 7))).FeedsUseCacheBranch);
    }

    [Fact]
    public void UseCacheBranch_IsFalseOnPrefill_AndTrueOnCachedSteps()
    {
        var engine = MergedEngine(ElementType.Boolean, _ => OneHot(8, 7));
        var generator = new TextGenerator(engine);

        generator.Generate(new long[] { 10, 11, 12 }, new GenerationConfig { MaxNewTokens = 3, DoSample = false });

        Assert.Equal(3, engine.Records.Count);
        // Prefill (past == null) selects the no-past branch; every subsequent cached step selects it.
        Assert.Equal(new bool?[] { false, true, true }, engine.Records.Select(r => r.UseCacheBranch).ToArray());

        // Empty past is still fed on the prefill pass so the merged graph's bindings all exist.
        Assert.Equal(0, engine.Records[0].PastSequenceLength);
        Assert.Contains("past_key_values.0.key", engine.Records[0].FeedNames);
        Assert.Contains("use_cache_branch", engine.Records[0].FeedNames);
    }

    [Fact]
    public void UseCacheBranch_AdaptsToInt64Dtype()
    {
        // When the model declares use_cache_branch as int64, 0/1 are fed in that dtype. A recording
        // engine captures the actual fed tensor so we can assert both its dtype and its values.
        Tensor? prefillFeed = null;
        Tensor? cachedFeed = null;
        var inputs = new[]
        {
            In("input_ids"),
            In("use_cache_branch", ElementType.Int64, new[] { 1 }),
            In("past_key_values.0.key", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.value", ElementType.Float32, new[] { 1, 2, 1, 4 }),
        };
        var outputs = new[] { Out("logits"), Out("present.0.key"), Out("present.0.value") };
        var engine = new RecordingBranchEngine(inputs, outputs, _ => OneHot(8, 7),
            (call, feed) => { if (call == 0) prefillFeed = feed; else cachedFeed ??= feed; });

        new TextGenerator(engine).Generate(new long[] { 5 }, new GenerationConfig { MaxNewTokens = 2, DoSample = false });

        Assert.IsType<Tensor<long>>(prefillFeed);
        Assert.IsType<Tensor<long>>(cachedFeed);
        Assert.Equal(0L, ((Tensor<long>)prefillFeed!).Span[0]);   // false on prefill
        Assert.Equal(1L, ((Tensor<long>)cachedFeed!).Span[0]);    // true on cached step
    }

    /// <summary>A KV-cache engine that forwards the use_cache_branch feed to a callback for dtype checks.</summary>
    private sealed class RecordingBranchEngine : IExecutionEngine
    {
        private readonly FakeEngine _inner;
        private readonly Action<int, Tensor> _onBranch;
        private int _call;

        public RecordingBranchEngine(
            IReadOnlyList<TensorInfo> inputs, IReadOnlyList<TensorInfo> outputs,
            Func<int, float[]> logits, Action<int, Tensor> onBranch)
        {
            _inner = new FakeEngine(inputs, outputs, logits);
            _onBranch = onBranch;
        }

        public IReadOnlyList<TensorInfo> Inputs => _inner.Inputs;
        public IReadOnlyList<TensorInfo> Outputs => _inner.Outputs;

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            if (feeds.TryGetValue("use_cache_branch", out NamedTensor? b)) _onBranch(_call, b.Tensor);
            _call++;
            return _inner.Run(feeds);
        }

        public void Dispose() => _inner.Dispose();
    }

    [Fact]
    public void Generate_EmptyPrompt_Throws()
    {
        var generator = new TextGenerator(NoCacheEngine(_ => OneHot(8, 7)));
        Assert.Throws<ArgumentException>(() =>
            generator.Generate(Array.Empty<long>(), new GenerationConfig { MaxNewTokens = 1 }));
    }
}
