using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Deterministic tests for the encoder-decoder generation loop. Everything runs against in-test fake
/// engines (one encoder, one decoder) that return scripted logits and record the feeds they see, so no
/// real model is required and the behaviour is fully reproducible.
/// </summary>
public class Seq2SeqGeneratorTests
{
    // ----------------------------------------------------------------------------------------
    // Fakes
    // ----------------------------------------------------------------------------------------

    /// <summary>Records one encoder invocation: the source length and whether a mask was fed.</summary>
    private sealed class EncoderCall
    {
        public int InputLength;
        public bool HadMask;
        public long[]? Mask;
    }

    /// <summary>
    /// Scripted encoder. Records every call and returns a fixed <c>last_hidden_state</c> of shape
    /// <c>[1, srcLen, hidden]</c>. Used to assert the encoder runs exactly once.
    /// </summary>
    private sealed class FakeEncoder : IExecutionEngine
    {
        private readonly int _hidden;
        public List<EncoderCall> Calls { get; } = new();
        public IReadOnlyList<TensorInfo> Inputs { get; }
        public IReadOnlyList<TensorInfo> Outputs { get; }

        public FakeEncoder(IReadOnlyList<TensorInfo> inputs, IReadOnlyList<TensorInfo> outputs, int hidden = 4)
        {
            Inputs = inputs;
            Outputs = outputs;
            _hidden = hidden;
        }

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            int srcLen = feeds["input_ids"].Tensor.Shape.Dimensions[1];
            Calls.Add(new EncoderCall
            {
                InputLength = srcLen,
                HadMask = feeds.ContainsKey("attention_mask"),
                Mask = feeds.ContainsKey("attention_mask") ? feeds["attention_mask"].Tensor.AsInt64().Span.ToArray() : null,
            });
            string outName = Outputs[0].Name;
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                [outName] = new NamedTensor(outName, new Tensor<float>(new TensorShape(1, srcLen, _hidden))),
            };
        }

        public void Dispose() { }
    }

    /// <summary>Captured snapshot of one decoder invocation.</summary>
    private sealed class DecoderCall
    {
        public string[] FeedNames = Array.Empty<string>();
        public int InputIdsLength;
        public bool HadEncoderHidden;
        public int EncoderHiddenSeq = -1;
        public long[]? EncoderMask;
        public int SelfPastSeq = -1;
        public int CrossPastSeq = -1;
        public bool? UseCacheBranch;
    }

    /// <summary>
    /// Scripted, recording decoder. Returns per-call logits, and (when configured with present.* outputs)
    /// fabricates self-attention KV that grows along the sequence axis and cross-attention KV sized to the
    /// encoder length. The self/cross split is by the ".decoder."/".encoder." infix.
    /// </summary>
    private sealed class FakeDecoder : IExecutionEngine
    {
        private readonly Func<int, float[]> _logits;
        private readonly int _encoderLen;
        private int _call;

        public List<DecoderCall> Calls { get; } = new();
        public IReadOnlyList<TensorInfo> Inputs { get; }
        public IReadOnlyList<TensorInfo> Outputs { get; }

        public FakeDecoder(IReadOnlyList<TensorInfo> inputs, IReadOnlyList<TensorInfo> outputs, Func<int, float[]> logits, int encoderLen)
        {
            Inputs = inputs;
            Outputs = outputs;
            _logits = logits;
            _encoderLen = encoderLen;
        }

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            var record = new DecoderCall
            {
                FeedNames = feeds.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray(),
                InputIdsLength = feeds["input_ids"].Tensor.Shape.Dimensions[1],
                HadEncoderHidden = feeds.ContainsKey("encoder_hidden_states"),
                EncoderHiddenSeq = feeds.ContainsKey("encoder_hidden_states")
                    ? feeds["encoder_hidden_states"].Tensor.Shape.Dimensions[1] : -1,
                EncoderMask = feeds.ContainsKey("encoder_attention_mask")
                    ? feeds["encoder_attention_mask"].Tensor.AsInt64().Span.ToArray() : null,
                SelfPastSeq = feeds.ContainsKey("past_key_values.0.decoder.key")
                    ? feeds["past_key_values.0.decoder.key"].Tensor.Shape.Dimensions[2] : -1,
                CrossPastSeq = feeds.ContainsKey("past_key_values.0.encoder.key")
                    ? feeds["past_key_values.0.encoder.key"].Tensor.Shape.Dimensions[2] : -1,
                UseCacheBranch = feeds.ContainsKey("use_cache_branch")
                    ? feeds["use_cache_branch"].Tensor.AsBool().Span[0] : (bool?)null,
            };
            Calls.Add(record);

            float[] row = _logits(_call);
            _call++;

            var result = new Dictionary<string, NamedTensor>(StringComparer.Ordinal);
            result["logits"] = new NamedTensor("logits",
                Tensor<float>.FromArray(new TensorShape(1, 1, row.Length), row));

            foreach (TensorInfo o in Outputs)
            {
                if (!o.Name.StartsWith("present.", StringComparison.Ordinal)) continue;
                if (o.Name.Contains(".encoder.", StringComparison.Ordinal))
                {
                    // Cross-attention present: sized to the encoder length, constant across steps.
                    result[o.Name] = new NamedTensor(o.Name, new Tensor<float>(new TensorShape(1, 2, _encoderLen, 4)));
                }
                else
                {
                    // Self-attention present: grows by the fed chunk length each step.
                    int pastSeq = record.SelfPastSeq < 0 ? 0 : record.SelfPastSeq;
                    int seq = pastSeq + record.InputIdsLength;
                    result[o.Name] = new NamedTensor(o.Name, new Tensor<float>(new TensorShape(1, 2, seq, 4)));
                }
            }
            return result;
        }

        public void Dispose() { }
    }

    private static TensorInfo In(string name, ElementType dt = ElementType.Int64, int[]? dims = null) =>
        new(name, dt, dims ?? Array.Empty<int>());

    private static TensorInfo Out(string name) => new(name, ElementType.Float32, Array.Empty<int>());

    private static float[] OneHot(int vocab, int peak)
    {
        var row = new float[vocab];
        row[peak] = 10f;
        return row;
    }

    // ---- engine builders ----

    private static FakeEncoder Encoder(int hidden = 4) =>
        new(new[] { In("input_ids"), In("attention_mask") }, new[] { Out("last_hidden_state") }, hidden);

    /// <summary>A no-cache decoder: re-fed the whole growing sequence each step.</summary>
    private static FakeDecoder NoCacheDecoder(Func<int, float[]> logits, int encoderLen)
    {
        var inputs = new[]
        {
            In("input_ids"),
            In("encoder_hidden_states"),
            In("encoder_attention_mask"),
        };
        var outputs = new[] { Out("logits") };
        return new FakeDecoder(inputs, outputs, logits, encoderLen);
    }

    /// <summary>A decoder-with-past decoder: separate self + cross attention KV caches.</summary>
    private static FakeDecoder KvCacheDecoder(Func<int, float[]> logits, int encoderLen)
    {
        var inputs = new[]
        {
            In("input_ids"),
            In("encoder_hidden_states"),
            In("encoder_attention_mask"),
            In("past_key_values.0.decoder.key", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.decoder.value", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.encoder.key", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.encoder.value", ElementType.Float32, new[] { 1, 2, 1, 4 }),
        };
        var outputs = new[]
        {
            Out("logits"),
            Out("present.0.decoder.key"), Out("present.0.decoder.value"),
            Out("present.0.encoder.key"), Out("present.0.encoder.value"),
        };
        return new FakeDecoder(inputs, outputs, logits, encoderLen);
    }

    // ----------------------------------------------------------------------------------------
    // Encoder-once + greedy correctness
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void Encoder_RunsExactlyOnce_NoCachePath()
    {
        var enc = Encoder();
        var dec = NoCacheDecoder(_ => OneHot(8, 7), encoderLen: 3);
        var gen = new Seq2SeqGenerator(enc, dec);

        gen.Generate(new long[] { 5, 6, 7 }, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        Assert.Single(enc.Calls);
        Assert.Equal(3, enc.Calls[0].InputLength);
        Assert.True(enc.Calls[0].HadMask);
    }

    [Fact]
    public void Encoder_RunsExactlyOnce_KvCachePath()
    {
        var enc = Encoder();
        var dec = KvCacheDecoder(_ => OneHot(8, 7), encoderLen: 3);
        var gen = new Seq2SeqGenerator(enc, dec);

        gen.Generate(new long[] { 5, 6, 7 }, new GenerationConfig { MaxNewTokens = 5, DoSample = false });

        Assert.Single(enc.Calls);
    }

    [Fact]
    public void Greedy_PicksArgmax_EachStep_AndExcludesDecoderStartToken()
    {
        int[] peaks = { 3, 1, 4, 2 };
        var gen = new Seq2SeqGenerator(Encoder(), NoCacheDecoder(c => OneHot(8, peaks[c]), 2));

        IReadOnlyList<long> outIds = gen.Generate(
            new long[] { 9, 9 }, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        // The decoder start token (0) seeds the loop but is never emitted.
        Assert.Equal(new long[] { 3, 1, 4, 2 }, outIds.ToArray());
    }

    [Fact]
    public void Greedy_StopsExactlyAtEos_IncludingTheEosToken()
    {
        int[] peaks = { 3, 1, 4, 9 }; // 4 is EOS; token 9 must never be produced
        var dec = NoCacheDecoder(c => OneHot(10, peaks[c]), 2);
        var gen = new Seq2SeqGenerator(Encoder(), dec);

        IReadOnlyList<long> outIds = gen.Generate(
            new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 50, DoSample = false, EosTokenIds = new[] { 4 } });

        Assert.Equal(new long[] { 3, 1, 4 }, outIds.ToArray());
        Assert.Equal(3, dec.Calls.Count); // stopped right after EOS, no extra decode
    }

    [Fact]
    public void Greedy_RespectsMaxNewTokens()
    {
        var gen = new Seq2SeqGenerator(Encoder(), NoCacheDecoder(_ => OneHot(8, 7), 2));

        IReadOnlyList<long> outIds = gen.Generate(
            new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 5, DoSample = false });

        Assert.Equal(5, outIds.Count);
        Assert.All(outIds, t => Assert.Equal(7, t));
    }

    // ----------------------------------------------------------------------------------------
    // Encoder hidden states + cross-attention mask threading
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void Decoder_ReceivesEncoderHiddenStates_AndEncoderMask_EveryStep()
    {
        var dec = NoCacheDecoder(_ => OneHot(8, 7), encoderLen: 3);
        var gen = new Seq2SeqGenerator(Encoder(), dec);

        gen.Generate(new long[] { 5, 6, 7 }, new GenerationConfig { MaxNewTokens = 3, DoSample = false });

        Assert.Equal(3, dec.Calls.Count);
        Assert.All(dec.Calls, c =>
        {
            Assert.True(c.HadEncoderHidden);
            Assert.Equal(3, c.EncoderHiddenSeq);                       // encoder hidden states fed every step
            Assert.Equal(new long[] { 1, 1, 1 }, c.EncoderMask);       // cross-attn mask covers all source tokens
        });
    }

    [Fact]
    public void SourceAttentionMask_IsForwardedToEncoderAndCrossAttention()
    {
        var enc = Encoder();
        var dec = NoCacheDecoder(_ => OneHot(8, 7), encoderLen: 3);
        var gen = new Seq2SeqGenerator(enc, dec);

        // Third source token is padding.
        gen.Generate(new long[] { 5, 6, 0 }, new GenerationConfig { MaxNewTokens = 2, DoSample = false },
            sourceAttentionMask: new long[] { 1, 1, 0 });

        Assert.Equal(new long[] { 1, 1, 0 }, enc.Calls[0].Mask);
        Assert.All(dec.Calls, c => Assert.Equal(new long[] { 1, 1, 0 }, c.EncoderMask));
    }

    // ----------------------------------------------------------------------------------------
    // KV cache: self-attn grows, cross-attn captured once and constant
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void KvCache_IsDetected_AndSplitsSelfAndCross()
    {
        var gen = new Seq2SeqGenerator(Encoder(), KvCacheDecoder(_ => OneHot(8, 7), 3));
        Assert.True(gen.UsesKvCache);
        Assert.True(gen.HasCrossAttentionCache);

        var noCache = new Seq2SeqGenerator(Encoder(), NoCacheDecoder(_ => OneHot(8, 7), 3));
        Assert.False(noCache.UsesKvCache);
        Assert.False(noCache.HasCrossAttentionCache);
    }

    [Fact]
    public void KvCache_SelfAttentionGrows_CrossAttentionStaysConstant()
    {
        var dec = KvCacheDecoder(_ => OneHot(8, 7), encoderLen: 3);
        var gen = new Seq2SeqGenerator(Encoder(), dec);

        // Decoder starts from the start token (1 token), so prefill self-past = 0, then grows 1,2,3.
        gen.Generate(new long[] { 5, 6, 7 }, new GenerationConfig { MaxNewTokens = 4, DoSample = false });

        Assert.Equal(4, dec.Calls.Count);

        // First pass feeds just the start token; subsequent passes feed one new token.
        Assert.Equal(1, dec.Calls[0].InputIdsLength);
        Assert.Equal(1, dec.Calls[1].InputIdsLength);

        // Self-attention past starts empty (0) then grows by 1 each step as the decoder sequence extends.
        Assert.Equal(new[] { 0, 1, 2, 3 }, dec.Calls.Select(c => c.SelfPastSeq).ToArray());

        // Cross-attention past is empty on the prefill pass (the graph computes it), then the captured
        // first-pass present (sized to the encoder length 3) is re-fed unchanged every later step.
        Assert.Equal(new[] { 0, 3, 3, 3 }, dec.Calls.Select(c => c.CrossPastSeq).ToArray());
    }

    [Fact]
    public void KvCache_ResolvesEmptyPast_FromExplicitHeadDims_WhenShapeUnknown()
    {
        // Decoder declares no KV shapes (like ManagedCpuEngine); head dims come from options.
        var inputs = new[]
        {
            In("input_ids"),
            In("encoder_hidden_states"),
            In("encoder_attention_mask"),
            In("past_key_values.0.decoder.key", ElementType.Float32),
            In("past_key_values.0.decoder.value", ElementType.Float32),
            In("past_key_values.0.encoder.key", ElementType.Float32),
            In("past_key_values.0.encoder.value", ElementType.Float32),
        };
        var outputs = new[]
        {
            Out("logits"),
            Out("present.0.decoder.key"), Out("present.0.decoder.value"),
            Out("present.0.encoder.key"), Out("present.0.encoder.value"),
        };
        var dec = new FakeDecoder(inputs, outputs, _ => OneHot(8, 7), encoderLen: 2);
        var gen = new Seq2SeqGenerator(Encoder(), dec,
            new Seq2SeqModelOptions { KvCacheNumHeads = 2, KvCacheHeadDim = 4 });

        gen.Generate(new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 2, DoSample = false });

        Assert.True(gen.UsesKvCache);
        Assert.Equal(0, dec.Calls[0].SelfPastSeq);   // empty past built from [1,2,0,4]
        Assert.Equal(1, dec.Calls[1].SelfPastSeq);   // threaded present (start token of length 1)
    }

    // ----------------------------------------------------------------------------------------
    // use_cache_branch (merged decoder export)
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void UseCacheBranch_IsFalseOnPrefill_AndTrueOnCachedSteps()
    {
        var inputs = new[]
        {
            In("input_ids"),
            In("encoder_hidden_states"),
            In("encoder_attention_mask"),
            In("use_cache_branch", ElementType.Boolean, new[] { 1 }),
            In("past_key_values.0.decoder.key", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.decoder.value", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.encoder.key", ElementType.Float32, new[] { 1, 2, 1, 4 }),
            In("past_key_values.0.encoder.value", ElementType.Float32, new[] { 1, 2, 1, 4 }),
        };
        var outputs = new[]
        {
            Out("logits"),
            Out("present.0.decoder.key"), Out("present.0.decoder.value"),
            Out("present.0.encoder.key"), Out("present.0.encoder.value"),
        };
        var dec = new FakeDecoder(inputs, outputs, _ => OneHot(8, 7), encoderLen: 2);
        var gen = new Seq2SeqGenerator(Encoder(), dec);

        Assert.True(gen.FeedsUseCacheBranch);
        gen.Generate(new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 3, DoSample = false });

        Assert.Equal(new bool?[] { false, true, true }, dec.Calls.Select(c => c.UseCacheBranch).ToArray());
    }

    // ----------------------------------------------------------------------------------------
    // Validation
    // ----------------------------------------------------------------------------------------

    [Fact]
    public void Generate_EmptySource_Throws()
    {
        var gen = new Seq2SeqGenerator(Encoder(), NoCacheDecoder(_ => OneHot(8, 7), 1));
        Assert.Throws<ArgumentException>(() =>
            gen.Generate(Array.Empty<long>(), new GenerationConfig { MaxNewTokens = 1 }));
    }

    [Fact]
    public void Generate_MismatchedMaskLength_Throws()
    {
        var gen = new Seq2SeqGenerator(Encoder(), NoCacheDecoder(_ => OneHot(8, 7), 2));
        Assert.Throws<ArgumentException>(() =>
            gen.Generate(new long[] { 1, 2 }, new GenerationConfig { MaxNewTokens = 1 }, sourceAttentionMask: new long[] { 1 }));
    }

    [Fact]
    public void Encoder_MissingInput_Throws()
    {
        var enc = new FakeEncoder(new[] { In("pixel_values") }, new[] { Out("last_hidden_state") });
        var dec = NoCacheDecoder(_ => OneHot(8, 7), 2);
        Assert.Throws<ModelSharpException>(() => new Seq2SeqGenerator(enc, dec));
    }
}
