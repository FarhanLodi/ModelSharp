using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using ModelSharp.Audio;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Manifest;
using ModelSharp.Pipeline;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Deterministic Whisper decode-wiring tests against fake encoder/decoder engines: the forced language/task
/// prompt is applied to the decoder, the encoder runs exactly once over the mel features, EOS stops
/// generation, and the produced ids decode to text with special tokens stripped. No real model required.
/// </summary>
public class WhisperPipelineTests
{
    // ---- fakes (mirror Seq2SeqGeneratorTests) ----

    /// <summary>Records the encoder feature input it saw; returns a fixed [1, encLen, hidden] hidden-state.</summary>
    private sealed class FakeAudioEncoder : IExecutionEngine
    {
        private readonly int _encLen, _hidden;
        public List<int[]> FeatureShapes { get; } = new();
        public int Calls { get; private set; }
        public IReadOnlyList<TensorInfo> Inputs { get; }
        public IReadOnlyList<TensorInfo> Outputs { get; }

        public FakeAudioEncoder(int encLen = 5, int hidden = 4)
        {
            _encLen = encLen; _hidden = hidden;
            Inputs = new[] { new TensorInfo("input_features", ElementType.Float32, new[] { 1, 80, 3000 }) };
            Outputs = new[] { new TensorInfo("last_hidden_state", ElementType.Float32, Array.Empty<int>()) };
        }

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            Calls++;
            FeatureShapes.Add(feeds["input_features"].Tensor.Shape.Dimensions.ToArray());
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["last_hidden_state"] = new NamedTensor("last_hidden_state", new Tensor<float>(new TensorShape(1, _encLen, _hidden))),
            };
        }

        public void Dispose() { }
    }

    /// <summary>A token-id encoder (declares <c>input_ids</c>) for the seq2seq-text negative test.</summary>
    private sealed class FakeTextEncoder : IExecutionEngine
    {
        public IReadOnlyList<TensorInfo> Inputs { get; } = new[]
        {
            new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()),
            new TensorInfo("attention_mask", ElementType.Int64, Array.Empty<int>()),
        };
        public IReadOnlyList<TensorInfo> Outputs { get; } = new[]
        {
            new TensorInfo("last_hidden_state", ElementType.Float32, Array.Empty<int>()),
        };

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            int srcLen = feeds["input_ids"].Tensor.Shape.Dimensions[1];
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["last_hidden_state"] = new NamedTensor("last_hidden_state", new Tensor<float>(new TensorShape(1, srcLen, 4))),
            };
        }

        public void Dispose() { }
    }

    /// <summary>Scripted decoder: records the decoder input ids each step, returns one-hot logits per call.</summary>
    private sealed class FakeTextDecoder : IExecutionEngine
    {
        private readonly Func<int, int> _peak;
        private readonly int _vocab;
        private int _call;
        public List<long[]> InputIds { get; } = new();
        public IReadOnlyList<TensorInfo> Inputs { get; }
        public IReadOnlyList<TensorInfo> Outputs { get; }

        public FakeTextDecoder(int vocab, Func<int, int> peak)
        {
            _vocab = vocab; _peak = peak;
            Inputs = new[]
            {
                new TensorInfo("input_ids", ElementType.Int64, Array.Empty<int>()),
                new TensorInfo("encoder_hidden_states", ElementType.Float32, Array.Empty<int>()),
            };
            Outputs = new[] { new TensorInfo("logits", ElementType.Float32, Array.Empty<int>()) };
        }

        public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
        {
            InputIds.Add(feeds["input_ids"].Tensor.AsInt64().Span.ToArray());
            var row = new float[_vocab];
            row[_peak(_call)] = 10f;
            _call++;
            return new Dictionary<string, NamedTensor>(StringComparer.Ordinal)
            {
                ["logits"] = new NamedTensor("logits", Tensor<float>.FromArray(new TensorShape(1, 1, _vocab), row)),
            };
        }

        public void Dispose() { }
    }

    // ---- a tiny byte-level vocab/merges on disk so WhisperProcessor can build its tokenizer ----

    private sealed class TempTokenizer : IDisposable
    {
        public string Dir { get; }
        public string Vocab { get; }
        public string Merges { get; }
        public Dictionary<string, int> Map { get; }

        public TempTokenizer()
        {
            Dir = Path.Combine(Path.GetTempPath(), "modelsharp-whisper-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Dir);
            // Byte-level tokens for the lowercase letters used in our scripted output ("hi"), plus a couple
            // of placeholder ids so the vocab spans a small contiguous range.
            Map = new Dictionary<string, int>
            {
                ["h"] = 0,
                ["i"] = 1,
                ["Ġ"] = 2, // byte-level space marker
                ["hi"] = 3,
            };
            Vocab = Path.Combine(Dir, "vocab.json");
            Merges = Path.Combine(Dir, "merges.txt");
            File.WriteAllText(Vocab, JsonSerializer.Serialize(Map));
            File.WriteAllText(Merges, "#version: 0.2\nh i\n"); // merge "h"+"i" -> "hi"
        }

        public void Dispose()
        {
            try { Directory.Delete(Dir, recursive: true); } catch { /* best effort */ }
        }
    }

    private static ModelManifest WhisperManifest(TempTokenizer tok, int sot, int eot, int lang, int transcribe, int notimestamps)
        => new()
        {
            Task = ModelTask.SpeechToTextSeq2Seq,
            Extra = new Dictionary<string, string>
            {
                ["vocab"] = tok.Vocab,
                ["merges"] = tok.Merges,
                ["sot_id"] = sot.ToString(),
                ["eot_id"] = eot.ToString(),
                ["language"] = lang.ToString(),
                ["transcribe_id"] = transcribe.ToString(),
                ["notimestamps_id"] = notimestamps.ToString(),
                ["max_new_tokens"] = "16",
            },
        };

    // ---- tests ----

    [Fact]
    public void ForcedPrompt_HasSotLanguageTaskNoTimestamps_InOrder()
    {
        var w = new WhisperGenerationOptions
        {
            StartOfTranscriptId = 100, LanguageTokenId = 50, TranscribeTaskId = 60, NoTimestampsId = 70, NoTimestamps = true,
        };
        Assert.Equal(new long[] { 100, 50, 60, 70 }, w.BuildForcedPrompt().ToArray());

        // Translate task + timestamps kept: language present, translate token, no <|notimestamps|>.
        var t = w with { Translate = true, TranslateTaskId = 61, NoTimestamps = false };
        Assert.Equal(new long[] { 100, 50, 61 }, t.BuildForcedPrompt().ToArray());

        // English-only style: no language token.
        var noLang = w with { LanguageTokenId = null };
        Assert.Equal(new long[] { 100, 60, 70 }, noLang.BuildForcedPrompt().ToArray());
    }

    [Fact]
    public void Decode_StripsSpecialTokens_AndKeepsText()
    {
        using var tok = new TempTokenizer();
        // Special-token block sits above the text ids; eot is the cutoff at 50257 by default but here we
        // configure a tiny vocab, so set eot low enough that text ids (0..3) stay below it.
        var ctx = new ProcessorContext(WhisperManifest(tok, sot: 9, eot: 4, lang: 8, transcribe: 7, notimestamps: 6),
            new[] { "input_ids" }, new[] { "logits" });
        var proc = new WhisperProcessor(ctx);

        // ids: forced specials (>=4) interleaved with text "h","i" (0,1) → decodes to "hi".
        string text = proc.Decode(new long[] { 9, 8, 7, 6, 0, 1, 4 });
        Assert.Equal("hi", text);
    }

    [Fact]
    public void Transcribe_AppliesForcedPrompt_RunsEncoderOnce_StopsAtEos()
    {
        using var tok = new TempTokenizer();
        const int Sot = 9, Eot = 4, Lang = 8, Transcribe = 7, NoTs = 6;

        // Scripted decoder output: emit "h"(0), "i"(1), then EOS(4). Vocab must cover the special ids → 10.
        int[] peaks = { 0, 1, Eot, 0 };
        var enc = new FakeAudioEncoder(encLen: 5, hidden: 4);
        var dec = new FakeTextDecoder(vocab: 10, peak: c => peaks[c]);

        ModelManifest manifest = WhisperManifest(tok, Sot, Eot, Lang, Transcribe, NoTs);
        using var pipeline = ModelSharpPipeline.BuildWhisper(enc, dec, manifest);

        var waveform = new float[WhisperFeatureExtractor.SampleRate]; // 1 s of silence is fine for the fake
        string text = pipeline.Transcribe(waveform);

        Assert.Equal("hi", text);

        // Encoder ran exactly once, over a [1, 80, 3000] mel tensor.
        Assert.Equal(1, enc.Calls);
        Assert.Equal(new[] { 1, 80, 3000 }, enc.FeatureShapes[0]);

        // First decoder call is fed the whole forced prompt: <|sot|> <|lang|> <|transcribe|> <|notimestamps|>.
        Assert.Equal(new long[] { Sot, Lang, Transcribe, NoTs }, dec.InputIds[0]);

        // Three decode steps: emit h, i, EOS — then it stops (no 4th call).
        Assert.Equal(3, dec.InputIds.Count);
    }

    [Fact]
    public void Transcribe_RespectsMaxNewTokens_WhenNoEos()
    {
        using var tok = new TempTokenizer();
        const int Sot = 9, Eot = 99, Lang = 8, Transcribe = 7, NoTs = 6; // EOS never produced

        var enc = new FakeAudioEncoder();
        var dec = new FakeTextDecoder(vocab: 10, peak: _ => 0); // always emit "h"
        ModelManifest manifest = WhisperManifest(tok, Sot, Eot, Lang, Transcribe, NoTs);
        using var pipeline = ModelSharpPipeline.BuildWhisper(enc, dec, manifest);

        string text = pipeline.Transcribe(new float[WhisperFeatureExtractor.SampleRate],
            GenerationConfig.Greedy(maxNewTokens: 5, eosTokenIds: new[] { Eot }));

        Assert.Equal(5, dec.InputIds.Count);          // exactly maxNewTokens decode steps
        Assert.Equal("hhhhh", text);
    }

    [Fact]
    public void Transcribe_OnNonWhisperPipeline_Throws()
    {
        using var tok = new TempTokenizer();
        // A seq2seq (text) pipeline does not support Transcribe.
        var manifest = new ModelManifest
        {
            Task = ModelTask.Seq2SeqGeneration,
            Extra = new Dictionary<string, string> { ["vocab"] = tok.Vocab, ["merges"] = tok.Merges },
        };
        var enc = new FakeTextEncoder();
        // Reuse a text-style decoder; the seq2seq processor only needs input_ids on the decoder.
        var dec = new FakeTextDecoder(vocab: 10, peak: _ => 0);
        using var pipeline = ModelSharpPipeline.BuildSeq2Seq(enc, dec, manifest);

        Assert.Throws<ModelSharpException>(() => { pipeline.Transcribe(new float[16000]); });
    }
}
