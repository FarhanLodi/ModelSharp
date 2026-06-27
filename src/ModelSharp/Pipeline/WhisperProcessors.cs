using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ModelSharp.Audio;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Tensors;
using ModelSharp.Text;

namespace ModelSharp.Pipeline;

/// <summary>
/// Whisper decoding conventions: the forced decoder prompt and the special-token ids the loop needs.
/// Whisper seeds the decoder with <c>&lt;|startoftranscript|&gt;</c>, a language token, the task token
/// (<c>&lt;|transcribe|&gt;</c> or <c>&lt;|translate|&gt;</c>) and — when timestamps are disabled —
/// <c>&lt;|notimestamps|&gt;</c>, then generates until <c>&lt;|endoftext|&gt;</c> (EOS). These ids are
/// fixed for a given Whisper vocab; defaults below match the standard multilingual checkpoints
/// (tiny/base/small/medium/large) where the special-token block starts at 50257.
/// </summary>
public sealed record WhisperGenerationOptions
{
    /// <summary><c>&lt;|startoftranscript|&gt;</c> — the decoder start token (also <c>decoder_start_token_id</c>). Default 50258.</summary>
    public int StartOfTranscriptId { get; init; } = 50258;

    /// <summary><c>&lt;|endoftext|&gt;</c> — EOS (and pad). Default 50257.</summary>
    public int EndOfTextId { get; init; } = 50257;

    /// <summary><c>&lt;|transcribe|&gt;</c> task token. Default 50359.</summary>
    public int TranscribeTaskId { get; init; } = 50359;

    /// <summary><c>&lt;|translate|&gt;</c> task token. Default 50358.</summary>
    public int TranslateTaskId { get; init; } = 50358;

    /// <summary><c>&lt;|notimestamps|&gt;</c> token. Default 50363.</summary>
    public int NoTimestampsId { get; init; } = 50363;

    /// <summary>
    /// The language token id (e.g. <c>&lt;|en|&gt;</c> = 50259, <c>&lt;|de|&gt;</c> = 50261). When null the
    /// language token is omitted from the prompt (English-only checkpoints have no language tokens; some
    /// callers also leave it out to let the model auto-detect, though that is not standard). Default
    /// <c>&lt;|en|&gt;</c> (50259).
    /// </summary>
    public int? LanguageTokenId { get; init; } = 50259;

    /// <summary>True to translate to English (<c>&lt;|translate|&gt;</c>); false to transcribe in-language. Default false.</summary>
    public bool Translate { get; init; }

    /// <summary>True to suppress timestamp tokens via <c>&lt;|notimestamps|&gt;</c> in the prompt. Default true.</summary>
    public bool NoTimestamps { get; init; } = true;

    /// <summary>Number of mel bins (80 for tiny…large-v2, 128 for large-v3). Default 80.</summary>
    public int NumMels { get; init; } = 80;

    /// <summary>Maximum new tokens to generate per clip. Default 224 (Whisper's per-window budget).</summary>
    public int MaxNewTokens { get; init; } = 224;

    /// <summary>
    /// Builds the forced decoder prompt: <c>&lt;|sot|&gt; [lang] &lt;|task|&gt; [&lt;|notimestamps|&gt;]</c>.
    /// This whole sequence seeds the decoder and is never emitted.
    /// </summary>
    public IReadOnlyList<long> BuildForcedPrompt()
    {
        var ids = new List<long> { StartOfTranscriptId };
        if (LanguageTokenId is int lang) ids.Add(lang);
        ids.Add(Translate ? TranslateTaskId : TranscribeTaskId);
        if (NoTimestamps) ids.Add(NoTimestampsId);
        return ids;
    }
}

/// <summary>
/// Builds the Whisper speech-to-text stack: the <see cref="WhisperFeatureExtractor"/> (PCM → log-mel), a
/// byte-level <see cref="BpeTokenizer"/> for decoding token ids → text, the <see cref="WhisperGenerationOptions"/>
/// forced-prompt conventions, and the <see cref="Seq2SeqModelOptions"/> wired for an audio encoder
/// (<c>input_features</c>) plus the standard Whisper decoder KV-cache layout. Like the seq2seq/text-generation
/// processors it does not implement the single-shot pre/post interfaces — transcription is an encoder-once +
/// autoregressive-decode loop — so it is consumed directly by <see cref="ModelSharpPipeline.BuildWhisper"/>.
///
/// <para>Manifest hints (under <c>Manifest.Extra</c>):</para>
/// <list type="bullet">
///   <item><description><c>vocab</c>, <c>merges</c> — Whisper tokenizer files (required for text output).</description></item>
///   <item><description><c>n_mels</c> — 80 (default) or 128 (large-v3).</description></item>
///   <item><description><c>language</c> — language token id (e.g. 50259 for English) or <c>none</c> to omit it.</description></item>
///   <item><description><c>task</c> — <c>transcribe</c> (default) or <c>translate</c>.</description></item>
///   <item><description><c>timestamps</c> — <c>true</c> to keep timestamp tokens (default false → forces <c>&lt;|notimestamps|&gt;</c>).</description></item>
///   <item><description><c>sot_id</c>, <c>eot_id</c>, <c>transcribe_id</c>, <c>translate_id</c>, <c>notimestamps_id</c> — override special-token ids.</description></item>
///   <item><description><c>encoder_input</c> — encoder feature input name (default <c>input_features</c>).</description></item>
///   <item><description><c>kv_num_heads</c>, <c>kv_head_dim</c> — KV-cache head dims when the engine reports no past-KV shape.</description></item>
///   <item><description><c>max_new_tokens</c> — generation budget (default 224).</description></item>
/// </list>
/// </summary>
public sealed class WhisperProcessor
{
    /// <summary>The Whisper log-mel feature extractor (configured for the manifest's mel-bin count).</summary>
    public WhisperFeatureExtractor FeatureExtractor { get; }

    /// <summary>The byte-level BPE tokenizer used to decode generated ids to text.</summary>
    public BpeTokenizer Tokenizer { get; }

    /// <summary>Whisper forced-prompt + special-token conventions.</summary>
    public WhisperGenerationOptions Whisper { get; }

    /// <summary>The encoder-decoder IO-binding options (audio encoder input name, KV-cache, start token).</summary>
    public Seq2SeqModelOptions Options { get; }

    /// <summary>The default decoding configuration (greedy, EOS = <c>&lt;|endoftext|&gt;</c>).</summary>
    public GenerationConfig DefaultConfig { get; }

    /// <summary>Builds the Whisper processor stack from a processor context (manifest + engine binding names).</summary>
    public WhisperProcessor(ProcessorContext ctx)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));
        IReadOnlyDictionary<string, string> extra = ctx.Manifest.Extra;

        if (!extra.TryGetValue("vocab", out string? vocabPath) || string.IsNullOrWhiteSpace(vocabPath))
            throw new ModelSharpException(
                "Whisper transcription requires a BPE vocab; set manifest Extra[\"vocab\"] to a vocab.json path.");
        if (!extra.TryGetValue("merges", out string? mergesPath) || string.IsNullOrWhiteSpace(mergesPath))
            throw new ModelSharpException(
                "Whisper transcription requires BPE merges; set manifest Extra[\"merges\"] to a merges.txt path.");
        if (!File.Exists(vocabPath))
            throw new ModelSharpException($"Vocab file not found for Whisper: '{vocabPath}'.");
        if (!File.Exists(mergesPath))
            throw new ModelSharpException($"Merges file not found for Whisper: '{mergesPath}'.");

        Tokenizer = BpeTokenizer.FromFiles(vocabPath!, mergesPath!);

        int nMels = TryGetInt(extra, "n_mels") ?? 80;
        FeatureExtractor = new WhisperFeatureExtractor(nMels);

        var defaults = new WhisperGenerationOptions();
        int? langToken = ResolveLanguage(extra, defaults.LanguageTokenId);
        bool translate = extra.TryGetValue("task", out string? task)
            && string.Equals(task, "translate", StringComparison.OrdinalIgnoreCase);
        bool keepTimestamps = TryGetBool(extra, "timestamps", defaultValue: false);

        Whisper = defaults with
        {
            NumMels = nMels,
            LanguageTokenId = langToken,
            Translate = translate,
            NoTimestamps = !keepTimestamps,
            StartOfTranscriptId = TryGetInt(extra, "sot_id") ?? defaults.StartOfTranscriptId,
            EndOfTextId = TryGetInt(extra, "eot_id") ?? defaults.EndOfTextId,
            TranscribeTaskId = TryGetInt(extra, "transcribe_id") ?? defaults.TranscribeTaskId,
            TranslateTaskId = TryGetInt(extra, "translate_id") ?? defaults.TranslateTaskId,
            NoTimestampsId = TryGetInt(extra, "notimestamps_id") ?? defaults.NoTimestampsId,
            MaxNewTokens = TryGetInt(extra, "max_new_tokens") ?? defaults.MaxNewTokens,
        };

        string encoderInput = extra.TryGetValue("encoder_input", out string? ei) && !string.IsNullOrWhiteSpace(ei)
            ? ei!
            : "input_features";

        Options = new Seq2SeqModelOptions
        {
            EncoderInputIdsName = encoderInput,
            EncoderHiddenStatesOutputName = "last_hidden_state",
            DecoderStartTokenId = Whisper.StartOfTranscriptId,
            KvCacheNumHeads = TryGetInt(extra, "kv_num_heads"),
            KvCacheHeadDim = TryGetInt(extra, "kv_head_dim"),
        };

        DefaultConfig = new GenerationConfig
        {
            MaxNewTokens = Whisper.MaxNewTokens,
            DoSample = false,
            EosTokenIds = new[] { Whisper.EndOfTextId },
        };
    }

    /// <summary>Turns a mono 16 kHz PCM waveform into the encoder feature tensor (<c>[1, n_mels, 3000]</c>).</summary>
    public NamedTensor ExtractFeatures(ReadOnlySpan<float> waveform)
    {
        Tensor<float> features = FeatureExtractor.Extract(waveform);
        return new NamedTensor(Options.EncoderInputIdsName, features);
    }

    /// <summary>The forced decoder prompt for this configuration (start, language, task, no-timestamps).</summary>
    public IReadOnlyList<long> ForcedPrompt() => Whisper.BuildForcedPrompt();

    /// <summary>Decodes generated token ids to text, stripping Whisper special tokens.</summary>
    public string Decode(IReadOnlyList<long> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        var ints = new List<int>(ids.Count);
        foreach (long id in ids)
        {
            int v = checked((int)id);
            // Whisper special tokens (>= 50257) carry timestamps / prompt markers — never text.
            if (v >= Whisper.EndOfTextId) continue;
            ints.Add(v);
        }
        return Tokenizer.Decode(ints, skipSpecial: true);
    }

    /// <summary>Builds a <see cref="Seq2SeqGenerator"/> over the supplied encoder/decoder engines.</summary>
    public Seq2SeqGenerator CreateGenerator(IExecutionEngine encoderEngine, IExecutionEngine decoderEngine)
        => new Seq2SeqGenerator(encoderEngine, decoderEngine, Options);

    /// <summary>Runs the full transcribe path: features → encoder-once → forced-prompt decode → text.</summary>
    public string Transcribe(Seq2SeqGenerator generator, ReadOnlySpan<float> waveform, GenerationConfig? config = null)
    {
        NamedTensor features = ExtractFeatures(waveform);
        IReadOnlyList<long> generated = generator.GenerateFromFeatures(
            features, ForcedPrompt(), config ?? DefaultConfig);
        return Decode(generated);
    }

    // ---- helpers ----

    private static int? ResolveLanguage(IReadOnlyDictionary<string, string> extra, int? fallback)
    {
        if (!extra.TryGetValue("language", out string? lang) || string.IsNullOrWhiteSpace(lang)) return fallback;
        if (string.Equals(lang, "none", StringComparison.OrdinalIgnoreCase)) return null;
        if (int.TryParse(lang, NumberStyles.Integer, CultureInfo.InvariantCulture, out int id)) return id;
        return fallback; // a non-numeric language code without a token map: keep the default token.
    }

    private static int? TryGetInt(IReadOnlyDictionary<string, string> extra, string key)
        => extra.TryGetValue(key, out string? s) && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
            ? v
            : null;

    private static bool TryGetBool(IReadOnlyDictionary<string, string> extra, string key, bool defaultValue)
        => extra.TryGetValue(key, out string? s) && bool.TryParse(s, out bool v) ? v : defaultValue;
}
