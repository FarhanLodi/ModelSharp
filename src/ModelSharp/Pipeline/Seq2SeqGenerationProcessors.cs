using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Text;

namespace ModelSharp.Pipeline;

/// <summary>
/// Builds the generation stack for a <see cref="Manifest.ModelTask.Seq2SeqGeneration"/> model
/// (text encoder-decoder: T5 / BART / MarianMT): a byte-level <see cref="BpeTokenizer"/> plus the
/// <see cref="Seq2SeqModelOptions"/> and default <see cref="GenerationConfig"/> derived from the manifest.
/// Like <see cref="TextGenerationProcessor"/> it does not implement the single-shot
/// <see cref="IPreprocessor"/>/<see cref="IPostprocessor"/> interfaces — encoder-decoder decoding is a loop —
/// so it is consumed directly by <see cref="ModelSharpPipeline.Build"/>, which constructs a
/// <see cref="Seq2SeqGenerator"/> over the engine(s) and hands the tokenizer + config to the
/// <see cref="Pipeline"/>'s generation entry points.
///
/// <para>Manifest hints (all under <c>Manifest.Extra</c>, all optional except the tokenizer files):</para>
/// <list type="bullet">
///   <item><description><c>vocab</c> — path to <c>vocab.json</c> (required).</description></item>
///   <item><description><c>merges</c> — path to <c>merges.txt</c> (required).</description></item>
///   <item><description><c>kv_num_heads</c>, <c>kv_head_dim</c> — KV-cache head dims for engines that do not
///   report concrete past-KV shapes.</description></item>
///   <item><description><c>decoder_start_token_id</c> — the id the decoder loop starts from (default 0, the T5 pad id).</description></item>
///   <item><description><c>eos_token_id</c> — end-of-sequence id (also accepted as <c>eos</c>).</description></item>
///   <item><description><c>bos_token</c>, <c>eos_token</c>, <c>unk_token</c> — optional special-token strings.</description></item>
///   <item><description><c>max_new_tokens</c> — default generation length.</description></item>
///   <item><description><c>add_special_tokens</c> — whether to add special tokens when encoding the source (default true,
///   since seq2seq source encoders normally append EOS).</description></item>
/// </list>
/// </summary>
public sealed class Seq2SeqGenerationProcessor
{
    /// <summary>The byte-level BPE tokenizer built from the manifest's <c>vocab.json</c> + <c>merges.txt</c>.</summary>
    public BpeTokenizer Tokenizer { get; }

    /// <summary>The encoder-decoder IO-binding options (KV-cache dims, cross-attn naming, start token).</summary>
    public Seq2SeqModelOptions Options { get; }

    /// <summary>The default decoding configuration, used by <see cref="Pipeline.Generate"/> when none is supplied.</summary>
    public GenerationConfig DefaultConfig { get; }

    /// <summary>Whether special tokens are added when encoding the source text.</summary>
    public bool AddSpecialTokens { get; }

    /// <summary>Builds the seq2seq generation stack from a processor context (manifest + engine binding names).</summary>
    public Seq2SeqGenerationProcessor(ProcessorContext ctx)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));
        IReadOnlyDictionary<string, string> extra = ctx.Manifest.Extra;

        if (!extra.TryGetValue("vocab", out string? vocabPath) || string.IsNullOrWhiteSpace(vocabPath))
            throw new ModelSharpException(
                "Seq2Seq generation requires a BPE vocab; set manifest Extra[\"vocab\"] to a vocab.json path.");
        if (!extra.TryGetValue("merges", out string? mergesPath) || string.IsNullOrWhiteSpace(mergesPath))
            throw new ModelSharpException(
                "Seq2Seq generation requires BPE merges; set manifest Extra[\"merges\"] to a merges.txt path.");
        if (!File.Exists(vocabPath))
            throw new ModelSharpException($"Vocab file not found for seq2seq generation: '{vocabPath}'.");
        if (!File.Exists(mergesPath))
            throw new ModelSharpException($"Merges file not found for seq2seq generation: '{mergesPath}'.");

        string? bos = extra.TryGetValue("bos_token", out string? b) ? b : null;
        string? eos = extra.TryGetValue("eos_token", out string? e) ? e : null;
        string? unk = extra.TryGetValue("unk_token", out string? u) ? u : null;
        Tokenizer = BpeTokenizer.FromFiles(vocabPath!, mergesPath!, specialTokens: null, bosToken: bos, eosToken: eos, unkToken: unk);

        AddSpecialTokens = TryGetBool(extra, "add_special_tokens", defaultValue: true);

        int? kvHeads = TryGetInt(extra, "kv_num_heads");
        int? kvHeadDim = TryGetInt(extra, "kv_head_dim");
        long startId = TryGetInt(extra, "decoder_start_token_id") ?? 0;
        Options = new Seq2SeqModelOptions
        {
            KvCacheNumHeads = kvHeads,
            KvCacheHeadDim = kvHeadDim,
            DecoderStartTokenId = startId,
        };

        int maxNewTokens = TryGetInt(extra, "max_new_tokens") ?? 20;
        int? eosId = TryGetInt(extra, "eos_token_id") ?? TryGetInt(extra, "eos");
        DefaultConfig = new GenerationConfig
        {
            MaxNewTokens = maxNewTokens,
            EosTokenIds = eosId is int id ? new[] { id } : null,
        };
    }

    /// <summary>Encodes source text to token ids using the configured BPE tokenizer.</summary>
    public IReadOnlyList<long> Encode(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        int[] ids = Tokenizer.Encode(text, addSpecial: AddSpecialTokens);
        var longs = new long[ids.Length];
        for (int i = 0; i < ids.Length; i++) longs[i] = ids[i];
        return longs;
    }

    /// <summary>Decodes generated target token ids back to text, dropping special tokens.</summary>
    public string Decode(IReadOnlyList<long> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        var ints = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++) ints[i] = checked((int)ids[i]);
        return Tokenizer.Decode(ints, skipSpecial: true);
    }

    /// <summary>Builds a <see cref="Seq2SeqGenerator"/> over the supplied encoder/decoder engines.</summary>
    public Seq2SeqGenerator CreateGenerator(IExecutionEngine encoderEngine, IExecutionEngine decoderEngine)
        => new Seq2SeqGenerator(encoderEngine, decoderEngine, Options);

    // ---- helpers ----

    private static int? TryGetInt(IReadOnlyDictionary<string, string> extra, string key)
        => extra.TryGetValue(key, out string? s) && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
            ? v
            : null;

    private static bool TryGetBool(IReadOnlyDictionary<string, string> extra, string key, bool defaultValue)
        => extra.TryGetValue(key, out string? s) && bool.TryParse(s, out bool v) ? v : defaultValue;
}
