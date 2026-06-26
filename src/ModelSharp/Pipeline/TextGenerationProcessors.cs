using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Text;

namespace ModelSharp.Pipeline;

/// <summary>
/// Builds the generation stack for a <see cref="Manifest.ModelTask.TextGeneration"/> model: a
/// byte-level <see cref="BpeTokenizer"/> (GPT-2 / completion family) plus the
/// <see cref="DecoderModelOptions"/> and default <see cref="GenerationConfig"/> derived from the
/// manifest. Unlike the single-shot embedding/vision processors this type does not implement the
/// <see cref="IPreprocessor"/>/<see cref="IPostprocessor"/> interfaces — autoregressive decoding is
/// a loop, not a single <c>ToFeeds → Run → Decode</c> pass — so it is consumed directly by
/// <see cref="ModelSharpPipeline.Build"/>, which constructs a <see cref="TextGenerator"/> over the
/// engine and hands the tokenizer + config to the <see cref="Pipeline"/>'s generation entry points.
///
/// <para>Manifest hints (all under <c>Manifest.Extra</c>, all optional except the tokenizer files):</para>
/// <list type="bullet">
///   <item><description><c>vocab</c> — path to <c>vocab.json</c> (required).</description></item>
///   <item><description><c>merges</c> — path to <c>merges.txt</c> (required).</description></item>
///   <item><description><c>kv_num_heads</c>, <c>kv_head_dim</c> — KV-cache head dims for engines that
///   do not report concrete past-KV shapes.</description></item>
///   <item><description><c>eos_token_id</c> — end-of-sequence id (also accepted as <c>eos</c>).</description></item>
///   <item><description><c>bos_token</c>, <c>eos_token</c>, <c>unk_token</c> — optional special-token strings.</description></item>
///   <item><description><c>max_new_tokens</c> — default generation length.</description></item>
///   <item><description><c>add_special_tokens</c> — whether to prepend/append BOS/EOS when encoding (default false).</description></item>
/// </list>
/// </summary>
public sealed class TextGenerationProcessor
{
    /// <summary>The byte-level BPE tokenizer built from the manifest's <c>vocab.json</c> + <c>merges.txt</c>.</summary>
    public BpeTokenizer Tokenizer { get; }

    /// <summary>The decoder IO-binding options (KV-cache head dims and binding names) for this model.</summary>
    public DecoderModelOptions DecoderOptions { get; }

    /// <summary>The default decoding configuration, used by <see cref="Pipeline.Generate"/> when none is supplied.</summary>
    public GenerationConfig DefaultConfig { get; }

    /// <summary>Whether BOS/EOS special tokens are added when encoding the prompt.</summary>
    public bool AddSpecialTokens { get; }

    /// <summary>Builds the generation stack from a processor context (manifest + engine binding names).</summary>
    public TextGenerationProcessor(ProcessorContext ctx)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));
        IReadOnlyDictionary<string, string> extra = ctx.Manifest.Extra;

        if (!TryGet(extra, out string? vocabPath, "vocab") || string.IsNullOrWhiteSpace(vocabPath))
            throw new ModelSharpException(
                "Text generation requires a BPE vocab; set manifest Extra[\"vocab\"] to a vocab.json path.");
        if (!TryGet(extra, out string? mergesPath, "merges") || string.IsNullOrWhiteSpace(mergesPath))
            throw new ModelSharpException(
                "Text generation requires BPE merges; set manifest Extra[\"merges\"] to a merges.txt path.");
        if (!File.Exists(vocabPath))
            throw new ModelSharpException($"Vocab file not found for text generation: '{vocabPath}'.");
        if (!File.Exists(mergesPath))
            throw new ModelSharpException($"Merges file not found for text generation: '{mergesPath}'.");

        string? bos = TryGet(extra, out string? b, "bos_token") ? b : null;
        string? eos = TryGet(extra, out string? e, "eos_token") ? e : null;
        string? unk = TryGet(extra, out string? u, "unk_token") ? u : null;
        Tokenizer = BpeTokenizer.FromFiles(vocabPath!, mergesPath!, specialTokens: null, bosToken: bos, eosToken: eos, unkToken: unk);

        AddSpecialTokens = TryGetBool(extra, "add_special_tokens", defaultValue: false);

        int? kvHeads = TryGetInt(extra, "kv_num_heads");
        int? kvHeadDim = TryGetInt(extra, "kv_head_dim");
        DecoderOptions = new DecoderModelOptions
        {
            KvCacheNumHeads = kvHeads,
            KvCacheHeadDim = kvHeadDim,
        };

        int maxNewTokens = TryGetInt(extra, "max_new_tokens") ?? 20;
        int? eosId = TryGetInt(extra, "eos_token_id") ?? TryGetInt(extra, "eos");
        DefaultConfig = new GenerationConfig
        {
            MaxNewTokens = maxNewTokens,
            EosTokenIds = eosId is int id ? new[] { id } : null,
        };
    }

    /// <summary>Encodes a prompt to token ids using the configured BPE tokenizer.</summary>
    public IReadOnlyList<long> Encode(string prompt)
    {
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        int[] ids = Tokenizer.Encode(prompt, addSpecial: AddSpecialTokens);
        var longs = new long[ids.Length];
        for (int i = 0; i < ids.Length; i++) longs[i] = ids[i];
        return longs;
    }

    /// <summary>Decodes generated token ids back to text, dropping special tokens.</summary>
    public string Decode(IReadOnlyList<long> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        var ints = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++) ints[i] = checked((int)ids[i]);
        return Tokenizer.Decode(ints, skipSpecial: true);
    }

    /// <summary>Builds a <see cref="TextGenerator"/> over the supplied engine using this model's decoder options.</summary>
    public TextGenerator CreateGenerator(IExecutionEngine engine)
        => new TextGenerator(engine, DecoderOptions);

    // ---- helpers ----

    private static bool TryGet(IReadOnlyDictionary<string, string> extra, out string? value, string key)
        => extra.TryGetValue(key, out value);

    private static int? TryGetInt(IReadOnlyDictionary<string, string> extra, string key)
        => extra.TryGetValue(key, out string? s) && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
            ? v
            : null;

    private static bool TryGetBool(IReadOnlyDictionary<string, string> extra, string key, bool defaultValue)
        => extra.TryGetValue(key, out string? s) && bool.TryParse(s, out bool v) ? v : defaultValue;
}
