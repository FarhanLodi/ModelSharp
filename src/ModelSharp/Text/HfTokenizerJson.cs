using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace ModelSharp.Text;

/// <summary>
/// Loads a Hugging Face <c>tokenizer.json</c> (the single-file <c>tokenizers</c> serialization) and
/// builds a working byte-level BPE <see cref="BpeTokenizer"/> from it — pure managed, zero extra
/// dependencies (System.Text.Json only). Supports the byte-level BPE family used by GPT-2, RoBERTa,
/// Llama-3 and Qwen2.5.
///
/// <para>The fields consumed are:</para>
/// <list type="bullet">
///   <item><c>model.type == "BPE"</c> — the only supported model kind.</item>
///   <item><c>model.vocab</c> — a <c>{ token : id }</c> dictionary (the byte-level subword vocabulary).</item>
///   <item><c>model.merges</c> — the rank-ordered merge rules, either as <c>"left right"</c> strings
///         (classic) or as <c>["left","right"]</c> two-element arrays (newer exports).</item>
///   <item><c>model.unk_token</c> — optional unknown-token fallback.</item>
///   <item><c>added_tokens</c> — the special / added tokens (e.g. <c>&lt;|endoftext|&gt;</c>,
///         <c>&lt;|im_start|&gt;</c>); each is matched as a single whole unit on encode and emitted /
///         suppressed as a unit on decode.</item>
///   <item><c>pre_tokenizer</c> — when it carries a <c>Split</c> regex pattern (as Qwen/Llama-3 do),
///         that exact pattern drives pre-tokenization so digit-by-digit splitting and the leading
///         non-letter prefix match the reference tokenizer; otherwise the GPT-2 default is used.</item>
/// </list>
/// </summary>
public static class HfTokenizerJson
{
    /// <summary>
    /// Parses the <c>tokenizer.json</c> at <paramref name="path"/> and returns a ready
    /// <see cref="BpeTokenizer"/>. See <see cref="Load(JsonDocument, string?, string?)"/> for the
    /// optional BOS/EOS handling.
    /// </summary>
    public static BpeTokenizer FromFile(string path, string? bosToken = null, string? eosToken = null)
    {
        if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path is required.", nameof(path));
        using FileStream fs = File.OpenRead(path);
        using JsonDocument doc = JsonDocument.Parse(fs);
        return Load(doc, bosToken, eosToken);
    }

    /// <summary>Parses a <c>tokenizer.json</c> from an in-memory string. Mainly for tests.</summary>
    public static BpeTokenizer FromJson(string json, string? bosToken = null, string? eosToken = null)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        return Load(doc, bosToken, eosToken);
    }

    /// <summary>
    /// Builds a <see cref="BpeTokenizer"/> from an already-parsed <c>tokenizer.json</c> document.
    /// </summary>
    /// <param name="doc">The parsed <c>tokenizer.json</c> document.</param>
    /// <param name="bosToken">Optional BOS token (must be a known vocab/added token); prepended when
    /// the caller encodes with <c>addSpecial</c>.</param>
    /// <param name="eosToken">Optional EOS token; appended when encoding with <c>addSpecial</c>.</param>
    public static BpeTokenizer Load(JsonDocument doc, string? bosToken = null, string? eosToken = null)
    {
        JsonElement root = doc.RootElement;

        if (!root.TryGetProperty("model", out JsonElement model))
            throw new InvalidDataException("tokenizer.json has no 'model' section.");

        if (model.TryGetProperty("type", out JsonElement typeEl) &&
            typeEl.ValueKind == JsonValueKind.String &&
            !string.Equals(typeEl.GetString(), "BPE", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"Only model.type == \"BPE\" is supported, got \"{typeEl.GetString()}\".");
        }

        Dictionary<string, int> vocab = ReadVocab(model);
        List<(string, string)> merges = ReadMerges(model);
        Dictionary<string, int> specials = ReadAddedTokens(root);

        string? unkToken = ReadUnkToken(model);
        string? prePattern = ReadPreTokenizerPattern(root);

        return new BpeTokenizer(vocab, merges, specials, bosToken, eosToken, unkToken, prePattern);
    }

    // ---- field readers ----

    private static Dictionary<string, int> ReadVocab(JsonElement model)
    {
        if (!model.TryGetProperty("vocab", out JsonElement vocabEl) || vocabEl.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("tokenizer.json model.vocab is missing or not an object.");

        var vocab = new Dictionary<string, int>(vocabEl.GetRawText().Length / 16);
        foreach (JsonProperty p in vocabEl.EnumerateObject())
            vocab[p.Name] = p.Value.GetInt32();
        return vocab;
    }

    private static List<(string, string)> ReadMerges(JsonElement model)
    {
        var merges = new List<(string, string)>();
        if (!model.TryGetProperty("merges", out JsonElement mergesEl) || mergesEl.ValueKind != JsonValueKind.Array)
            return merges;

        foreach (JsonElement m in mergesEl.EnumerateArray())
        {
            if (m.ValueKind == JsonValueKind.String)
            {
                // Classic form: "left right". Split on the FIRST space only — byte-level tokens never
                // contain a raw space (0x20 is encoded as 'Ġ'), so the first space is the separator.
                string s = m.GetString()!;
                int sp = s.IndexOf(' ');
                if (sp <= 0 || sp >= s.Length - 1) continue;
                merges.Add((s.Substring(0, sp), s.Substring(sp + 1)));
            }
            else if (m.ValueKind == JsonValueKind.Array && m.GetArrayLength() == 2)
            {
                // Newer form: ["left", "right"].
                string left = m[0].GetString() ?? string.Empty;
                string right = m[1].GetString() ?? string.Empty;
                if (left.Length == 0 || right.Length == 0) continue;
                merges.Add((left, right));
            }
        }
        return merges;
    }

    private static Dictionary<string, int> ReadAddedTokens(JsonElement root)
    {
        var specials = new Dictionary<string, int>();
        if (!root.TryGetProperty("added_tokens", out JsonElement added) || added.ValueKind != JsonValueKind.Array)
            return specials;

        foreach (JsonElement t in added.EnumerateArray())
        {
            if (t.ValueKind != JsonValueKind.Object) continue;
            if (!t.TryGetProperty("content", out JsonElement contentEl) || contentEl.ValueKind != JsonValueKind.String)
                continue;
            if (!t.TryGetProperty("id", out JsonElement idEl) || !idEl.TryGetInt32(out int id))
                continue;
            string content = contentEl.GetString()!;
            if (content.Length == 0) continue;
            // Registering it as a special token makes BpeTokenizer match it as a single whole unit on
            // encode and (with skipSpecial) drop it on decode — both "special" and merely-"added" tokens
            // are treated this way so they never get byte-level-split.
            specials[content] = id;
        }
        return specials;
    }

    private static string? ReadUnkToken(JsonElement model)
    {
        if (model.TryGetProperty("unk_token", out JsonElement unkEl) && unkEl.ValueKind == JsonValueKind.String)
        {
            string s = unkEl.GetString()!;
            return s.Length == 0 ? null : s;
        }
        return null;
    }

    /// <summary>
    /// Extracts the byte-level pre-tokenization split regex from <c>pre_tokenizer</c>. The HF
    /// pre-tokenizer is usually a <c>Sequence</c> of a <c>Split</c> (carrying the regex) followed by a
    /// <c>ByteLevel</c> step; this digs out the <c>Split</c>'s <c>pattern.Regex</c>. Returns null when
    /// none is present (so the tokenizer falls back to the GPT-2 default pattern).
    /// </summary>
    private static string? ReadPreTokenizerPattern(JsonElement root)
    {
        if (!root.TryGetProperty("pre_tokenizer", out JsonElement pre) || pre.ValueKind != JsonValueKind.Object)
            return null;

        return FindSplitRegex(pre);

        static string? FindSplitRegex(JsonElement el)
        {
            if (el.ValueKind != JsonValueKind.Object) return null;

            string? type = el.TryGetProperty("type", out JsonElement t) && t.ValueKind == JsonValueKind.String
                ? t.GetString()
                : null;

            if (string.Equals(type, "Split", StringComparison.Ordinal) &&
                el.TryGetProperty("pattern", out JsonElement pat) && pat.ValueKind == JsonValueKind.Object &&
                pat.TryGetProperty("Regex", out JsonElement rx) && rx.ValueKind == JsonValueKind.String)
            {
                string s = rx.GetString()!;
                return s.Length == 0 ? null : s;
            }

            // Recurse into a Sequence's child pretokenizers (first Split wins).
            if (el.TryGetProperty("pretokenizers", out JsonElement kids) && kids.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement kid in kids.EnumerateArray())
                {
                    string? found = FindSplitRegex(kid);
                    if (found is not null) return found;
                }
            }
            return null;
        }
    }
}
