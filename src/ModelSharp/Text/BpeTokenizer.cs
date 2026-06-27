using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ModelSharp.Text;

/// <summary>
/// GPT-2 / RoBERTa style byte-level BPE tokenizer (the family used by most LLM completion models) —
/// pure managed, no dependencies. Pipeline: split out special tokens → GPT-2 regex pre-tokenization
/// → UTF-8 bytes mapped through <see cref="ByteLevel"/> → rank-based BPE merges (cached per word) →
/// vocab ids. Decoding reverses each step, so arbitrary UTF-8 (including emoji / multibyte text)
/// round-trips. Built from the standard <c>vocab.json</c> + <c>merges.txt</c> artifacts.
/// </summary>
/// <remarks>Not thread-safe: <see cref="Encode"/> mutates an internal per-word merge cache.</remarks>
public sealed class BpeTokenizer
{
    // GPT-2 pre-tokenization pattern: contractions, letter runs, number runs, punctuation runs and
    // whitespace runs (the trailing whitespace of a run attaches to the following word).
    private static readonly Regex _pat = new Regex(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    private readonly Regex _splitPat;
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<string, List<string>> _cache = new();
    private readonly Dictionary<string, int> _specialTokens;
    private readonly Dictionary<int, string> _specialIdToString;
    private readonly Regex? _specialRegex;
    private readonly int _bosId;
    private readonly int _eosId;
    private readonly int _unkId;
    private readonly int _vocabSize;

    /// <summary>
    /// Creates a tokenizer from already-parsed artifacts.
    /// </summary>
    /// <param name="vocab">Token string → id (the contents of <c>vocab.json</c>).</param>
    /// <param name="merges">Ordered BPE merge rules; index = merge rank (lower merges first).</param>
    /// <param name="specialTokens">Optional special / added tokens (string → id) matched as whole units before byte-level splitting.</param>
    /// <param name="bosToken">Optional beginning-of-sequence token added when <c>addSpecial</c> is set; its id is resolved from <paramref name="specialTokens"/> or <paramref name="vocab"/>.</param>
    /// <param name="eosToken">Optional end-of-sequence token added when <c>addSpecial</c> is set.</param>
    /// <param name="unkToken">Optional fallback token for byte-level pieces absent from the vocab; when null such pieces throw.</param>
    /// <param name="preTokenizePattern">Optional override for the pre-tokenization split regex (e.g. the Qwen/Llama-3 pattern, which splits digits singly and attaches a leading non-letter to letter runs). Defaults to the GPT-2 pattern.</param>
    public BpeTokenizer(
        IReadOnlyDictionary<string, int> vocab,
        IReadOnlyList<(string left, string right)> merges,
        IReadOnlyDictionary<string, int>? specialTokens = null,
        string? bosToken = null,
        string? eosToken = null,
        string? unkToken = null,
        string? preTokenizePattern = null)
    {
        _splitPat = preTokenizePattern is null
            ? _pat
            : new Regex(preTokenizePattern, RegexOptions.Compiled);
        _vocab = new Dictionary<string, int>(vocab.Count);
        _idToToken = new Dictionary<int, string>(vocab.Count);
        foreach (KeyValuePair<string, int> kv in vocab)
        {
            _vocab[kv.Key] = kv.Value;
            _idToToken[kv.Value] = kv.Key;
        }

        _bpeRanks = new Dictionary<(string, string), int>(merges.Count);
        for (int i = 0; i < merges.Count; i++)
            _bpeRanks.TryAdd(merges[i], i); // keep the lowest (earliest) rank on duplicates

        var specialMap = new Dictionary<string, int>();
        if (specialTokens != null)
            foreach (KeyValuePair<string, int> kv in specialTokens) specialMap[kv.Key] = kv.Value;

        _bosId = bosToken is null ? -1 : ResolveId(bosToken, specialMap);
        _eosId = eosToken is null ? -1 : ResolveId(eosToken, specialMap);
        _unkId = unkToken is null ? -1 : ResolveId(unkToken, specialMap);

        // BOS/EOS must be matched as whole units in text and emitted as specials on decode.
        if (bosToken != null) specialMap[bosToken] = _bosId;
        if (eosToken != null) specialMap[eosToken] = _eosId;

        _specialTokens = specialMap;
        _specialIdToString = new Dictionary<int, string>(specialMap.Count);
        foreach (KeyValuePair<string, int> kv in specialMap) _specialIdToString[kv.Value] = kv.Key;

        _specialRegex = specialMap.Count == 0
            ? null
            // Longest first so e.g. "<|endoftext|>" wins over any shorter special prefix.
            : new Regex(
                string.Join("|", specialMap.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape)),
                RegexOptions.Compiled);

        _vocabSize = _vocab.Count;
        foreach (string key in specialMap.Keys)
            if (!_vocab.ContainsKey(key)) _vocabSize++;

        int ResolveId(string token, Dictionary<string, int> special)
        {
            if (special.TryGetValue(token, out int id)) return id;
            if (_vocab.TryGetValue(token, out id)) return id;
            throw new ArgumentException($"Token '{token}' is not present in the special tokens or the vocab.");
        }
    }

    /// <summary>Loads a tokenizer from the standard <c>vocab.json</c> and <c>merges.txt</c> files.</summary>
    public static BpeTokenizer FromFiles(
        string vocabJsonPath,
        string mergesTxtPath,
        IReadOnlyDictionary<string, int>? specialTokens = null,
        string? bosToken = null,
        string? eosToken = null,
        string? unkToken = null)
    {
        string json = File.ReadAllText(vocabJsonPath);
        Dictionary<string, int> vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(json)
            ?? throw new InvalidDataException($"Failed to parse vocab JSON: {vocabJsonPath}");
        string[] mergeLines = File.ReadAllLines(mergesTxtPath);
        return FromVocabAndMerges(vocab, mergeLines, specialTokens, bosToken, eosToken, unkToken);
    }

    /// <summary>
    /// Builds a tokenizer from an in-memory vocab and <c>merges.txt</c>-style lines (each
    /// <c>"left right"</c>); the <c>#version</c> header, comments and blank lines are skipped.
    /// </summary>
    public static BpeTokenizer FromVocabAndMerges(
        IReadOnlyDictionary<string, int> vocab,
        IEnumerable<string> mergeRules,
        IReadOnlyDictionary<string, int>? specialTokens = null,
        string? bosToken = null,
        string? eosToken = null,
        string? unkToken = null,
        string? preTokenizePattern = null)
    {
        return new BpeTokenizer(vocab, ParseMergeLines(mergeRules), specialTokens, bosToken, eosToken, unkToken, preTokenizePattern);
    }

    /// <summary>Number of distinct known tokens (base vocab plus any special tokens not already in it).</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Looks up the id of a token, checking special tokens first, then the base vocab.</summary>
    public bool TryGetId(string token, out int id)
    {
        if (_specialTokens.TryGetValue(token, out id)) return true;
        return _vocab.TryGetValue(token, out id);
    }

    /// <summary>
    /// Encodes <paramref name="text"/> to token ids. When <paramref name="addSpecial"/> is set and a
    /// BOS / EOS token was configured, they are prepended / appended.
    /// </summary>
    public int[] Encode(string text, bool addSpecial = false)
    {
        var ids = new List<int>();
        if (addSpecial && _bosId >= 0) ids.Add(_bosId);
        EncodeChunk(text, ids);
        if (addSpecial && _eosId >= 0) ids.Add(_eosId);
        return ids.ToArray();
    }

    /// <summary>
    /// Decodes token ids back to text. Byte-level tokens are mapped back to UTF-8 bytes; special
    /// tokens are dropped when <paramref name="skipSpecial"/> is set, otherwise emitted literally.
    /// </summary>
    public string Decode(IReadOnlyList<int> ids, bool skipSpecial = true)
    {
        var sb = new StringBuilder();
        var buffer = new List<byte>();
        foreach (int id in ids)
        {
            if (_specialIdToString.TryGetValue(id, out string? special))
            {
                FlushBytes(buffer, sb);
                if (!skipSpecial) sb.Append(special);
            }
            else if (_idToToken.TryGetValue(id, out string? token))
            {
                foreach (char c in token)
                    if (ByteLevel.TryDecodeChar(c, out byte b)) buffer.Add(b);
            }
            // Unknown ids are silently skipped.
        }
        FlushBytes(buffer, sb);
        return sb.ToString();
    }

    /// <summary>
    /// Returns the byte-level BPE subword strings for <paramref name="text"/> (e.g. with the GPT-2
    /// <c>Ġ</c> space marker), without mapping to ids. Special tokens are emitted as whole pieces.
    /// </summary>
    public IReadOnlyList<string> TokenizeToPieces(string text)
    {
        var pieces = new List<string>();
        if (_specialRegex is null) { PlainPieces(text, pieces); return pieces; }

        int last = 0;
        foreach (Match m in _specialRegex.Matches(text))
        {
            if (m.Index > last) PlainPieces(text.Substring(last, m.Index - last), pieces);
            pieces.Add(m.Value);
            last = m.Index + m.Length;
        }
        if (last < text.Length) PlainPieces(text.Substring(last), pieces);
        return pieces;
    }

    /// <summary>Applies only the GPT-2 pre-tokenization regex, returning the raw split pieces.</summary>
    public static IReadOnlyList<string> PreTokenize(string text)
    {
        var list = new List<string>();
        if (string.IsNullOrEmpty(text)) return list;
        foreach (Match m in _pat.Matches(text)) list.Add(m.Value);
        return list;
    }

    // ---- internals ----

    private void EncodeChunk(string text, List<int> ids)
    {
        if (string.IsNullOrEmpty(text)) return;
        if (_specialRegex is null) { EncodePlain(text, ids); return; }

        int last = 0;
        foreach (Match m in _specialRegex.Matches(text))
        {
            if (m.Index > last) EncodePlain(text.Substring(last, m.Index - last), ids);
            ids.Add(_specialTokens[m.Value]);
            last = m.Index + m.Length;
        }
        if (last < text.Length) EncodePlain(text.Substring(last), ids);
    }

    private void EncodePlain(string text, List<int> ids)
    {
        foreach (Match m in _splitPat.Matches(text))
            foreach (string piece in Bpe(MapBytes(m.Value)))
            {
                if (_vocab.TryGetValue(piece, out int id)) ids.Add(id);
                else if (_unkId >= 0) ids.Add(_unkId);
                else throw new InvalidOperationException($"Byte-level BPE token '{piece}' is not present in the vocab.");
            }
    }

    private void PlainPieces(string text, List<string> pieces)
    {
        foreach (Match m in _splitPat.Matches(text))
            pieces.AddRange(Bpe(MapBytes(m.Value)));
    }

    /// <summary>Encodes a pre-tokenized piece to its byte-level character string (UTF-8 bytes → mapped chars).</summary>
    private static string MapBytes(string piece)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(piece);
        var sb = new StringBuilder(bytes.Length);
        foreach (byte b in bytes) sb.Append(ByteLevel.EncodeByte(b));
        return sb.ToString();
    }

    /// <summary>Standard rank-based BPE: repeatedly merge the lowest-ranked adjacent symbol pair. Cached per word.</summary>
    private List<string> Bpe(string token)
    {
        if (_cache.TryGetValue(token, out List<string>? cached)) return cached;

        var word = new List<string>(token.Length);
        foreach (char c in token) word.Add(c.ToString());

        if (word.Count >= 2)
        {
            HashSet<(string, string)> pairs = GetPairs(word);
            while (true)
            {
                // Pick the adjacent pair with the lowest merge rank (ranks are unique, so no ties).
                (string, string) best = (string.Empty, string.Empty);
                int bestRank = int.MaxValue;
                bool found = false;
                foreach ((string, string) p in pairs)
                    if (_bpeRanks.TryGetValue(p, out int r) && r < bestRank) { bestRank = r; best = p; found = true; }
                if (!found) break;

                string first = best.Item1, second = best.Item2;
                var merged = new List<string>(word.Count);
                int i = 0;
                while (i < word.Count)
                {
                    int j = word.IndexOf(first, i);
                    if (j < 0) { for (int k = i; k < word.Count; k++) merged.Add(word[k]); break; }
                    for (int k = i; k < j; k++) merged.Add(word[k]);
                    i = j;
                    if (i < word.Count - 1 && word[i + 1] == second) { merged.Add(first + second); i += 2; }
                    else { merged.Add(word[i]); i += 1; }
                }
                word = merged;
                if (word.Count == 1) break;
                pairs = GetPairs(word);
            }
        }

        _cache[token] = word;
        return word;
    }

    private static HashSet<(string, string)> GetPairs(List<string> word)
    {
        var pairs = new HashSet<(string, string)>();
        for (int i = 1; i < word.Count; i++) pairs.Add((word[i - 1], word[i]));
        return pairs;
    }

    private static void FlushBytes(List<byte> buffer, StringBuilder sb)
    {
        if (buffer.Count == 0) return;
        sb.Append(System.Text.Encoding.UTF8.GetString(buffer.ToArray()));
        buffer.Clear();
    }

    private static List<(string, string)> ParseMergeLines(IEnumerable<string> lines)
    {
        var merges = new List<(string, string)>();
        foreach (string raw in lines)
        {
            string line = raw.TrimEnd('\r', '\n');
            if (line.Length == 0 || line[0] == '#') continue; // skip #version header / comments
            int sp = line.IndexOf(' ');
            if (sp <= 0 || sp >= line.Length - 1) continue;
            merges.Add((line.Substring(0, sp), line.Substring(sp + 1)));
        }
        return merges;
    }
}
