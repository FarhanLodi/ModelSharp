using System.Collections.Generic;

namespace ModelSharp.Text;

/// <summary>
/// WordPiece tokenizer (BERT family) — pure managed, no dependencies. Runs BERT-style basic
/// tokenization via <see cref="BasicTokenizer"/> (clean → CJK per-character splitting → Unicode
/// normalization → optional lowercase + accent strip → whitespace/punctuation splitting) then
/// greedy longest-match WordPiece, and assembles [CLS]/[SEP] encodings.
/// </summary>
public sealed class WordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly BasicTokenizer _basic;
    private readonly string _unk;
    private readonly string _cls;
    private readonly string _sep;
    private readonly string _continuation;
    private readonly int _maxChars;

    public WordPieceTokenizer(
        IReadOnlyDictionary<string, int> vocab,
        bool lowercase = true,
        bool? stripAccents = null,
        string unkToken = "[UNK]",
        string clsToken = "[CLS]",
        string sepToken = "[SEP]",
        string continuationPrefix = "##",
        int maxInputCharsPerWord = 100)
    {
        _vocab = new Dictionary<string, int>(vocab.Count);
        foreach (KeyValuePair<string, int> kv in vocab) _vocab[kv.Key] = kv.Value;
        _basic = new BasicTokenizer(lowercase, stripAccents);
        _unk = unkToken;
        _cls = clsToken;
        _sep = sepToken;
        _continuation = continuationPrefix;
        _maxChars = maxInputCharsPerWord;
    }

    /// <summary>Builds a tokenizer from vocab.txt lines (one token per line; id = line index).</summary>
    public static WordPieceTokenizer FromVocab(IEnumerable<string> lines, bool lowercase = true)
    {
        var vocab = new Dictionary<string, int>();
        int i = 0;
        foreach (string line in lines)
        {
            string tok = line.TrimEnd('\n', '\r');
            if (!vocab.ContainsKey(tok)) vocab[tok] = i;
            i++;
        }
        return new WordPieceTokenizer(vocab, lowercase);
    }

    public int VocabSize => _vocab.Count;

    public bool TryGetId(string token, out int id) => _vocab.TryGetValue(token, out id);

    /// <summary>Basic + WordPiece tokenization to subword strings (no special tokens).</summary>
    public List<string> TokenizeToPieces(string text)
    {
        var pieces = new List<string>();
        foreach (string word in _basic.Tokenize(text)) WordPiece(word, pieces);
        return pieces;
    }

    /// <summary>Full encoding with [CLS]/[SEP] and attention/type id arrays.</summary>
    public Encoding Encode(string text, bool addSpecialTokens = true)
    {
        var tokens = new List<string>();
        if (addSpecialTokens) tokens.Add(_cls);
        tokens.AddRange(TokenizeToPieces(text));
        if (addSpecialTokens) tokens.Add(_sep);

        var ids = new int[tokens.Count];
        var mask = new int[tokens.Count];
        var types = new int[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
        {
            ids[i] = _vocab.TryGetValue(tokens[i], out int id) ? id : _vocab[_unk];
            mask[i] = 1;
            types[i] = 0;
        }
        return new Encoding(ids, mask, types, tokens);
    }

    private void WordPiece(string word, List<string> outPieces)
    {
        if (word.Length > _maxChars) { outPieces.Add(_unk); return; }

        int start = 0;
        var sub = new List<string>();
        bool bad = false;
        while (start < word.Length)
        {
            int end = word.Length;
            string? cur = null;
            while (start < end)
            {
                string piece = word.Substring(start, end - start);
                if (start > 0) piece = _continuation + piece;
                if (_vocab.ContainsKey(piece)) { cur = piece; break; }
                end--;
            }
            if (cur is null) { bad = true; break; }
            sub.Add(cur);
            start = end;
        }
        if (bad) outPieces.Add(_unk);
        else outPieces.AddRange(sub);
    }
}
