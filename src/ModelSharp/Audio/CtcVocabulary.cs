using System;
using System.Collections.Generic;
using System.Text;

namespace ModelSharp.Audio;

/// <summary>
/// Maps CTC token indices to their string/character form and joins a decoded index
/// sequence into readable text. Mirrors the common wav2vec2/DeepSpeech vocabularies:
/// a word-delimiter token (e.g. <c>"|"</c>) is rendered as a space and special tokens
/// (<c>&lt;pad&gt;</c>/<c>&lt;s&gt;</c>/<c>&lt;/s&gt;</c>/<c>&lt;unk&gt;</c>) render as the empty string.
/// </summary>
/// <remarks>
/// The vocabulary is built from an ordered list of token strings where the list position is
/// the token index (index 0 is conventionally the CTC blank, often shared with <c>&lt;pad&gt;</c>).
/// <see cref="Decode"/> performs a pure index→text mapping; it does NOT collapse repeats or
/// strip the blank — that is <see cref="CtcDecoder"/>'s responsibility. Feed it an already
/// collapsed/blank-removed sequence.
/// </remarks>
public sealed class CtcVocabulary
{
    /// <summary>The default special tokens that render as the empty string.</summary>
    public static readonly IReadOnlyList<string> DefaultSpecialTokens =
        new[] { "<pad>", "<s>", "</s>", "<unk>" };

    private readonly string[] _tokens;
    private readonly HashSet<string> _specials;
    private readonly string? _wordDelimiter;
    private readonly string _wordDelimiterReplacement;

    /// <summary>Builds a vocabulary from an ordered token list (index = position).</summary>
    /// <param name="tokens">Ordered token strings; the index of each token is its position.</param>
    /// <param name="wordDelimiter">Token rendered as <paramref name="wordDelimiterReplacement"/>; null/empty disables this.</param>
    /// <param name="wordDelimiterReplacement">Text emitted in place of the word delimiter (default a single space).</param>
    /// <param name="specialTokens">Tokens rendered as empty; defaults to <see cref="DefaultSpecialTokens"/>.</param>
    public CtcVocabulary(
        IReadOnlyList<string> tokens,
        string? wordDelimiter = "|",
        string wordDelimiterReplacement = " ",
        IEnumerable<string>? specialTokens = null)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Count == 0) throw new ArgumentException("Vocabulary must contain at least one token.", nameof(tokens));

        _tokens = new string[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
            _tokens[i] = tokens[i] ?? throw new ArgumentException($"Token at index {i} is null.", nameof(tokens));

        _wordDelimiter = string.IsNullOrEmpty(wordDelimiter) ? null : wordDelimiter;
        _wordDelimiterReplacement = wordDelimiterReplacement ?? string.Empty;
        _specials = new HashSet<string>(specialTokens ?? DefaultSpecialTokens, StringComparer.Ordinal);
    }

    /// <summary>The number of tokens in the vocabulary.</summary>
    public int Count => _tokens.Length;

    /// <summary>The ordered token strings (index = position).</summary>
    public IReadOnlyList<string> Tokens => _tokens;

    /// <summary>The raw token string at <paramref name="index"/> (no special-token rendering).</summary>
    public string TokenAt(int index)
    {
        if (index < 0 || index >= _tokens.Length)
            throw new ArgumentOutOfRangeException(nameof(index), index, $"Index must be in [0, {_tokens.Length}).");
        return _tokens[index];
    }

    /// <summary>
    /// Joins a sequence of token indices into text: the word delimiter becomes a space,
    /// special tokens are dropped, and every other token contributes its string form.
    /// </summary>
    public string Decode(IEnumerable<int> indices)
    {
        if (indices is null) throw new ArgumentNullException(nameof(indices));

        var sb = new StringBuilder();
        foreach (int i in indices)
        {
            if (i < 0 || i >= _tokens.Length)
                throw new ArgumentOutOfRangeException(nameof(indices), i, $"Index must be in [0, {_tokens.Length}).");

            string tok = _tokens[i];
            if (_wordDelimiter is not null && string.Equals(tok, _wordDelimiter, StringComparison.Ordinal))
            {
                sb.Append(_wordDelimiterReplacement);
                continue;
            }
            if (_specials.Contains(tok)) continue;
            sb.Append(tok);
        }
        return sb.ToString();
    }
}
