using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace ModelSharp.Text;

/// <summary>
/// BERT-style "basic" tokenizer — the whitespace / punctuation / CJK segmentation that runs
/// before WordPiece subword splitting. Faithful managed port of HuggingFace BERT
/// <c>BasicTokenizer</c>: text cleanup, CJK per-character splitting, Unicode (NFC) normalization,
/// optional lowercasing + accent stripping, and punctuation splitting.
/// </summary>
public sealed class BasicTokenizer
{
    private readonly bool _lowercase;
    private readonly bool _stripAccents;
    private readonly bool _tokenizeChineseChars;

    /// <param name="lowercase">Lowercase tokens (BERT <c>do_lower_case</c>).</param>
    /// <param name="stripAccents">
    /// Strip combining accents. When <c>null</c> it follows <paramref name="lowercase"/>, matching
    /// BERT's default where accent stripping is tied to lowercasing.
    /// </param>
    /// <param name="tokenizeChineseChars">Wrap each CJK character with spaces so it becomes its own token.</param>
    public BasicTokenizer(bool lowercase = true, bool? stripAccents = null, bool tokenizeChineseChars = true)
    {
        _lowercase = lowercase;
        _stripAccents = stripAccents ?? lowercase;
        _tokenizeChineseChars = tokenizeChineseChars;
    }

    /// <summary>
    /// Splits <paramref name="text"/> into whitespace / punctuation / CJK-delimited tokens,
    /// applying cleanup, normalization and (when enabled) lowercasing + accent stripping. This is
    /// the pre-tokenization that feeds WordPiece; no special tokens are added.
    /// </summary>
    public List<string> Tokenize(string text)
    {
        var output = new List<string>();
        if (text is null) return output;

        string cleaned = Clean(text);
        // Wrap CJK characters with whitespace so each becomes its own token (BERT does this for
        // multilingual/Chinese models; harmless for others).
        if (_tokenizeChineseChars) cleaned = TokenizeChineseChars(cleaned);
        // NFC so that identical glyphs encoded with different code points compare equal.
        string normalized = SafeNormalize(cleaned, NormalizationForm.FormC);

        foreach (string token in normalized.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            string t = token;
            if (_lowercase) t = t.ToLowerInvariant();
            if (_stripAccents) t = StripAccents(t);
            SplitOnPunctuation(t, output);
        }
        return output;
    }

    /// <summary>Drops null / replacement / control chars and folds whitespace runs to single spaces.</summary>
    private static string Clean(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (c == '\0' || c == '�' || IsControl(c)) continue;
            sb.Append(IsWhitespace(c) ? ' ' : c);
        }
        return sb.ToString();
    }

    /// <summary>Adds whitespace around every CJK character so it is split into its own token.</summary>
    private static string TokenizeChineseChars(string text)
    {
        var sb = new StringBuilder(text.Length);
        int i = 0;
        while (i < text.Length)
        {
            char c = text[i];
            int cp;
            int width;
            if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                cp = char.ConvertToUtf32(c, text[i + 1]);
                width = 2;
            }
            else
            {
                cp = c;
                width = 1;
            }

            if (IsChineseChar(cp))
            {
                sb.Append(' ');
                sb.Append(text, i, width);
                sb.Append(' ');
            }
            else
            {
                sb.Append(text, i, width);
            }
            i += width;
        }
        return sb.ToString();
    }

    /// <summary>NFD-normalizes then drops combining marks (BERT <c>_run_strip_accents</c>).</summary>
    private static string StripAccents(string text)
    {
        string norm = SafeNormalize(text, NormalizationForm.FormD);
        var sb = new StringBuilder(norm.Length);
        foreach (char c in norm)
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
                sb.Append(c);
        return sb.ToString();
    }

    /// <summary>Splits a whitespace-delimited token on punctuation, emitting punctuation as its own token.</summary>
    private static void SplitOnPunctuation(string text, List<string> outWords)
    {
        var sb = new StringBuilder();
        foreach (char c in text)
        {
            if (IsPunctuation(c))
            {
                if (sb.Length > 0) { outWords.Add(sb.ToString()); sb.Clear(); }
                outWords.Add(c.ToString());
            }
            else sb.Append(c);
        }
        if (sb.Length > 0) outWords.Add(sb.ToString());
    }

    /// <summary><see cref="string.Normalize(NormalizationForm)"/> that is a no-op on invalid Unicode (e.g. unpaired surrogates).</summary>
    private static string SafeNormalize(string text, NormalizationForm form)
    {
        try { return text.Normalize(form); }
        catch (ArgumentException) { return text; }
    }

    private static bool IsWhitespace(char c) => char.IsWhiteSpace(c);

    private static bool IsControl(char c)
    {
        if (c == '\t' || c == '\n' || c == '\r') return false;
        return char.IsControl(c);
    }

    private static bool IsPunctuation(char c)
    {
        // BERT treats all ASCII non-alphanumeric as punctuation, plus Unicode P* categories.
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126))
            return true;
        UnicodeCategory cat = char.GetUnicodeCategory(c);
        return cat is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation;
    }

    private static bool IsChineseChar(int cp)
    {
        // CJK Unicode blocks per BERT _is_chinese_char (unified ideographs + extensions A–E +
        // compatibility ideographs). Hangul / Hiragana / Katakana are deliberately excluded.
        return (cp >= 0x4E00 && cp <= 0x9FFF)
            || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x20000 && cp <= 0x2A6DF)
            || (cp >= 0x2A700 && cp <= 0x2B73F)
            || (cp >= 0x2B740 && cp <= 0x2B81F)
            || (cp >= 0x2B820 && cp <= 0x2CEAF)
            || (cp >= 0xF900 && cp <= 0xFAFF)
            || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }
}
