using System.Collections.Generic;
using System.Linq;
using ModelSharp.Text;
using Xunit;

namespace ModelSharp.Tests;

public class WordPieceTokenizerTests
{
    private static WordPieceTokenizer Make()
    {
        var vocab = new Dictionary<string, int>
        {
            ["[PAD]"] = 0, ["[UNK]"] = 1, ["[CLS]"] = 2, ["[SEP]"] = 3,
            ["un"] = 4, ["##aff"] = 5, ["##able"] = 6, ["play"] = 7, ["##ing"] = 8,
            ["hello"] = 9, ["world"] = 10, ["!"] = 11, ["low"] = 12, ["##er"] = 13, ["##s"] = 14,
            // Extra pieces exercised by the basic-tokenization tests below.
            ["don"] = 15, ["'"] = 16, ["t"] = 17, ["."] = 18,
            ["中"] = 19, ["文"] = 20, ["cafe"] = 21,
        };
        return new WordPieceTokenizer(vocab, lowercase: true);
    }

    [Fact]
    public void WordPiece_Splits_Known_Subwords()
    {
        var tok = Make();
        Assert.Equal(new[] { "un", "##aff", "##able" }, tok.TokenizeToPieces("unaffable"));
        Assert.Equal(new[] { "play", "##ing" }, tok.TokenizeToPieces("playing"));
        Assert.Equal(new[] { "low", "##er", "##s" }, tok.TokenizeToPieces("lowers"));
    }

    [Fact]
    public void Unknown_Word_Becomes_Unk()
    {
        Assert.Equal(new[] { "[UNK]" }, Make().TokenizeToPieces("unknownword"));
    }

    [Fact]
    public void Lowercases_And_Splits_Punctuation()
    {
        Assert.Equal(new[] { "hello", "world", "!" }, Make().TokenizeToPieces("Hello World!"));
    }

    [Fact]
    public void Encode_Adds_Cls_Sep_And_Maps_Ids()
    {
        Encoding e = Make().Encode("unaffable");
        Assert.Equal(new[] { "[CLS]", "un", "##aff", "##able", "[SEP]" }, e.Tokens.ToArray());
        Assert.Equal(new[] { 2, 4, 5, 6, 3 }, e.InputIds.ToArray());
        Assert.All(e.AttentionMask, m => Assert.Equal(1, m));
        Assert.All(e.TokenTypeIds, t => Assert.Equal(0, t));
    }

    [Fact]
    public void FromVocab_Assigns_Ids_By_Line_Index()
    {
        var tok = WordPieceTokenizer.FromVocab(new[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello" });
        Assert.True(tok.TryGetId("hello", out int id));
        Assert.Equal(4, id);
    }

    // ---- Basic tokenization (BERT BasicTokenizer) ----

    [Fact]
    public void Basic_Splits_Internal_Punctuation()
    {
        // Each punctuation char becomes its own token, splitting the surrounding run.
        Assert.Equal(
            new[] { "don", "'", "t", "." },
            new BasicTokenizer(lowercase: true).Tokenize("don't."));
    }

    [Fact]
    public void Basic_Splits_Cjk_Per_Character()
    {
        Assert.Equal(new[] { "中", "文" }, new BasicTokenizer(lowercase: true).Tokenize("中文"));
        // CJK chars are separated even when glued to Latin text.
        Assert.Equal(new[] { "ab", "中", "cd" }, new BasicTokenizer(lowercase: true).Tokenize("ab中cd"));
    }

    [Fact]
    public void Basic_Splits_Cjk_Regardless_Of_Case_Setting()
    {
        // BERT splits CJK and punctuation even when lowercasing is disabled.
        Assert.Equal(new[] { "中", "文" }, new BasicTokenizer(lowercase: false).Tokenize("中文"));
    }

    [Fact]
    public void Basic_Lowercases_And_Strips_Accents_When_Lowercasing()
    {
        Assert.Equal(new[] { "cafe" }, new BasicTokenizer(lowercase: true).Tokenize("Café"));
        Assert.Equal(new[] { "naive" }, new BasicTokenizer(lowercase: true).Tokenize("NaÏve"));
    }

    [Fact]
    public void Basic_Preserves_Case_And_Accents_When_Not_Lowercasing()
    {
        // lowercase:false keeps case and accents, but still splits punctuation.
        Assert.Equal(new[] { "Café", "!" }, new BasicTokenizer(lowercase: false).Tokenize("Café!"));
    }

    [Fact]
    public void Basic_Normalizes_Equivalent_Codepoints()
    {
        // "e" + combining acute (NFD) normalizes (NFC) and de-accents to the same as precomposed "é".
        string composed = "Café";    // 'é' as precomposed U+00E9
        string decomposed = "Café";        // "Cafe" + U+0301 COMBINING ACUTE ACCENT
        var basic = new BasicTokenizer(lowercase: true);
        Assert.Equal(basic.Tokenize(composed), basic.Tokenize(decomposed));
        Assert.Equal(new[] { "cafe" }, basic.Tokenize(decomposed));
    }

    // ---- Composition with WordPiece + Encode ----

    [Fact]
    public void WordPiece_Composes_With_Punctuation_Split()
    {
        Assert.Equal(new[] { "don", "'", "t", "." }, Make().TokenizeToPieces("Don't."));
    }

    [Fact]
    public void WordPiece_Composes_With_Cjk_Split()
    {
        Assert.Equal(new[] { "中", "文" }, Make().TokenizeToPieces("中文"));
    }

    [Fact]
    public void WordPiece_Composes_With_Accent_Stripping()
    {
        Assert.Equal(new[] { "cafe" }, Make().TokenizeToPieces("Café"));
    }

    [Fact]
    public void Encode_Splits_Cjk_With_Special_Tokens()
    {
        Encoding e = Make().Encode("中文");
        Assert.Equal(new[] { "[CLS]", "中", "文", "[SEP]" }, e.Tokens.ToArray());
        Assert.Equal(new[] { 2, 19, 20, 3 }, e.InputIds.ToArray());
        Assert.All(e.AttentionMask, m => Assert.Equal(1, m));
        Assert.All(e.TokenTypeIds, t => Assert.Equal(0, t));
    }

    [Fact]
    public void Encode_Splits_Punctuation_With_Special_Tokens()
    {
        Encoding e = Make().Encode("don't.");
        Assert.Equal(new[] { "[CLS]", "don", "'", "t", ".", "[SEP]" }, e.Tokens.ToArray());
        Assert.Equal(new[] { 2, 15, 16, 17, 18, 3 }, e.InputIds.ToArray());
    }
}
