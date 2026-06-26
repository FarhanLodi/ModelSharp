using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Text;
using Xunit;

namespace ModelSharp.Tests;

public class BpeTokenizerTests
{
    /// <summary>A vocab containing every one of the 256 byte-level characters, so any UTF-8 input encodes.</summary>
    private static Dictionary<string, int> ByteVocab()
    {
        var vocab = new Dictionary<string, int>();
        for (int b = 0; b < 256; b++)
        {
            string s = ByteLevel.EncodeByte((byte)b).ToString();
            if (!vocab.ContainsKey(s)) vocab[s] = vocab.Count;
        }
        return vocab;
    }

    /// <summary>
    /// Builds a tokenizer over the full byte vocab plus the supplied merges. Every merge result is
    /// added to the vocab so round-tripping always succeeds.
    /// </summary>
    private static BpeTokenizer Make(
        IEnumerable<string>? merges = null,
        IReadOnlyDictionary<string, int>? special = null,
        string? bos = null,
        string? eos = null)
    {
        Dictionary<string, int> vocab = ByteVocab();
        List<string> mergeList = (merges ?? Array.Empty<string>()).ToList();
        foreach (string rule in mergeList)
        {
            string[] parts = rule.Split(' ');
            string mergedTok = parts[0] + parts[1];
            if (!vocab.ContainsKey(mergedTok)) vocab[mergedTok] = vocab.Count;
        }
        return BpeTokenizer.FromVocabAndMerges(vocab, mergeList, special, bos, eos);
    }

    // ---- byte-level mapping ----

    [Fact]
    public void BytesToUnicode_Is_A_Bijection_Over_256_Codepoints()
    {
        IReadOnlyList<char> table = ByteLevel.ByteEncoderTable;
        Assert.Equal(256, table.Count);
        Assert.Equal(256, table.Distinct().Count()); // all distinct

        for (int b = 0; b < 256; b++)
        {
            char c = ByteLevel.EncodeByte((byte)b);
            Assert.True(ByteLevel.TryDecodeChar(c, out byte back));
            Assert.Equal((byte)b, back); // round-trips
        }
    }

    [Fact]
    public void Space_Maps_To_The_G_Marker()
    {
        // GPT-2's leading-space marker is U+0120 'Ġ' (byte 0x20 → 256 + 32).
        Assert.Equal((char)0x0120, ByteLevel.EncodeByte((byte)' '));
    }

    // ---- BPE merge sequence ----

    [Fact]
    public void Known_Word_Splits_Into_Expected_Merge_Sequence()
    {
        // Merges: (l,o)->"lo" rank 0, then (lo,w)->"low" rank 1. "lower" partially merges.
        BpeTokenizer tok = Make(new[] { "l o", "lo w" });
        Assert.Equal(new[] { "low", "e", "r" }, tok.TokenizeToPieces("lower").ToArray());

        // Ids follow the same sequence.
        Assert.True(tok.TryGetId("low", out int lowId));
        Assert.True(tok.TryGetId("e", out int eId));
        Assert.True(tok.TryGetId("r", out int rId));
        Assert.Equal(new[] { lowId, eId, rId }, tok.Encode("lower"));
    }

    [Fact]
    public void Bpe_With_No_Applicable_Merges_Is_One_Token_Per_Byte()
    {
        BpeTokenizer tok = Make(); // no merges
        Assert.Equal(new[] { "a", "b", "c" }, tok.TokenizeToPieces("abc").ToArray());
    }

    // ---- round-trip ----

    [Theory]
    [InlineData("Hello, world!")]
    [InlineData("  leading spaces and  doubled  ones")]
    [InlineData("héllo \U0001F600 世界 café")] // accents, emoji, CJK
    [InlineData("")]
    public void Decode_Encode_Round_Trips(string text)
    {
        BpeTokenizer tok = Make(new[] { "l o", "lo w" });
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void Leading_Space_Produces_G_Marked_Token()
    {
        BpeTokenizer tok = Make();
        string g = ByteLevel.EncodeByte((byte)' ').ToString(); // the 'Ġ' marker
        Assert.Equal(new[] { g, "h", "i" }, tok.TokenizeToPieces(" hi").ToArray());
    }

    // ---- special tokens ----

    [Fact]
    public void Special_Tokens_Are_Matched_Whole_And_Round_Trip()
    {
        var special = new Dictionary<string, int> { ["<|endoftext|>"] = 50000 };
        BpeTokenizer tok = Make(special: special);

        int[] ids = tok.Encode("Hello<|endoftext|>World");
        Assert.Contains(50000, ids);
        // The special id appears exactly once and as a single unit.
        Assert.Single(ids, i => i == 50000);

        Assert.Equal("Hello<|endoftext|>World", tok.Decode(ids, skipSpecial: false));
        Assert.Equal("HelloWorld", tok.Decode(ids, skipSpecial: true));
    }

    [Fact]
    public void AddSpecial_Wraps_With_Bos_And_Eos()
    {
        var special = new Dictionary<string, int> { ["<s>"] = 60000, ["</s>"] = 60001 };
        BpeTokenizer tok = Make(special: special, bos: "<s>", eos: "</s>");

        int[] ids = tok.Encode("hi", addSpecial: true);
        Assert.Equal(60000, ids[0]);
        Assert.Equal(60001, ids[^1]);

        Assert.Equal("hi", tok.Decode(ids));                       // skipSpecial defaults to true
        Assert.Equal("<s>hi</s>", tok.Decode(ids, skipSpecial: false));

        // Without addSpecial the wrappers are absent.
        Assert.Equal("hi", tok.Decode(tok.Encode("hi"), skipSpecial: false));
    }

    [Fact]
    public void VocabSize_Counts_Specials_Not_In_Base_Vocab()
    {
        var special = new Dictionary<string, int> { ["<|endoftext|>"] = 50000 };
        BpeTokenizer plain = Make();
        BpeTokenizer withSpecial = Make(special: special);
        Assert.Equal(256, plain.VocabSize);
        Assert.Equal(257, withSpecial.VocabSize);
    }

    // ---- pre-tokenization regex ----

    [Fact]
    public void PreTokenize_Splits_Contractions()
    {
        Assert.Equal(new[] { "don", "'t" }, BpeTokenizer.PreTokenize("don't").ToArray());
        Assert.Equal(new[] { "I", "'ve" }, BpeTokenizer.PreTokenize("I've").ToArray());
        Assert.Equal(new[] { "we", "'re", " here" }, BpeTokenizer.PreTokenize("we're here").ToArray());
    }

    [Fact]
    public void PreTokenize_Handles_Whitespace_Runs()
    {
        // A run of spaces: all but the last become a lone whitespace token; the last attaches to the
        // following word (GPT-2 behaviour).
        Assert.Equal(new[] { "hello", " ", " world" }, BpeTokenizer.PreTokenize("hello  world").ToArray());
        // Leading single space attaches to the word.
        Assert.Equal(new[] { " hello" }, BpeTokenizer.PreTokenize(" hello").ToArray());
        // Numbers and punctuation split out from letters.
        Assert.Equal(new[] { "abc", "123", "!?" }, BpeTokenizer.PreTokenize("abc123!?").ToArray());
    }

    // ---- file construction ----

    [Fact]
    public void FromFiles_Parses_Vocab_Json_And_Merges_Txt()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ms_bpe_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            Dictionary<string, int> vocab = ByteVocab();
            vocab["lo"] = vocab.Count;
            vocab["low"] = vocab.Count;
            string vocabPath = Path.Combine(dir, "vocab.json");
            string mergesPath = Path.Combine(dir, "merges.txt");

            string json = "{" + string.Join(",", vocab.Select(kv =>
                System.Text.Json.JsonSerializer.Serialize(kv.Key) + ":" + kv.Value)) + "}";
            File.WriteAllText(vocabPath, json);
            File.WriteAllText(mergesPath, "#version: 0.2\nl o\nlo w\n");

            BpeTokenizer tok = BpeTokenizer.FromFiles(vocabPath, mergesPath);
            Assert.Equal(new[] { "low", "e", "r" }, tok.TokenizeToPieces("lower").ToArray());
            Assert.Equal("lower", tok.Decode(tok.Encode("lower")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
