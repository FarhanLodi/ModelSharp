using System;
using System.Linq;
using ModelSharp.Audio;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class CtcDecoderTests
{
    private static Tensor<float> Emissions(int frames, int vocab, float[] data)
    {
        Assert.Equal(frames * vocab, data.Length);
        return new Tensor<float>(new TensorShape(frames, vocab), data);
    }

    // ---------------------------------------------------------------------------------------
    // Greedy (best-path) decoding
    // ---------------------------------------------------------------------------------------

    [Fact]
    public void Greedy_Collapses_Duplicates_And_Removes_Blank()
    {
        // vocab: 0 = blank, 1 = 'a', 2 = 'b'. Argmax path: [a, a, blank, b, b] -> "ab".
        var e = Emissions(5, 3, new float[]
        {
            0, 5, 0,   // a
            0, 5, 0,   // a  (duplicate -> collapsed)
            5, 0, 0,   // blank (removed)
            0, 0, 5,   // b
            0, 0, 5,   // b  (duplicate -> collapsed)
        });

        Assert.Equal(new[] { 1, 2 }, CtcDecoder.GreedyDecode(e));
    }

    [Fact]
    public void Greedy_Honors_Configurable_Blank_Index()
    {
        // Same shape, but now blank = index 2. Argmax path: [0, 0, blank, 1, 1] -> [0, 1].
        var e = Emissions(5, 3, new float[]
        {
            5, 0, 0,   // 0
            5, 0, 0,   // 0 (collapsed)
            0, 0, 5,   // blank (index 2, removed)
            0, 5, 0,   // 1
            0, 5, 0,   // 1 (collapsed)
        });

        Assert.Equal(new[] { 0, 1 }, CtcDecoder.GreedyDecode(e, blank: 2));
        // With the default blank (0) the leading 0s would be stripped: a different result.
        Assert.Equal(new[] { 2, 1 }, CtcDecoder.GreedyDecode(e, blank: 0));
    }

    [Fact]
    public void Greedy_Repeat_Across_Blank_Is_Kept()
    {
        // Path [a, blank, a] must decode to "aa" (the blank separates the two a's).
        var e = Emissions(3, 2, new float[]
        {
            0, 5,   // a
            5, 0,   // blank
            0, 5,   // a
        });

        Assert.Equal(new[] { 1, 1 }, CtcDecoder.GreedyDecode(e));
    }

    [Fact]
    public void Greedy_To_Text_Uses_Vocabulary()
    {
        var vocab = new CtcVocabulary(new[] { "<pad>", "a", "b" });
        var e = Emissions(5, 3, new float[]
        {
            0, 5, 0,
            0, 5, 0,
            5, 0, 0,
            0, 0, 5,
            0, 0, 5,
        });

        Assert.Equal("ab", CtcDecoder.GreedyDecode(e, vocab));
    }

    // ---------------------------------------------------------------------------------------
    // Prefix beam search
    // ---------------------------------------------------------------------------------------

    [Fact]
    public void Beam_Matches_Greedy_On_Peaky_Emissions()
    {
        // Near one-hot frames: A, A, blank, B, B -> "AB". Best path == best labeling.
        var e = Emissions(5, 3, new float[]
        {
            0.02f, 0.96f, 0.02f,
            0.02f, 0.96f, 0.02f,
            0.96f, 0.02f, 0.02f,
            0.02f, 0.02f, 0.96f,
            0.02f, 0.02f, 0.96f,
        });

        int[] greedy = CtcDecoder.GreedyDecode(e);
        var beam = CtcDecoder.BeamSearchDecode(e, beamWidth: 25, inputKind: CtcInputKind.Probabilities);

        Assert.Equal(new[] { 1, 2 }, greedy);
        Assert.Equal(greedy, beam[0].Tokens);
    }

    [Fact]
    public void Beam_Finds_Higher_Probability_Labeling_Than_Greedy()
    {
        // vocab: 0 = blank, 1 = A, 2 = B. Each frame: blank=0.4, A=0.35, B=0.25 over 3 frames.
        // Best single PATH is blank,blank,blank -> "" (0.4^3 = 0.064), so greedy returns [].
        // Best LABELING summed over all collapsing paths is "A". Enumerating the 8 paths over
        // {blank=0.4, A=0.35} that collapse to the single-symbol "A" (the A's must be contiguous):
        //   bbA + bAb + Abb + bAA + AAb + AAA = 0.056*3 + 0.049*2 + 0.042875 = 0.308875
        // (AbA -> "AA", excluded). That beats "B" (0.185625) and "" (0.064), so beam recovers [A] = [1].
        var e = Emissions(3, 3, new float[]
        {
            0.40f, 0.35f, 0.25f,
            0.40f, 0.35f, 0.25f,
            0.40f, 0.35f, 0.25f,
        });

        int[] greedy = CtcDecoder.GreedyDecode(e);
        Assert.Empty(greedy); // best path is all-blank

        var nbest = CtcDecoder.BeamSearchDecode(
            e, beamWidth: 25, inputKind: CtcInputKind.Probabilities, topN: 25);

        // Beam's best labeling differs from greedy and is "A".
        Assert.Equal(new[] { 1 }, nbest[0].Tokens);
        Assert.NotEqual(greedy, nbest[0].Tokens);

        // Score matches the analytic total probability of labeling "A".
        Assert.Equal(Math.Log(0.308875), nbest[0].LogProbability, 3);

        // The empty labeling is present but ranked strictly below "A".
        CtcHypothesis empty = nbest.First(h => h.Tokens.Count == 0);
        Assert.True(nbest[0].LogProbability > empty.LogProbability);
        Assert.Equal(Math.Log(0.064), empty.LogProbability, 3);
    }

    [Fact]
    public void Beam_Accepts_Logits_And_Top_Equals_Argmax_When_Peaky()
    {
        // Same peaky case expressed as logits; log-softmax happens internally.
        var e = Emissions(4, 3, new float[]
        {
            0, 8, 0,   // A
            8, 0, 0,   // blank
            0, 0, 8,   // B
            0, 0, 8,   // B
        });

        var beam = CtcDecoder.BeamSearchDecode(e, beamWidth: 16, inputKind: CtcInputKind.Logits);
        Assert.Equal(new[] { 1, 2 }, beam[0].Tokens);
    }

    // ---------------------------------------------------------------------------------------
    // Vocabulary
    // ---------------------------------------------------------------------------------------

    [Fact]
    public void Vocabulary_Renders_Word_Delimiter_As_Space()
    {
        var vocab = new CtcVocabulary(new[] { "<pad>", "|", "H", "I", "<unk>" });

        // 2=H, 3=I, 1=| -> space, 2=H, 3=I
        Assert.Equal("HI HI", vocab.Decode(new[] { 2, 3, 1, 2, 3 }));
    }

    [Fact]
    public void Vocabulary_Suppresses_Special_Tokens()
    {
        var vocab = new CtcVocabulary(new[] { "<pad>", "|", "H", "I", "<unk>" });

        // <pad> and <unk> render as empty.
        Assert.Equal("HI", vocab.Decode(new[] { 0, 2, 3, 4 }));
    }

    [Fact]
    public void Vocabulary_Out_Of_Range_Index_Throws()
    {
        var vocab = new CtcVocabulary(new[] { "<pad>", "a", "b" });
        Assert.Throws<ArgumentOutOfRangeException>(() => vocab.Decode(new[] { 0, 1, 99 }));
    }

    [Fact]
    public void Vocabulary_Custom_Specials_And_Replacement()
    {
        var vocab = new CtcVocabulary(
            new[] { "_", "X", "a", "b" },
            wordDelimiter: "X",
            wordDelimiterReplacement: "-",
            specialTokens: new[] { "_" });

        // 2=a, 1=X -> "-", 3=b, 0=_ -> empty
        Assert.Equal("a-b", vocab.Decode(new[] { 2, 1, 3, 0 }));
    }
}
