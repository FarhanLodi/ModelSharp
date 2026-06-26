using System.Collections.Generic;

namespace ModelSharp.Text;

/// <summary>The result of tokenizing text: model-ready id arrays plus the string tokens.</summary>
public sealed class Encoding
{
    /// <summary>Token ids, including any special tokens.</summary>
    public IReadOnlyList<int> InputIds { get; }

    /// <summary>1 for real tokens, 0 for padding.</summary>
    public IReadOnlyList<int> AttentionMask { get; }

    /// <summary>Segment ids (0 for the first/only sequence).</summary>
    public IReadOnlyList<int> TokenTypeIds { get; }

    /// <summary>The string tokens corresponding to <see cref="InputIds"/>.</summary>
    public IReadOnlyList<string> Tokens { get; }

    public Encoding(
        IReadOnlyList<int> inputIds,
        IReadOnlyList<int> attentionMask,
        IReadOnlyList<int> tokenTypeIds,
        IReadOnlyList<string> tokens)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
        TokenTypeIds = tokenTypeIds;
        Tokens = tokens;
    }
}
