using System;

namespace ModelSharp.Generation;

/// <summary>
/// Decoding parameters for autoregressive text generation. Mirrors the knobs exposed by
/// the Hugging Face <c>generate</c> API so that values copied from a model card behave the
/// same way here. All fields are immutable; build variants with <c>with</c> expressions.
/// </summary>
public sealed record GenerationConfig
{
    /// <summary>Maximum number of <em>new</em> tokens to produce (excludes the prompt). Default 20.</summary>
    public int MaxNewTokens { get; init; } = 20;

    /// <summary>
    /// Optional hard cap on the <em>total</em> sequence length (prompt + generated). <c>0</c> disables
    /// the cap, leaving <see cref="MaxNewTokens"/> as the sole length bound.
    /// </summary>
    public int MaxLength { get; init; }

    /// <summary>
    /// Softmax temperature applied before sampling. <c>1.0</c> is a no-op; <c>&lt;1</c> sharpens the
    /// distribution (more deterministic), <c>&gt;1</c> flattens it. Must be &gt; 0. Ignored when greedy.
    /// </summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Keep only the <c>K</c> highest-probability tokens before sampling. <c>0</c> disables Top-K.</summary>
    public int TopK { get; init; }

    /// <summary>
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative probability reaches
    /// <c>TopP</c>. <c>1.0</c> disables Top-P. Valid range (0, 1].
    /// </summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>
    /// Penalty applied to logits of tokens already present in the context. <c>1.0</c> disables it;
    /// values &gt; 1 discourage repetition (HF convention: positive logits are divided, negative
    /// logits are multiplied by the penalty). Must be &gt; 0.
    /// </summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>
    /// Tokens that terminate generation. When the next token equals any of these, it is appended
    /// and the loop stops. <c>null</c> means "no EOS" (run until a length bound is hit).
    /// </summary>
    public int[]? EosTokenIds { get; init; }

    /// <summary>Padding token id. Reserved for batched/padded scenarios; unused for single-sequence runs.</summary>
    public int PadTokenId { get; init; }

    /// <summary><c>false</c> (default) selects greedy argmax decoding; <c>true</c> enables stochastic sampling.</summary>
    public bool DoSample { get; init; }

    /// <summary>
    /// Optional RNG seed for reproducible sampling. With a fixed seed the same prompt and model
    /// produce the same sequence. Ignored when <see cref="DoSample"/> is <c>false</c>.
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>Convenience factory for a greedy (argmax) configuration.</summary>
    /// <param name="maxNewTokens">Maximum number of new tokens to generate.</param>
    /// <param name="eosTokenIds">Optional end-of-sequence token ids.</param>
    public static GenerationConfig Greedy(int maxNewTokens, int[]? eosTokenIds = null) =>
        new() { MaxNewTokens = maxNewTokens, DoSample = false, EosTokenIds = eosTokenIds };

    /// <summary>Throws if any field is outside its valid range. Called by <see cref="TextGenerator"/>.</summary>
    public void Validate()
    {
        if (MaxNewTokens < 0)
            throw new ArgumentOutOfRangeException(nameof(MaxNewTokens), MaxNewTokens, "Must be >= 0.");
        if (MaxLength < 0)
            throw new ArgumentOutOfRangeException(nameof(MaxLength), MaxLength, "Must be >= 0 (0 disables).");
        if (DoSample && Temperature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature, "Must be > 0 when sampling.");
        if (TopK < 0)
            throw new ArgumentOutOfRangeException(nameof(TopK), TopK, "Must be >= 0 (0 disables).");
        if (TopP <= 0f || TopP > 1f)
            throw new ArgumentOutOfRangeException(nameof(TopP), TopP, "Must be in (0, 1].");
        if (RepetitionPenalty <= 0f)
            throw new ArgumentOutOfRangeException(nameof(RepetitionPenalty), RepetitionPenalty, "Must be > 0.");
    }
}
