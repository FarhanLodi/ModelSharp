using System;
using System.Collections.Generic;
using System.IO;
using ModelSharp.Tensors;
using ModelSharp.Text;

namespace ModelSharp.Pipeline;

/// <summary>
/// A one-slot holder that lets the embedding pre/post processors cooperate within a single
/// pipeline: the preprocessor records the attention mask it emitted (the model <em>input</em>
/// mask), and the postprocessor reads it back so mean-pooling skips padding even when the model
/// does not echo the mask among its outputs. Single-call, single-threaded usage is assumed.
/// </summary>
public sealed class AttentionMaskHolder
{
    /// <summary>The most recently emitted attention mask (1 = real token, 0 = padding), or null.</summary>
    public float[]? LastMask { get; set; }
}

/// <summary>
/// Text → feed-tensors for sentence-embedding models (BERT family). Builds a
/// <see cref="WordPieceTokenizer"/> from the vocab named in the manifest
/// (<c>Manifest.Extra["vocab"]</c>), tokenizes the input string, and emits
/// <c>input_ids</c> / <c>attention_mask</c> / <c>token_type_ids</c> as <c>Tensor&lt;long&gt;</c>
/// of shape <c>[1, S]</c> — but only for the feed names the engine actually declares.
/// </summary>
public sealed class TextEmbeddingPreprocessor : IPreprocessor
{
    private readonly WordPieceTokenizer _tokenizer;
    private readonly IReadOnlyList<string> _inputNames;
    private readonly AttentionMaskHolder? _maskHolder;

    /// <summary>Builds the preprocessor from a processor context (manifest + engine input names).</summary>
    public TextEmbeddingPreprocessor(ProcessorContext ctx)
        : this(ctx, null)
    {
    }

    /// <summary>
    /// Builds the preprocessor, optionally sharing an <see cref="AttentionMaskHolder"/> with the
    /// matching postprocessor so padded inputs are mean-pooled over real tokens only.
    /// </summary>
    public TextEmbeddingPreprocessor(ProcessorContext ctx, AttentionMaskHolder? maskHolder)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));
        _inputNames = ctx.InputNames;
        _maskHolder = maskHolder;

        if (!ctx.Manifest.Extra.TryGetValue("vocab", out string? vocabPath) || string.IsNullOrWhiteSpace(vocabPath))
            throw new ModelSharpException(
                "Text embedding requires a WordPiece vocab; set manifest Extra[\"vocab\"] to a vocab.txt path.");
        if (!File.Exists(vocabPath))
            throw new ModelSharpException($"Vocab file not found for text embedding: '{vocabPath}'.");

        bool lowercase = true;
        if (ctx.Manifest.Extra.TryGetValue("lowercase", out string? lc) && bool.TryParse(lc, out bool parsed))
            lowercase = parsed;

        _tokenizer = WordPieceTokenizer.FromVocab(File.ReadLines(vocabPath), lowercase);
    }

    /// <summary>Tokenizes the input string and produces the model's feed tensors.</summary>
    public IReadOnlyDictionary<string, NamedTensor> ToFeeds(object input)
    {
        if (input is not string text)
            throw new ModelSharpException(
                $"TextEmbeddingPreprocessor expects a string input, got '{input?.GetType().Name ?? "null"}'.");

        Encoding enc = _tokenizer.Encode(text);
        int s = enc.InputIds.Count;
        var ids = new long[s];
        var mask = new long[s];
        var types = new long[s];
        for (int i = 0; i < s; i++)
        {
            ids[i] = enc.InputIds[i];
            mask[i] = enc.AttentionMask[i];
            types[i] = enc.TokenTypeIds[i];
        }

        // Record the input mask so the cooperating postprocessor can pool over real tokens only.
        if (_maskHolder is not null)
        {
            var maskF = new float[s];
            for (int i = 0; i < s; i++) maskF[i] = mask[i];
            _maskHolder.LastMask = maskF;
        }

        var shape = new TensorShape(1, s);
        var feeds = new Dictionary<string, NamedTensor>();
        foreach (string name in _inputNames)
        {
            long[]? data = name switch
            {
                "input_ids" or "input.1" or "ids" => ids,
                "attention_mask" or "input_mask" or "mask" => mask,
                "token_type_ids" or "segment_ids" or "token_type" => types,
                _ => null,
            };
            if (data is not null)
                feeds[name] = new NamedTensor(name, new Tensor<long>(shape, data));
        }

        if (feeds.Count == 0)
            throw new ModelSharpException(
                "Could not map any engine input to token tensors; expected an input named 'input_ids'. " +
                $"Engine inputs: [{string.Join(", ", _inputNames)}].");

        return feeds;
    }
}

/// <summary>
/// Mean-pools a transformer's token-level hidden state <c>[1, S, H]</c> into a single
/// sentence vector and L2-normalizes it, returning a <c>float[]</c>. The attention mask is taken
/// (in order) from the cooperating preprocessor's recorded <em>input</em> mask, then from an
/// <c>attention_mask</c> model <em>output</em> if present; failing both, all tokens are pooled.
/// A model that already emits a pooled vector <c>[1, H]</c> is L2-normalized as-is.
/// </summary>
public sealed class MeanPoolEmbeddingPostprocessor : IPostprocessor
{
    private readonly AttentionMaskHolder? _maskHolder;

    /// <summary>Builds a postprocessor that relies solely on an output mask (or pools all tokens).</summary>
    public MeanPoolEmbeddingPostprocessor()
        : this(null)
    {
    }

    /// <summary>
    /// Builds a postprocessor that prefers the input mask recorded by the cooperating
    /// <see cref="TextEmbeddingPreprocessor"/> via the shared <see cref="AttentionMaskHolder"/>.
    /// </summary>
    public MeanPoolEmbeddingPostprocessor(AttentionMaskHolder? maskHolder)
    {
        _maskHolder = maskHolder;
    }

    /// <summary>Decodes engine outputs into an L2-normalized embedding vector.</summary>
    public object Decode(IReadOnlyDictionary<string, NamedTensor> outputs)
    {
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));

        // Prefer a rank-3 [B, S, H] hidden state; fall back to the first float output.
        Tensor<float>? hidden = null;
        foreach (NamedTensor nt in outputs.Values)
        {
            if (nt.Tensor is Tensor<float> tf)
            {
                if (tf.Shape.Rank == 3) { hidden = tf; break; }
                hidden ??= tf;
            }
        }
        if (hidden is null)
            throw new ModelSharpException("No float output found to pool into an embedding.");

        // Precedence: input mask (recorded by the preprocessor) → output mask → pool all tokens.
        float[]? mask = _maskHolder?.LastMask ?? TryGetMask(outputs);

        int h = hidden.Shape.Dimensions[^1];
        if (h == 0) throw new ModelSharpException("Embedding output has a zero-sized hidden dimension.");
        int rows = checked((int)(hidden.Length / h));

        var pooled = new float[h];
        ReadOnlySpan<float> span = hidden.Span;
        float denom = 0f;
        for (int r = 0; r < rows; r++)
        {
            float m = mask is not null && r < mask.Length ? mask[r] : 1f;
            denom += m;
            int baseIdx = r * h;
            for (int k = 0; k < h; k++) pooled[k] += span[baseIdx + k] * m;
        }
        float inv = 1f / MathF.Max(denom, 1f);
        for (int k = 0; k < h; k++) pooled[k] *= inv;

        // L2-normalize.
        double norm = 0;
        for (int k = 0; k < h; k++) norm += (double)pooled[k] * pooled[k];
        norm = Math.Sqrt(norm);
        if (norm > 0)
            for (int k = 0; k < h; k++) pooled[k] = (float)(pooled[k] / norm);

        return pooled;
    }

    private static float[]? TryGetMask(IReadOnlyDictionary<string, NamedTensor> outputs)
    {
        if (!outputs.TryGetValue("attention_mask", out NamedTensor? nt)) return null;
        switch (nt.Tensor)
        {
            case Tensor<long> ml:
            {
                ReadOnlySpan<long> src = ml.Span;
                var m = new float[src.Length];
                for (int i = 0; i < src.Length; i++) m[i] = src[i];
                return m;
            }
            case Tensor<int> mi:
            {
                ReadOnlySpan<int> src = mi.Span;
                var m = new float[src.Length];
                for (int i = 0; i < src.Length; i++) m[i] = src[i];
                return m;
            }
            case Tensor<float> mf:
                return mf.Span.ToArray();
            default:
                return null;
        }
    }
}
