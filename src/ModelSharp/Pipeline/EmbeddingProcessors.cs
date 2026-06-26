using System;
using System.Collections.Generic;
using System.IO;
using ModelSharp.Tensors;
using ModelSharp.Text;

namespace ModelSharp.Pipeline;

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

    /// <summary>Builds the preprocessor from a processor context (manifest + engine input names).</summary>
    public TextEmbeddingPreprocessor(ProcessorContext ctx)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));
        _inputNames = ctx.InputNames;

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
/// sentence vector and L2-normalizes it, returning a <c>float[]</c>. Uses the attention
/// mask when it is present among the outputs; otherwise pools over all tokens. A model
/// that already emits a pooled vector <c>[1, H]</c> is L2-normalized as-is.
/// </summary>
public sealed class MeanPoolEmbeddingPostprocessor : IPostprocessor
{
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

        float[]? mask = TryGetMask(outputs);

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
