using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Llm;

/// <summary>
/// ONNXRuntime <c>com.microsoft.GroupQueryAttention</c> contrib op. Grouped-query
/// attention: <c>num_heads</c> query heads share <c>kv_num_heads</c> key/value
/// heads, so each KV head serves <c>num_heads / kv_num_heads</c> query heads
/// (repeat-KV). GQA is <b>causal</b>: query position <c>i</c> may only attend to
/// key positions <c>≤ pastSeq + i</c>.
///
/// <para><b>Supported inputs</b> (positional, ORT order):
/// <c>query</c> [B, Sq, num_heads·head_dim] (required),
/// <c>key</c> [B, Skv, kv_num_heads·head_dim],
/// <c>value</c> [B, Skv, kv_num_heads·head_dim],
/// optional <c>past_key</c>/<c>past_value</c> [B, kv_num_heads, pastSeq, head_dim]
/// (input indices 3/4). <c>seqlens_k</c> (5), <c>total_sequence_length</c> (6),
/// <c>cos_cache</c> (7) and <c>sin_cache</c> (8) are ignored — rotary embedding is
/// assumed already applied to Q/K (or absent).</para>
///
/// <para><b>Outputs</b>: <c>output</c> [B, Sq, num_heads·head_dim],
/// <c>present_key</c>/<c>present_value</c> [B, kv_num_heads, totalSeq, head_dim]
/// (past concatenated with the new K/V along the sequence axis).</para>
///
/// <para><b>Assumptions</b>: causal masking; no rotary applied here; scale defaults
/// to 1/√head_dim (overridable via the <c>scale</c> attribute). batch ≥ 1 is fine.</para>
/// </summary>
public sealed class GroupQueryAttentionKernel : IKernel
{
    public string OpType => "GroupQueryAttention";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        int numHeads = (int)Attr.Int(node, "num_heads", 0);
        int kvNumHeads = (int)Attr.Int(node, "kv_num_heads", 0);
        if (numHeads <= 0 || kvNumHeads <= 0)
            throw new ModelSharpException("GroupQueryAttention requires positive 'num_heads' and 'kv_num_heads'.");
        if (numHeads % kvNumHeads != 0)
            throw new ModelSharpException(
                $"GroupQueryAttention num_heads={numHeads} must be a multiple of kv_num_heads={kvNumHeads}.");
        int groupSize = numHeads / kvNumHeads;

        Tensor<float> query = ctx.Get(node.Inputs[0]);
        if (query.Shape.Rank != 3)
            throw new ModelSharpException(
                $"GroupQueryAttention supports only unpacked 3-D query [B, S, hidden]; got rank {query.Shape.Rank}.");

        string keyName = node.Inputs.Count > 1 ? node.Inputs[1] : "";
        string valueName = node.Inputs.Count > 2 ? node.Inputs[2] : "";
        if (keyName.Length == 0 || valueName.Length == 0)
            throw new ModelSharpException(
                "GroupQueryAttention requires unpacked 'key' and 'value' inputs; packed-QKV is not supported.");

        Tensor<float> key = ctx.Get(keyName);
        Tensor<float> value = ctx.Get(valueName);
        if (key.Shape.Rank != 3 || value.Shape.Rank != 3)
            throw new ModelSharpException("GroupQueryAttention supports only unpacked 3-D key/value.");

        int batch = query.Shape[0];
        int qSeq = query.Shape[1];
        int qHidden = query.Shape[2];
        int kvSeq = key.Shape[1];
        int kvHidden = key.Shape[2];

        if (key.Shape[0] != batch || value.Shape[0] != batch)
            throw new ModelSharpException("GroupQueryAttention query/key/value batch sizes disagree.");
        if (value.Shape[1] != kvSeq || value.Shape[2] != kvHidden)
            throw new ModelSharpException("GroupQueryAttention key/value shapes disagree.");
        if (qHidden % numHeads != 0)
            throw new ModelSharpException(
                $"GroupQueryAttention query hidden {qHidden} not divisible by num_heads={numHeads}.");
        int headDim = qHidden / numHeads;
        if (kvHidden != kvNumHeads * headDim)
            throw new ModelSharpException(
                $"GroupQueryAttention key/value hidden {kvHidden} must equal kv_num_heads·head_dim = {kvNumHeads * headDim}.");

        // Optional past_key / past_value: [B, kv_num_heads, pastSeq, head_dim].
        Tensor<float>? pastKey = node.Inputs.Count > 3 && node.Inputs[3].Length > 0 ? ctx.Get(node.Inputs[3]) : null;
        Tensor<float>? pastValue = node.Inputs.Count > 4 && node.Inputs[4].Length > 0 ? ctx.Get(node.Inputs[4]) : null;
        int pastSeq = 0;
        if (pastKey is not null)
        {
            if (pastValue is null)
                throw new ModelSharpException("GroupQueryAttention past_key provided without past_value.");
            if (pastKey.Shape.Rank != 4 || pastKey.Shape[0] != batch ||
                pastKey.Shape[1] != kvNumHeads || pastKey.Shape[3] != headDim)
                throw new ModelSharpException(
                    "GroupQueryAttention past_key must be [B, kv_num_heads, pastSeq, head_dim].");
            pastSeq = pastKey.Shape[2];
        }
        int totalSeq = pastSeq + kvSeq;

        float scaleAttr = Attr.Float(node, "scale", 0f);
        float scale = scaleAttr != 0f ? scaleAttr : 1f / MathF.Sqrt(headDim);

        ReadOnlySpan<float> qSpan = query.Span;
        ReadOnlySpan<float> kSpan = key.Span;
        ReadOnlySpan<float> vSpan = value.Span;

        // present_key / present_value: [B, kv_num_heads, totalSeq, head_dim].
        var presentK = new Tensor<float>(new TensorShape(batch, kvNumHeads, totalSeq, headDim));
        var presentV = new Tensor<float>(new TensorShape(batch, kvNumHeads, totalSeq, headDim));
        Span<float> pk = presentK.Span, pv = presentV.Span;

        if (pastKey is not null)
        {
            ReadOnlySpan<float> psk = pastKey.Span;
            ReadOnlySpan<float> psv = pastValue!.Span;
            for (int b = 0; b < batch; b++)
            for (int g = 0; g < kvNumHeads; g++)
            for (int s = 0; s < pastSeq; s++)
            {
                int src = ((b * kvNumHeads + g) * pastSeq + s) * headDim;
                int dst = ((b * kvNumHeads + g) * totalSeq + s) * headDim;
                for (int d = 0; d < headDim; d++) { pk[dst + d] = psk[src + d]; pv[dst + d] = psv[src + d]; }
            }
        }

        // Scatter new K/V [B, Skv, kv_num_heads·head_dim] into present at offset pastSeq.
        for (int b = 0; b < batch; b++)
        for (int s = 0; s < kvSeq; s++)
        for (int g = 0; g < kvNumHeads; g++)
        {
            int src = (b * kvSeq + s) * kvHidden + g * headDim;
            int dst = ((b * kvNumHeads + g) * totalSeq + pastSeq + s) * headDim;
            for (int d = 0; d < headDim; d++) { pk[dst + d] = kSpan[src + d]; pv[dst + d] = vSpan[src + d]; }
        }

        var outT = new Tensor<float>(new TensorShape(batch, qSeq, qHidden));
        Span<float> outSpan = outT.Span;
        Span<float> scores = new float[totalSeq];

        for (int b = 0; b < batch; b++)
        for (int h = 0; h < numHeads; h++)
        {
            int g = h / groupSize;   // shared KV head for this query head.
            for (int qi = 0; qi < qSeq; qi++)
            {
                int qRow = (b * qSeq + qi) * qHidden + h * headDim;
                // Causal: query position qi corresponds to absolute position pastSeq+qi,
                // so it may attend to key positions 0..pastSeq+qi inclusive.
                int kLimit = pastSeq + qi;

                float mx = float.NegativeInfinity;
                for (int kj = 0; kj <= kLimit; kj++)
                {
                    int kBase = ((b * kvNumHeads + g) * totalSeq + kj) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qSpan[qRow + d] * pk[kBase + d];
                    float sc = dot * scale;
                    scores[kj] = sc;
                    if (sc > mx) mx = sc;
                }

                float sum = 0f;
                for (int kj = 0; kj <= kLimit; kj++) { float e = MathF.Exp(scores[kj] - mx); scores[kj] = e; sum += e; }
                float inv = sum > 0f ? 1f / sum : 0f;

                int outRow = (b * qSeq + qi) * qHidden + h * headDim;
                for (int d = 0; d < headDim; d++) outSpan[outRow + d] = 0f;
                for (int kj = 0; kj <= kLimit; kj++)
                {
                    float w = scores[kj] * inv;
                    int vBase = ((b * kvNumHeads + g) * totalSeq + kj) * headDim;
                    for (int d = 0; d < headDim; d++) outSpan[outRow + d] += w * pv[vBase + d];
                }
            }
        }

        ctx.Set(node.Outputs[0], outT);
        if (node.Outputs.Count > 1 && node.Outputs[1].Length > 0) ctx.Set(node.Outputs[1], presentK);
        if (node.Outputs.Count > 2 && node.Outputs[2].Length > 0) ctx.Set(node.Outputs[2], presentV);
    }
}
