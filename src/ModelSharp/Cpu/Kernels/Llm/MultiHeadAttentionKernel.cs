using System;
using System.Threading.Tasks;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Llm;

/// <summary>
/// ONNXRuntime <c>com.microsoft.MultiHeadAttention</c> contrib op for the
/// <b>unpacked</b> Q/K/V form. Computes standard scaled-dot-product multi-head
/// attention:
/// <list type="bullet">
///   <item>reshape query/key/value [B, S, hidden] → [B, num_heads, S, head_dim];</item>
///   <item>scores = Q·Kᵀ · scale (scale defaults to 1/√head_dim);</item>
///   <item>optional additive attention mask;</item>
///   <item>softmax over the key axis, then ·V;</item>
///   <item>reshape the per-head context back to [B, S, hidden].</item>
/// </list>
///
/// <para><b>Supported inputs</b> (positional, ORT order):
/// <c>query</c> [B, Sq, hidden] (required), <c>key</c> [B, Skv, hidden],
/// <c>value</c> [B, Skv, v_hidden], optional <c>bias</c>,
/// optional <c>key_padding_mask</c>/attention mask (input index 4),
/// optional <c>past_key</c>/<c>past_value</c> (input indices 6/7).
/// The query/key/value hidden sizes may differ (value gives the output hidden).</para>
///
/// <para><b>Supported outputs</b>: <c>attention</c> [B, Sq, v_hidden] (required);
/// <c>present_key</c>/<c>present_value</c> when those outputs exist and
/// <c>past_key</c>/<c>past_value</c> are provided (basic concat along the
/// sequence axis).</para>
///
/// <para><b>Unsupported</b>: packed-QKV / packed-KV layouts (query carrying all
/// of Q,K,V, or 5-D inputs) and cross-attention relative-position bias throw a
/// <see cref="ModelSharpException"/>.</para>
/// </summary>
public sealed class MultiHeadAttentionKernel : IKernel
{
    public string OpType => "MultiHeadAttention";

    /// <summary>Serial below this many inner MACs (units × keys × head_dim); see GQA.</summary>
    private const long ParallelThreshold = 1L << 18;

    public void Execute(GraphNode node, GraphContext ctx)
    {
        int numHeads = (int)Attr.Int(node, "num_heads", 0);
        if (numHeads <= 0)
            throw new ModelSharpException("MultiHeadAttention requires a positive 'num_heads' attribute.");

        Tensor<float> query = ctx.Get(node.Inputs[0]);
        if (query.Shape.Rank != 3)
            throw new ModelSharpException(
                $"MultiHeadAttention supports only the unpacked 3-D query [B, S, hidden]; got rank {query.Shape.Rank}. " +
                "Packed-QKV (5-D) variants are not supported.");

        // key / value are required for the unpacked form.
        string keyName = node.Inputs.Count > 1 ? node.Inputs[1] : "";
        string valueName = node.Inputs.Count > 2 ? node.Inputs[2] : "";
        if (keyName.Length == 0 || valueName.Length == 0)
            throw new ModelSharpException(
                "MultiHeadAttention requires unpacked 'key' and 'value' inputs; packed-QKV/packed-KV is not supported.");

        Tensor<float> key = ctx.Get(keyName);
        Tensor<float> value = ctx.Get(valueName);
        if (key.Shape.Rank != 3 || value.Shape.Rank != 3)
            throw new ModelSharpException(
                "MultiHeadAttention supports only unpacked 3-D key/value [B, S, hidden]; packed/5-D layouts are not supported.");

        int batch = query.Shape[0];
        int qSeq = query.Shape[1];
        int qHidden = query.Shape[2];
        int kvSeq = key.Shape[1];
        int kHidden = key.Shape[2];
        int vHidden = value.Shape[2];

        if (key.Shape[0] != batch || value.Shape[0] != batch)
            throw new ModelSharpException("MultiHeadAttention query/key/value batch sizes disagree.");
        if (value.Shape[1] != kvSeq)
            throw new ModelSharpException("MultiHeadAttention key/value sequence lengths disagree.");
        if (qHidden % numHeads != 0 || kHidden % numHeads != 0 || vHidden % numHeads != 0)
            throw new ModelSharpException(
                $"MultiHeadAttention hidden sizes must be divisible by num_heads={numHeads} " +
                $"(q={qHidden}, k={kHidden}, v={vHidden}).");

        int qHead = qHidden / numHeads;
        int vHead = vHidden / numHeads;
        if (kHidden / numHeads != qHead)
            throw new ModelSharpException("MultiHeadAttention query and key head dimensions disagree.");

        // Optional fused bias [qHidden + kHidden + vHidden] applied to Q|K|V.
        Tensor<float>? bias = null;
        if (node.Inputs.Count > 3 && node.Inputs[3].Length > 0)
        {
            bias = ctx.Get(node.Inputs[3]);
            int expected = qHidden + kHidden + vHidden;
            if (bias.Shape.Length != expected)
                throw new ModelSharpException(
                    $"MultiHeadAttention bias must have {expected} elements (q+k+v hidden); got {bias.Shape.Length}.");
        }

        // Optional additive attention mask (input index 4). Supported broadcastable
        // float shapes: [B, Sq, Skv], [B, 1, Sq, Skv], [B, Skv], [Sq, Skv], [B, 1, 1, Skv].
        Tensor<float>? mask = null;
        if (node.Inputs.Count > 4 && node.Inputs[4].Length > 0)
            mask = ctx.Get(node.Inputs[4]);

        // Optional past_key / past_value (input indices 6 / 7), concatenated along seq.
        Tensor<float>? pastKey = node.Inputs.Count > 6 && node.Inputs[6].Length > 0 ? ctx.Get(node.Inputs[6]) : null;
        Tensor<float>? pastValue = node.Inputs.Count > 7 && node.Inputs[7].Length > 0 ? ctx.Get(node.Inputs[7]) : null;

        float scaleAttr = Attr.Float(node, "scale", 0f);
        float scale = scaleAttr != 0f ? scaleAttr : 1f / MathF.Sqrt(qHead);

        ReadOnlySpan<float> qSpan = query.Span;
        ReadOnlySpan<float> kSpan = key.Span;
        ReadOnlySpan<float> vSpan = value.Span;
        ReadOnlySpan<float> biasSpan = bias is null ? default : bias.Span;

        // Effective key/value (with past concatenated). totalSeq = pastSeq + kvSeq.
        int pastSeq = 0;
        if (pastKey is not null)
        {
            // past_key/value: [B, num_heads, pastSeq, head_dim].
            if (pastKey.Shape.Rank != 4 || pastKey.Shape[0] != batch || pastKey.Shape[1] != numHeads || pastKey.Shape[3] != qHead)
                throw new ModelSharpException("MultiHeadAttention past_key must be [B, num_heads, pastSeq, head_dim].");
            pastSeq = pastKey.Shape[2];
        }
        int totalSeq = pastSeq + kvSeq;

        // Build per-head K and V buffers [B, num_heads, totalSeq, head_dim] so we can
        // optionally emit present_key/present_value. K uses qHead, V uses vHead.
        var presentK = new Tensor<float>(new TensorShape(batch, numHeads, totalSeq, qHead));
        var presentV = new Tensor<float>(new TensorShape(batch, numHeads, totalSeq, vHead));
        Span<float> pk = presentK.Span, pv = presentV.Span;

        if (pastKey is not null)
        {
            ReadOnlySpan<float> psk = pastKey.Span;
            ReadOnlySpan<float> psv = pastValue is null ? default : pastValue.Span;
            if (pastValue is null)
                throw new ModelSharpException("MultiHeadAttention past_key was provided without past_value.");
            for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            for (int s = 0; s < pastSeq; s++)
            {
                int srcKBase = ((b * numHeads + h) * pastSeq + s) * qHead;
                int dstKBase = ((b * numHeads + h) * totalSeq + s) * qHead;
                for (int d = 0; d < qHead; d++) pk[dstKBase + d] = psk[srcKBase + d];
                int srcVBase = ((b * numHeads + h) * pastSeq + s) * vHead;
                int dstVBase = ((b * numHeads + h) * totalSeq + s) * vHead;
                for (int d = 0; d < vHead; d++) pv[dstVBase + d] = psv[srcVBase + d];
            }
        }

        // Scatter the new key/value [B, S, hidden] into the per-head present buffers.
        for (int b = 0; b < batch; b++)
        for (int s = 0; s < kvSeq; s++)
        for (int h = 0; h < numHeads; h++)
        {
            int kRow = (b * kvSeq + s) * kHidden + h * qHead;
            int dstKBase = ((b * numHeads + h) * totalSeq + pastSeq + s) * qHead;
            for (int d = 0; d < qHead; d++)
            {
                float kv = kSpan[kRow + d];
                if (bias is not null) kv += biasSpan[qHidden + h * qHead + d];
                pk[dstKBase + d] = kv;
            }
            int vRow = (b * kvSeq + s) * vHidden + h * vHead;
            int dstVBase = ((b * numHeads + h) * totalSeq + pastSeq + s) * vHead;
            for (int d = 0; d < vHead; d++)
            {
                float vv = vSpan[vRow + d];
                if (bias is not null) vv += biasSpan[qHidden + kHidden + h * vHead + d];
                pv[dstVBase + d] = vv;
            }
        }

        var outT = new Tensor<float>(new TensorShape(batch, qSeq, vHidden));

        // Fold the optional Q-bias into a dense per-head Q buffer [B, num_heads, Sq, qHead]
        // once, so the inner Q·K can be a plain SIMD dot (K/V bias is already folded into
        // present_key/value during the scatter above). Without bias this is just a repack.
        float[] qPacked = new float[(long)batch * numHeads * qSeq * qHead <= int.MaxValue
            ? batch * numHeads * qSeq * qHead : throw new ModelSharpException("MultiHeadAttention tensor too large.")];
        for (int b = 0; b < batch; b++)
        for (int h = 0; h < numHeads; h++)
        for (int qi = 0; qi < qSeq; qi++)
        {
            int qRow = (b * qSeq + qi) * qHidden + h * qHead;
            int dst = ((b * numHeads + h) * qSeq + qi) * qHead;
            for (int d = 0; d < qHead; d++)
            {
                float qv = qSpan[qRow + d];
                if (bias is not null) qv += biasSpan[h * qHead + d];
                qPacked[dst + d] = qv;
            }
        }

        // Parallelize over the independent (batch × head × query_position) units; each writes
        // a disjoint [vHead] output slice. Q·K and attn·V are SIMD-vectorized.
        float[] pkArr = KernelSimd.Array(presentK);
        float[] pvArr = KernelSimd.Array(presentV);
        float[] outArr = KernelSimd.Array(outT);
        Tensor<float>? maskT = mask;
        int totalUnits = batch * numHeads * qSeq;
        bool parallel = (long)totalUnits * totalSeq * Math.Max(qHead, vHead) >= ParallelThreshold;

        void Compute(int unit, float[] scores)
        {
            int qi = unit % qSeq;
            int hb = unit / qSeq;
            int h = hb % numHeads;
            int b = hb / numHeads;

            int qBase = ((b * numHeads + h) * qSeq + qi) * qHead;
            int kvHeadBase = (b * numHeads + h) * totalSeq;

            // scores[k] = scale * (Q · K_k) + mask.
            float mx = float.NegativeInfinity;
            for (int kj = 0; kj < totalSeq; kj++)
            {
                float dot = KernelSimd.Dot(qPacked, qBase, pkArr, (kvHeadBase + kj) * qHead, qHead);
                float sc = dot * scale + MaskAdd(maskT, batch, numHeads, qSeq, totalSeq, b, h, qi, kj);
                scores[kj] = sc;
                if (sc > mx) mx = sc;
            }

            float sum = 0f;
            for (int kj = 0; kj < totalSeq; kj++) { float e = MathF.Exp(scores[kj] - mx); scores[kj] = e; sum += e; }
            float inv = sum > 0f ? 1f / sum : 0f;

            int outRow = (b * qSeq + qi) * vHidden + h * vHead;
            System.Array.Clear(outArr, outRow, vHead);
            for (int kj = 0; kj < totalSeq; kj++)
                KernelSimd.AxpyInto(outArr, outRow, pvArr, (kvHeadBase + kj) * vHead, scores[kj] * inv, vHead);
        }

        if (parallel)
        {
            Parallel.For(0, totalUnits,
                () => new float[totalSeq],
                (unit, _, scores) => { Compute(unit, scores); return scores; },
                _ => { });
        }
        else
        {
            float[] scores = new float[totalSeq];
            for (int unit = 0; unit < totalUnits; unit++) Compute(unit, scores);
        }

        ctx.Set(node.Outputs[0], outT);
        if (node.Outputs.Count > 1 && node.Outputs[1].Length > 0) ctx.Set(node.Outputs[1], presentK);
        if (node.Outputs.Count > 2 && node.Outputs[2].Length > 0) ctx.Set(node.Outputs[2], presentV);
    }

    /// <summary>
    /// Additive-mask lookup that broadcasts the common ORT float-mask shapes to a
    /// scalar bias for (batch b, head h, query qi, key kj). Returns 0 if no mask.
    /// </summary>
    private static float MaskAdd(
        Tensor<float>? mask, int batch, int numHeads, int qSeq, int totalSeq,
        int b, int h, int qi, int kj)
    {
        if (mask is null) return 0f;
        ReadOnlySpan<int> d = mask.Shape.Dimensions;
        ReadOnlySpan<float> m = mask.Span;
        return d.Length switch
        {
            // [Skv] or [B] - simplest broadcasts.
            1 when d[0] == totalSeq => m[kj],
            // [B, Skv]
            2 when d[0] == batch && d[1] == totalSeq => m[b * totalSeq + kj],
            // [Sq, Skv]
            2 when d[0] == qSeq && d[1] == totalSeq => m[qi * totalSeq + kj],
            // [B, Sq, Skv]
            3 when d[0] == batch && d[1] == qSeq && d[2] == totalSeq
                => m[(b * qSeq + qi) * totalSeq + kj],
            // [B, num_heads, Sq, Skv]
            4 when d[0] == batch && d[1] == numHeads && d[2] == qSeq && d[3] == totalSeq
                => m[((b * numHeads + h) * qSeq + qi) * totalSeq + kj],
            // [B, 1, Sq, Skv]
            4 when d[0] == batch && d[1] == 1 && d[2] == qSeq && d[3] == totalSeq
                => m[(b * qSeq + qi) * totalSeq + kj],
            // [B, 1, 1, Skv]
            4 when d[0] == batch && d[1] == 1 && d[2] == 1 && d[3] == totalSeq
                => m[b * totalSeq + kj],
            _ => throw new ModelSharpException(
                $"MultiHeadAttention unsupported mask shape {mask.Shape} for B={batch}, H={numHeads}, Sq={qSeq}, Skv={totalSeq}."),
        };
    }
}
