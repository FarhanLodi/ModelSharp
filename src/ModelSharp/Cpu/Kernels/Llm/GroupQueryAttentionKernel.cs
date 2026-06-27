using System;
using System.Threading.Tasks;
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
/// <para><b>Inputs</b> (positional, ORT order):
/// 0 <c>query</c>, 1 <c>key</c>, 2 <c>value</c>, 3 <c>past_key</c>, 4 <c>past_value</c>,
/// 5 <c>seqlens_k</c> (int32 [B], total key length − 1 per batch),
/// 6 <c>total_sequence_length</c> (int32 scalar),
/// 7 <c>cos_cache</c>, 8 <c>sin_cache</c> (each [max_seq, rotary_dim/2]).</para>
///
/// <para>Two input layouts are supported:</para>
/// <list type="bullet">
/// <item><description><b>Unpacked</b>: <c>key</c>/<c>value</c> names supplied. <c>query</c>
/// is [B, Sq, num_heads·head_dim]; <c>key</c>/<c>value</c> are
/// [B, Skv, kv_num_heads·head_dim].</description></item>
/// <item><description><b>Packed-QKV</b> (onnxruntime-genai INT4 exports): <c>key</c> and
/// <c>value</c> input names are empty; <c>query</c> holds Q,K,V concatenated on the last
/// dim with shape [B, S, (num_heads + 2·kv_num_heads)·head_dim] and is split into Q
/// [B,S,num_heads·head_dim], K/V [B,S,kv_num_heads·head_dim] each.</description></item>
/// </list>
///
/// <para><b>In-op rotary</b>: when the <c>do_rotary</c> attribute is 1, RoPE is applied to
/// Q and K (not V) before attention, using <c>cos_cache</c>/<c>sin_cache</c> indexed by the
/// absolute position <c>pastSeq + i</c> of each token. <c>rotary_interleaved</c> selects the
/// pairing convention (0 = NeoX/half-split, 1 = GPT-J/interleaved); the math mirrors
/// <see cref="RotaryEmbeddingKernel"/> exactly. When <c>do_rotary</c> is 0 (or unset) Q/K are
/// assumed pre-rotated and the cos/sin caches are ignored.</para>
///
/// <para><b>Outputs</b>: <c>output</c> [B, Sq, num_heads·head_dim],
/// <c>present_key</c>/<c>present_value</c> [B, kv_num_heads, totalSeq, head_dim]
/// (past concatenated with the new K/V along the sequence axis).</para>
///
/// <para><b>seqlens_k / total_sequence_length</b>: tolerated when present. The causal key
/// bound per (batch, query) is min(pastSeq + qi, seqlens_k[b]); for the common batch-1
/// prefill/decode case this matches the existing pastSeq+qi logic. <c>local_window_size</c>
/// (sliding-window attention), when ≥ 0, additionally bounds the key range <b>below</b>: a query
/// at absolute position <c>qAbs = pastSeq + qi</c> attends only to keys in the inclusive window
/// <c>[max(0, qAbs − local_window_size + 1), qAbs]</c> (the current key plus the
/// <c>local_window_size − 1</c> preceding ones). −1 (default) disables the lower bound (full
/// causal). Models: Mistral-7B v0.1/0.2, Phi-3-small.</para>
/// </summary>
public sealed class GroupQueryAttentionKernel : IKernel
{
    public string OpType => "GroupQueryAttention";

    /// <summary>
    /// Below this many fused multiply-adds (units × keys × head_dim) the attention
    /// loop runs serially: thread-pool dispatch costs more than the work itself for
    /// tiny prefill/decode steps. ~256K MACs ≈ a handful of small heads.
    /// </summary>
    private const long ParallelThreshold = 1L << 18;

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
                $"GroupQueryAttention supports only 3-D query [B, S, hidden]; got rank {query.Shape.Rank}.");

        string keyName = node.Inputs.Count > 1 ? node.Inputs[1] : "";
        string valueName = node.Inputs.Count > 2 ? node.Inputs[2] : "";
        bool packed = keyName.Length == 0 || valueName.Length == 0;

        int batch = query.Shape[0];
        int qSeq = query.Shape[1];

        // Materialize per-token Q/K/V buffers laid out as [B, S, heads·head_dim], for either
        // the packed (split the single 'query' tensor) or unpacked (separate K/V) layout.
        // These buffers are mutable so in-op rotary can be applied to Q and K below.
        int headDim;
        int kvSeq;
        float[] qBuf, kBuf, vBuf;

        if (packed)
        {
            // query = [B, S, (num_heads + 2·kv_num_heads)·head_dim]. head_dim is derived.
            int hidden = query.Shape[2];
            int unit = numHeads + 2 * kvNumHeads;
            if (hidden % unit != 0)
                throw new ModelSharpException(
                    $"GroupQueryAttention packed-QKV hidden {hidden} not divisible by "
                    + $"(num_heads + 2·kv_num_heads) = {unit}.");
            headDim = hidden / unit;
            kvSeq = qSeq;                 // packed K/V share the query sequence length.
            int qHidden = numHeads * headDim;
            int kvHidden = kvNumHeads * headDim;

            qBuf = new float[batch * qSeq * qHidden];
            kBuf = new float[batch * kvSeq * kvHidden];
            vBuf = new float[batch * kvSeq * kvHidden];
            ReadOnlySpan<float> src = query.Span;
            for (int b = 0; b < batch; b++)
            for (int s = 0; s < qSeq; s++)
            {
                int row = (b * qSeq + s) * hidden;
                int qDst = (b * qSeq + s) * qHidden;
                int kvDst = (b * kvSeq + s) * kvHidden;
                for (int i = 0; i < qHidden; i++) qBuf[qDst + i] = src[row + i];
                for (int i = 0; i < kvHidden; i++) kBuf[kvDst + i] = src[row + qHidden + i];
                for (int i = 0; i < kvHidden; i++) vBuf[kvDst + i] = src[row + qHidden + kvHidden + i];
            }
        }
        else
        {
            Tensor<float> key = ctx.Get(keyName);
            Tensor<float> value = ctx.Get(valueName);
            if (key.Shape.Rank != 3 || value.Shape.Rank != 3)
                throw new ModelSharpException("GroupQueryAttention supports only unpacked 3-D key/value.");

            int qHidden = query.Shape[2];
            kvSeq = key.Shape[1];
            int kvHidden = key.Shape[2];

            if (key.Shape[0] != batch || value.Shape[0] != batch)
                throw new ModelSharpException("GroupQueryAttention query/key/value batch sizes disagree.");
            if (value.Shape[1] != kvSeq || value.Shape[2] != kvHidden)
                throw new ModelSharpException("GroupQueryAttention key/value shapes disagree.");
            if (qHidden % numHeads != 0)
                throw new ModelSharpException(
                    $"GroupQueryAttention query hidden {qHidden} not divisible by num_heads={numHeads}.");
            headDim = qHidden / numHeads;
            if (kvHidden != kvNumHeads * headDim)
                throw new ModelSharpException(
                    $"GroupQueryAttention key/value hidden {kvHidden} must equal kv_num_heads·head_dim = {kvNumHeads * headDim}.");

            qBuf = query.Span.ToArray();
            kBuf = key.Span.ToArray();
            vBuf = value.Span.ToArray();
        }

        int qHid = numHeads * headDim;
        int kvHid = kvNumHeads * headDim;

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

        // Optional seqlens_k (int32 [B]): the 0-based index of the last valid key per batch.
        // Used only to bound the causal key range; absent → fall back to pastSeq+kvSeq.
        long[]? seqlensK = node.Inputs.Count > 5 && node.Inputs[5].Length > 0
            ? TensorInts.Read(ctx.GetTensor(node.Inputs[5]))
            : null;
        // total_sequence_length (int32 scalar) is read for tolerance/diagnostics; the present
        // buffers already span pastSeq+kvSeq so it carries no extra layout information here.
        // (Read but otherwise unused — present so genai graphs don't choke on the wire.)
        _ = node.Inputs.Count > 6 && node.Inputs[6].Length > 0 ? ctx.GetTensor(node.Inputs[6]) : null;

        // ---- In-op rotary embedding (do_rotary == 1) ----------------------------------------
        bool doRotary = Attr.Int(node, "do_rotary", 0) != 0;
        if (doRotary)
        {
            if (node.Inputs.Count <= 8 || node.Inputs[7].Length == 0 || node.Inputs[8].Length == 0)
                throw new ModelSharpException(
                    "GroupQueryAttention do_rotary=1 requires cos_cache (input 7) and sin_cache (input 8).");
            Tensor<float> cosCache = ctx.Get(node.Inputs[7]);
            Tensor<float> sinCache = ctx.Get(node.Inputs[8]);
            bool interleaved = Attr.Int(node, "rotary_interleaved", 0) != 0;
            ApplyRotary(qBuf, batch, qSeq, numHeads, headDim, pastSeq, cosCache, sinCache, interleaved);
            ApplyRotary(kBuf, batch, kvSeq, kvNumHeads, headDim, pastSeq, cosCache, sinCache, interleaved);
        }

        float scaleAttr = Attr.Float(node, "scale", 0f);
        float scale = scaleAttr != 0f ? scaleAttr : 1f / MathF.Sqrt(headDim);

        // local_window_size (sliding-window attention): when ≥ 0, query at absolute
        // position qAbs = pastSeq + qi may attend only to key positions in the inclusive
        // causal window [qAbs - local_window_size + 1, qAbs] — i.e. the current key plus the
        // (local_window_size − 1) immediately preceding keys. Older keys are excluded from the
        // softmax. −1 (default) disables the lower bound → full causal (range starts at 0).
        int localWindow = (int)Attr.Int(node, "local_window_size", -1);

        ReadOnlySpan<float> kSpan = kBuf;
        ReadOnlySpan<float> vSpan = vBuf;

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
            int src = (b * kvSeq + s) * kvHid + g * headDim;
            int dst = ((b * kvNumHeads + g) * totalSeq + pastSeq + s) * headDim;
            for (int d = 0; d < headDim; d++) { pk[dst + d] = kSpan[src + d]; pv[dst + d] = vSpan[src + d]; }
        }

        var outT = new Tensor<float>(new TensorShape(batch, qSeq, qHid));

        // Parallelize over the independent (batch × query_head × query_position) outer
        // index: each iteration computes one output head-vector and writes a disjoint
        // [headDim] slice of the output, so no locks are needed. The Q·K dot and the
        // attn·V accumulation are SIMD-vectorized. KV is read-only and shared.
        float[] qArr = qBuf;          // [B, Sq, qHid]
        float[] pkArr = KernelSimd.Array(presentK); // [B, kvNumHeads, totalSeq, headDim]
        float[] pvArr = KernelSimd.Array(presentV);
        float[] outArr = KernelSimd.Array(outT);    // [B, Sq, qHid]
        int totalUnits = batch * numHeads * qSeq;

        // Tiny problems stay serial to dodge thread-pool dispatch overhead.
        bool parallel = (long)totalUnits * totalSeq * headDim >= ParallelThreshold;

        void Compute(int unit, float[] scores)
        {
            int qi = unit % qSeq;
            int hb = unit / qSeq;
            int h = hb % numHeads;
            int b = hb / numHeads;

            int seqBound = seqlensK is not null && seqlensK.Length > 0
                ? (int)seqlensK[Math.Min(b, seqlensK.Length - 1)]
                : totalSeq - 1;

            int g = h / groupSize;   // shared KV head for this query head.
            int qRow = (b * qSeq + qi) * qHid + h * headDim;
            // Causal: query position qi corresponds to absolute position pastSeq+qi,
            // so it may attend to key positions 0..pastSeq+qi inclusive, further
            // capped by the last valid key (seqlens_k) for this batch.
            int qAbs = pastSeq + qi;
            int kLimit = qAbs;
            if (kLimit > seqBound) kLimit = seqBound;
            // Sliding window: drop keys older than the window's lower bound. With localWindow
            // disabled (−1) kStart stays 0 → full causal range, identical to before.
            int kStart = localWindow >= 0 ? Math.Max(0, qAbs - localWindow + 1) : 0;

            int kvBase = (b * kvNumHeads + g) * totalSeq * headDim;

            float mx = float.NegativeInfinity;
            for (int kj = kStart; kj <= kLimit; kj++)
            {
                float sc = KernelSimd.Dot(qArr, qRow, pkArr, kvBase + kj * headDim, headDim) * scale;
                scores[kj] = sc;
                if (sc > mx) mx = sc;
            }

            float sum = 0f;
            for (int kj = kStart; kj <= kLimit; kj++) { float e = MathF.Exp(scores[kj] - mx); scores[kj] = e; sum += e; }
            float inv = sum > 0f ? 1f / sum : 0f;

            int outRow = (b * qSeq + qi) * qHid + h * headDim;
            System.Array.Clear(outArr, outRow, headDim);
            for (int kj = kStart; kj <= kLimit; kj++)
                KernelSimd.AxpyInto(outArr, outRow, pvArr, kvBase + kj * headDim, scores[kj] * inv, headDim);
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
    /// Applies RoPE in place to a [B, S, heads·head_dim] buffer, mirroring
    /// <see cref="RotaryEmbeddingKernel"/>. cos/sin caches are [max_seq, rotary_dim/2];
    /// the token at sequence index <c>s</c> uses cache row <c>pastSeq + s</c> (its absolute
    /// position). <paramref name="interleaved"/> selects the pairing:
    /// false = half-split (pair d with d+rotHalf), true = interleaved (pair 2j with 2j+1).
    /// Channels beyond the rotary span pass through unchanged.
    /// </summary>
    private static void ApplyRotary(
        float[] buf, int batch, int seq, int heads, int headDim, int pastSeq,
        Tensor<float> cosCache, Tensor<float> sinCache, bool interleaved)
    {
        int half = cosCache.Shape.Dimensions[cosCache.Shape.Rank - 1];
        int rotary = 2 * half;
        if (rotary > headDim) rotary = headDim;
        int rotHalf = rotary / 2;
        ReadOnlySpan<float> cos = cosCache.Span;
        ReadOnlySpan<float> sin = sinCache.Span;
        int maxPos = cosCache.Shape[0];

        for (int b = 0; b < batch; b++)
        for (int s = 0; s < seq; s++)
        {
            int pos = pastSeq + s;
            if (pos >= maxPos) pos = maxPos - 1;
            int cacheBase = pos * half;
            for (int h = 0; h < heads; h++)
            {
                int rowBase = ((b * seq + s) * heads + h) * headDim;
                for (int j = 0; j < rotHalf; j++)
                {
                    float c = cos[cacheBase + j];
                    float sn = sin[cacheBase + j];
                    int i0, i1;
                    if (interleaved) { i0 = rowBase + 2 * j; i1 = i0 + 1; }
                    else { i0 = rowBase + j; i1 = rowBase + j + rotHalf; }
                    float a = buf[i0];
                    float bb = buf[i1];
                    buf[i0] = a * c - bb * sn;
                    buf[i1] = bb * c + a * sn;
                }
            }
        }
    }
}
