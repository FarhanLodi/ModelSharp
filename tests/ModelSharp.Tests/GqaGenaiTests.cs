using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Llm;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the onnxruntime-genai variant of
/// <see cref="GroupQueryAttentionKernel"/>: packed-QKV input, in-op rotary embedding
/// (cos/sin caches), and the seqlens_k / total_sequence_length inputs used by real
/// INT4 LLM exports (Qwen2.5, Mistral-7B, Phi-3). Each test builds the node + context
/// by hand and checks against a reference computed inline.
/// </summary>
public class GqaGenaiTests
{
    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<int> I(int[] dims, params int[] data) =>
        Tensor<int>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    // Reference causal SDPA, single batch + single head: Q [Sq,hd], K/V [Sk,hd].
    // pastOffset is the absolute position of query row 0 (so query i sees keys 0..pastOffset+i).
    private static float[] RefCausal(float[,] q, float[,] k, float[,] v, float scale, int pastOffset)
    {
        int sq = q.GetLength(0), hd = q.GetLength(1), sk = k.GetLength(0), vhd = v.GetLength(1);
        var outp = new float[sq * vhd];
        for (int i = 0; i < sq; i++)
        {
            int limit = Math.Min(pastOffset + i, sk - 1);
            var scores = new float[sk];
            float mx = float.NegativeInfinity;
            for (int j = 0; j <= limit; j++)
            {
                float dot = 0f;
                for (int d = 0; d < hd; d++) dot += q[i, d] * k[j, d];
                scores[j] = dot * scale;
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0f;
            for (int j = 0; j <= limit; j++) { scores[j] = MathF.Exp(scores[j] - mx); sum += scores[j]; }
            for (int d = 0; d < vhd; d++)
            {
                float acc = 0f;
                for (int j = 0; j <= limit; j++) acc += (scores[j] / sum) * v[j, d];
                outp[i * vhd + d] = acc;
            }
        }
        return outp;
    }

    // Half-split (NeoX) rotary on one [hd] vector for cos/sin half-rows c[],s[] (length hd/2).
    private static float[] RotHalf(float[] x, float[] c, float[] s)
    {
        int hd = x.Length, half = hd / 2;
        var y = (float[])x.Clone();
        for (int j = 0; j < half; j++)
        {
            float a = x[j], b = x[j + half];
            y[j] = a * c[j] - b * s[j];
            y[j + half] = b * c[j] + a * s[j];
        }
        return y;
    }

    // Interleaved (GPT-J) rotary on one [hd] vector.
    private static float[] RotInterleaved(float[] x, float[] c, float[] s)
    {
        int hd = x.Length, half = hd / 2;
        var y = (float[])x.Clone();
        for (int j = 0; j < half; j++)
        {
            float a = x[2 * j], b = x[2 * j + 1];
            y[2 * j] = a * c[j] - b * s[j];
            y[2 * j + 1] = b * c[j] + a * s[j];
        }
        return y;
    }

    // ----------------------------------------------------------------------------------------
    // 1. Packed-QKV, no rotary: split + grouped causal attention. Must equal both a manual
    //    reference AND the equivalent UNPACKED call.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void PackedQkv_NoRotary_MatchesUnpackedAndReference()
    {
        // B=1, S=2, num_heads=2, kv_num_heads=1, head_dim=2.
        // packed hidden = (2 + 2*1)*2 = 8 per token: [Q(4) | K(2) | V(2)].
        int s = 2, headDim = 2, numHeads = 2, kvHeads = 1;
        // Token0 Q=[1,0, 0,1] K=[1,0] V=[2,3];  Token1 Q=[0,1, 1,0] K=[0,1] V=[5,7].
        float[] packed =
        {
            1, 0, 0, 1,  1, 0,  2, 3,    // token0
            0, 1, 1, 0,  0, 1,  5, 7,    // token1
        };
        var ctxP = Ctx(("qkv", F(new[] { 1, s, 8 }, packed)));
        // key/value names empty => packed path.
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention", new[] { "qkv", "", "" }, new[] { "out", "pk", "pv" },
                new Dictionary<string, object> { ["num_heads"] = 2L, ["kv_num_heads"] = 1L }), ctxP);
        float[] outP = ctxP.Get("out").Span.ToArray();

        // Equivalent unpacked call: split packed into separate q/k/v tensors.
        float[] q = { 1, 0, 0, 1,  0, 1, 1, 0 };   // [B,S,num_heads*hd]
        float[] k = { 1, 0,  0, 1 };               // [B,S,kv_heads*hd]
        float[] v = { 2, 3,  5, 7 };
        var ctxU = Ctx(("q", F(new[] { 1, s, 4 }, q)),
            ("k", F(new[] { 1, s, 2 }, k)), ("v", F(new[] { 1, s, 2 }, v)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention", new[] { "q", "k", "v" }, new[] { "out", "pk", "pv" },
                new Dictionary<string, object> { ["num_heads"] = 2L, ["kv_num_heads"] = 1L }), ctxU);
        float[] outU = ctxU.Get("out").Span.ToArray();

        // Packed and unpacked must agree exactly.
        Close(outU, outP, 1e-6f);

        // Manual reference: both query heads share the single kv head.
        float scale = 1f / MathF.Sqrt(headDim);
        var kh = new float[,] { { 1, 0 }, { 0, 1 } };
        var vh = new float[,] { { 2, 3 }, { 5, 7 } };
        for (int h = 0; h < numHeads; h++)
        {
            var qh = new float[s, headDim];
            for (int t = 0; t < s; t++)
            for (int d = 0; d < headDim; d++)
                qh[t, d] = q[(t * numHeads + h) * headDim + d];
            float[] r = RefCausal(qh, kh, vh, scale, 0);
            for (int t = 0; t < s; t++)
            for (int d = 0; d < headDim; d++)
            {
                float got = outP[(t * numHeads + h) * headDim + d];
                Assert.True(MathF.Abs(r[t * headDim + d] - got) <= 1e-4f,
                    $"head {h} pos {t} dim {d}: expected {r[t * headDim + d]}, got {got}");
            }
        }

        // present_key/value shape [B, kv_num_heads, S, head_dim].
        Assert.Equal(new[] { 1, kvHeads, s, headDim }, ctxP.GetTensor("pk").Shape.Dimensions.ToArray());
    }

    // ----------------------------------------------------------------------------------------
    // 2. Packed-QKV + rotary (do_rotary=1, half-split). Rotary applied to Q and K (not V),
    //    then causal attention. Compared to a manual rotary+attention reference.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void PackedQkv_Rotary_HalfSplit_MatchesManual()
    {
        int s = 2, headDim = 4, numHeads = 1, kvHeads = 1;
        // packed hidden = (1 + 2)*4 = 12: [Q(4) | K(4) | V(4)].
        float[] q0 = { 1, 2, 3, 4 }, k0 = { 1, 0, 1, 0 }, v0 = { 1, 1, 1, 1 };
        float[] q1 = { 0, 1, 0, 1 }, k1 = { 2, 1, 0, 1 }, v1 = { 2, 2, 2, 2 };
        float[] packed =
        {
            q0[0], q0[1], q0[2], q0[3],  k0[0], k0[1], k0[2], k0[3],  v0[0], v0[1], v0[2], v0[3],
            q1[0], q1[1], q1[2], q1[3],  k1[0], k1[1], k1[2], k1[3],  v1[0], v1[1], v1[2], v1[3],
        };
        // cos/sin cache: [max_seq=2, rotary_dim/2=2]. Distinct rows per position.
        // pos0: cos=[1,1] sin=[0,0] (identity). pos1: cos=[0.6,0.8] sin=[0.8,0.6].
        float[] cos = { 1f, 1f,  0.6f, 0.8f };
        float[] sin = { 0f, 0f,  0.8f, 0.6f };

        var ctx = Ctx(("qkv", F(new[] { 1, s, 12 }, packed)),
            ("cos", F(new[] { 2, 2 }, cos)), ("sin", F(new[] { 2, 2 }, sin)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention",
                new[] { "qkv", "", "", "", "", "", "", "cos", "sin" },
                new[] { "out" },
                new Dictionary<string, object>
                {
                    ["num_heads"] = 1L, ["kv_num_heads"] = 1L, ["do_rotary"] = 1L,
                }), ctx);
        float[] got = ctx.Get("out").Span.ToArray();

        // Manual: rotate Q/K per position (half-split), V untouched, then causal SDPA.
        float[] c0 = { 1f, 1f }, sn0 = { 0f, 0f };
        float[] c1 = { 0.6f, 0.8f }, sn1 = { 0.8f, 0.6f };
        float[] rq0 = RotHalf(q0, c0, sn0), rq1 = RotHalf(q1, c1, sn1);
        float[] rk0 = RotHalf(k0, c0, sn0), rk1 = RotHalf(k1, c1, sn1);

        var qh = new[,] { { rq0[0], rq0[1], rq0[2], rq0[3] }, { rq1[0], rq1[1], rq1[2], rq1[3] } };
        var kh = new[,] { { rk0[0], rk0[1], rk0[2], rk0[3] }, { rk1[0], rk1[1], rk1[2], rk1[3] } };
        var vh = new[,] { { v0[0], v0[1], v0[2], v0[3] }, { v1[0], v1[1], v1[2], v1[3] } };
        float scale = 1f / MathF.Sqrt(headDim);
        float[] expected = RefCausal(qh, kh, vh, scale, 0);

        Close(expected, got);
    }

    // ----------------------------------------------------------------------------------------
    // 3. rotary_interleaved=1 vs 0 produce different results (and each matches its own
    //    reference rotary convention).
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void Rotary_Interleaved_vs_HalfSplit_Differ()
    {
        // S>=2 is needed to expose rotary on Q/K (S=1 has no attention mixing); use S=2 with a non-identity cache row1.
        int s = 2;
        float[] q0 = { 1, 2, 3, 4 }, k0 = { 1, 1, 1, 1 }, v0 = { 1, 0, 0, 0 };
        float[] q1 = { 4, 3, 2, 1 }, k1 = { 1, 2, 3, 4 }, v1 = { 0, 1, 0, 0 };
        float[] packed =
        {
            q0[0], q0[1], q0[2], q0[3],  k0[0], k0[1], k0[2], k0[3],  v0[0], v0[1], v0[2], v0[3],
            q1[0], q1[1], q1[2], q1[3],  k1[0], k1[1], k1[2], k1[3],  v1[0], v1[1], v1[2], v1[3],
        };
        float[] cos = { 1f, 1f,  0.6f, 0.8f };
        float[] sin = { 0f, 0f,  0.8f, 0.6f };

        float[] Run(long interleaved)
        {
            var ctx = Ctx(("qkv", F(new[] { 1, s, 12 }, packed)),
                ("cos", F(new[] { 2, 2 }, cos)), ("sin", F(new[] { 2, 2 }, sin)));
            new GroupQueryAttentionKernel().Execute(
                Node("GroupQueryAttention",
                    new[] { "qkv", "", "", "", "", "", "", "cos", "sin" },
                    new[] { "out" },
                    new Dictionary<string, object>
                    {
                        ["num_heads"] = 1L, ["kv_num_heads"] = 1L, ["do_rotary"] = 1L,
                        ["rotary_interleaved"] = interleaved,
                    }), ctx);
            return ctx.Get("out").Span.ToArray();
        }

        float[] half = Run(0);
        float[] inter = Run(1);

        // The two conventions must produce different outputs on row 1 (where cache is non-identity).
        bool differ = false;
        for (int i = 0; i < half.Length; i++)
            if (MathF.Abs(half[i] - inter[i]) > 1e-4f) differ = true;
        Assert.True(differ, "interleaved and half-split rotary produced identical output");

        // And each matches its own reference convention.
        float[] c0 = { 1f, 1f }, sn0 = { 0f, 0f };
        float[] c1 = { 0.6f, 0.8f }, sn1 = { 0.8f, 0.6f };
        float scale = 1f / MathF.Sqrt(4f);

        float[] RefWith(Func<float[], float[], float[], float[]> rot)
        {
            float[] rq0 = rot(q0, c0, sn0), rq1 = rot(q1, c1, sn1);
            float[] rk0 = rot(k0, c0, sn0), rk1 = rot(k1, c1, sn1);
            var qh = new[,] { { rq0[0], rq0[1], rq0[2], rq0[3] }, { rq1[0], rq1[1], rq1[2], rq1[3] } };
            var kh = new[,] { { rk0[0], rk0[1], rk0[2], rk0[3] }, { rk1[0], rk1[1], rk1[2], rk1[3] } };
            var vh = new[,] { { v0[0], v0[1], v0[2], v0[3] }, { v1[0], v1[1], v1[2], v1[3] } };
            return RefCausal(qh, kh, vh, scale, 0);
        }

        Close(RefWith(RotHalf), half);
        Close(RefWith(RotInterleaved), inter);
    }

    // ----------------------------------------------------------------------------------------
    // 4. Decode step: non-empty past_key/past_value + seqlens_k present. Unpacked input,
    //    single new query token attending over past + current key.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void Decode_WithPast_AndSeqlensK()
    {
        // num_heads=2, kv_num_heads=1, head_dim=2. past has 2 keys, 1 new token => total 3.
        int headDim = 2, numHeads = 2, kvHeads = 1;
        // present query (S=1): two heads.
        float[] q = { 1, 0,  0, 1 };          // [B,1,num_heads*hd]
        float[] kNew = { 1, 1 };              // [B,1,kv*hd]
        float[] vNew = { 9, 9 };
        // past_key/value: [B, kv_heads, pastSeq=2, hd].
        float[] pk = { 1, 0,  0, 1 };         // key0=[1,0] key1=[0,1]
        float[] pv = { 2, 0,  0, 4 };         // val0=[2,0] val1=[0,4]
        // seqlens_k: last valid key index = total-1 = 2.
        var ctx = Ctx(
            ("q", F(new[] { 1, 1, 4 }, q)),
            ("k", F(new[] { 1, 1, 2 }, kNew)),
            ("v", F(new[] { 1, 1, 2 }, vNew)),
            ("pk", F(new[] { 1, 1, 2, 2 }, pk)),
            ("pv", F(new[] { 1, 1, 2, 2 }, pv)),
            ("seqlens", I(new[] { 1 }, 2)),
            ("totlen", I(new[] { 1 }, 3)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention",
                new[] { "q", "k", "v", "pk", "pv", "seqlens", "totlen" },
                new[] { "out", "presk", "presv" },
                new Dictionary<string, object> { ["num_heads"] = 2L, ["kv_num_heads"] = 1L }), ctx);
        float[] got = ctx.Get("out").Span.ToArray();

        // Reference: combined keys = [past0, past1, new] = [[1,0],[0,1],[1,1]];
        // values = [[2,0],[0,4],[9,9]]. Query pos absolute = pastSeq(2)+0 = 2 => sees all 3.
        var kAll = new[,] { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } };
        var vAll = new[,] { { 2f, 0f }, { 0f, 4f }, { 9f, 9f } };
        float scale = 1f / MathF.Sqrt(headDim);
        for (int h = 0; h < numHeads; h++)
        {
            var qh = new float[1, headDim];
            for (int d = 0; d < headDim; d++) qh[0, d] = q[h * headDim + d];
            float[] r = RefCausal(qh, kAll, vAll, scale, 2);
            for (int d = 0; d < headDim; d++)
            {
                float gv = got[h * headDim + d];
                Assert.True(MathF.Abs(r[d] - gv) <= 1e-4f,
                    $"head {h} dim {d}: expected {r[d]}, got {gv}");
            }
        }

        // present_key/value [B, kv_heads, total=3, hd].
        Assert.Equal(new[] { 1, kvHeads, 3, headDim }, ctx.GetTensor("presk").Shape.Dimensions.ToArray());
        // present_key row 2 (new key) == kNew.
        float[] presk = ctx.Get("presk").Span.ToArray();
        Close(new[] { 1f, 1f }, new[] { presk[2 * headDim], presk[2 * headDim + 1] });
    }

    // ----------------------------------------------------------------------------------------
    // 5. seqlens_k that bounds the key range below total: a shorter valid length must mask
    //    out the tail key.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void SeqlensK_BoundsKeyRange()
    {
        // num_heads=1, kv_num_heads=1, head_dim=1, S=2, no past. seqlens_k=0 => only key0 valid.
        var ctx = Ctx(
            ("q", F(new[] { 1, 2, 1 }, 1f, 1f)),
            ("k", F(new[] { 1, 2, 1 }, 0.5f, 0.5f)),
            ("v", F(new[] { 1, 2, 1 }, 3f, 9f)),
            ("seqlens", I(new[] { 1 }, 0)),     // last valid key index = 0
            ("totlen", I(new[] { 1 }, 2)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention",
                new[] { "q", "k", "v", "", "", "seqlens", "totlen" },
                new[] { "out" },
                new Dictionary<string, object> { ["num_heads"] = 1L, ["kv_num_heads"] = 1L }), ctx);
        float[] o = ctx.Get("out").Span.ToArray();
        // Both query positions are bounded to key0 only => out == V[0] = 3 everywhere.
        Close(new[] { 3f, 3f }, o);
    }

    // ----------------------------------------------------------------------------------------
    // Regression: the existing UNPACKED, no-rotary path is unchanged (mirrors the existing
    // AttentionContribTests causal-mask case).
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void Unpacked_NoRotary_Unchanged()
    {
        var ctx = Ctx(
            ("q", F(new[] { 1, 2, 2 }, 1, 1, 1, 1)),
            ("k", F(new[] { 1, 2, 1 }, 0.5f, 0.5f)),
            ("v", F(new[] { 1, 2, 1 }, 3f, 9f)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention", new[] { "q", "k", "v" }, new[] { "out" },
                new Dictionary<string, object> { ["num_heads"] = 2L, ["kv_num_heads"] = 1L }), ctx);
        float[] o = ctx.Get("out").Span.ToArray();
        // pos0 sees key0 only -> V[0]=3 for both heads; pos1 uniform -> 6.
        Close(new[] { 3f, 3f, 6f, 6f }, o);
    }
}
