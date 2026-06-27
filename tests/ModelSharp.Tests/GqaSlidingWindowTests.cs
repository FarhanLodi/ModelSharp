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
/// Sliding-window attention (<c>local_window_size</c>) tests for
/// <see cref="GroupQueryAttentionKernel"/>. ORT semantics: when <c>local_window_size ≥ 0</c>,
/// a query at absolute position <c>qAbs = pastSeq + qi</c> attends only to key positions in the
/// inclusive causal window <c>[max(0, qAbs − local_window_size + 1), qAbs]</c> — the current key
/// plus the <c>local_window_size − 1</c> immediately preceding keys; older keys are excluded from
/// the softmax. <c>local_window_size = −1</c> (default) disables the window → full causal.
/// Each test builds the node + context by hand (mirroring <see cref="GqaGenaiTests"/>) and checks
/// against an inline windowed-softmax reference.
/// </summary>
public class GqaSlidingWindowTests
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

    // Windowed causal SDPA, single batch + single head: Q [Sq,hd], K/V [Sk,hd].
    // pastOffset is the absolute position of query row 0. localWindow < 0 => full causal.
    // Query row i (absolute qAbs = pastOffset + i) attends to keys
    // [max(0, qAbs - localWindow + 1), min(qAbs, Sk-1)] inclusive.
    private static float[] RefWindow(float[,] q, float[,] k, float[,] v, float scale,
        int pastOffset, int localWindow)
    {
        int sq = q.GetLength(0), hd = q.GetLength(1), sk = k.GetLength(0), vhd = v.GetLength(1);
        var outp = new float[sq * vhd];
        for (int i = 0; i < sq; i++)
        {
            int qAbs = pastOffset + i;
            int limit = Math.Min(qAbs, sk - 1);
            int start = localWindow >= 0 ? Math.Max(0, qAbs - localWindow + 1) : 0;
            var scores = new float[sk];
            float mx = float.NegativeInfinity;
            for (int j = start; j <= limit; j++)
            {
                float dot = 0f;
                for (int d = 0; d < hd; d++) dot += q[i, d] * k[j, d];
                scores[j] = dot * scale;
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0f;
            for (int j = start; j <= limit; j++) { scores[j] = MathF.Exp(scores[j] - mx); sum += scores[j]; }
            for (int d = 0; d < vhd; d++)
            {
                float acc = 0f;
                for (int j = start; j <= limit; j++) acc += (scores[j] / sum) * v[j, d];
                outp[i * vhd + d] = acc;
            }
        }
        return outp;
    }

    private static float[] RunUnpacked(float[] q, float[] k, float[] v, int s, int numHeads,
        int kvHeads, int headDim, long localWindow)
    {
        var attrs = new Dictionary<string, object>
        {
            ["num_heads"] = (long)numHeads,
            ["kv_num_heads"] = (long)kvHeads,
            ["local_window_size"] = localWindow,
        };
        var ctx = Ctx(
            ("q", F(new[] { 1, s, numHeads * headDim }, q)),
            ("k", F(new[] { 1, s, kvHeads * headDim }, k)),
            ("v", F(new[] { 1, s, kvHeads * headDim }, v)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention", new[] { "q", "k", "v" }, new[] { "out" }, attrs), ctx);
        return ctx.Get("out").Span.ToArray();
    }

    // ----------------------------------------------------------------------------------------
    // 1. local_window_size = 2: each query attends to exactly its own + 1 previous key.
    //    Single head, head_dim=1, S=5, empty past. Compared to a windowed-softmax reference.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void Window2_SingleHead_AttendsSelfPlusOnePrevious()
    {
        int s = 5, headDim = 1, numHeads = 1, kvHeads = 1;
        // Distinct values so each masked/unmasked combination yields a distinct output.
        float[] q = { 0.3f, 0.7f, 1.1f, 0.2f, 0.9f };
        float[] k = { 0.5f, 0.2f, 0.8f, 0.4f, 0.6f };
        float[] v = { 1f, 2f, 3f, 4f, 5f };

        float[] got = RunUnpacked(q, k, v, s, numHeads, kvHeads, headDim, localWindow: 2);

        var qh = new float[s, headDim];
        var kh = new float[s, headDim];
        var vh = new float[s, headDim];
        for (int t = 0; t < s; t++) { qh[t, 0] = q[t]; kh[t, 0] = k[t]; vh[t, 0] = v[t]; }
        float scale = 1f / MathF.Sqrt(headDim);
        float[] expected = RefWindow(qh, kh, vh, scale, pastOffset: 0, localWindow: 2);

        Close(expected, got);

        // Sanity on the window structure itself:
        //  - query 0 (qAbs=0) attends to key {0} only => out[0] == v[0] = 1.
        Assert.True(MathF.Abs(got[0] - 1f) <= 1e-4f, $"q0 should equal v0; got {got[0]}");
        //  - query 1 (qAbs=1) attends to keys {0,1}, NOT later; output is a convex combo of v0,v1
        //    so it must lie strictly inside (1,2).
        Assert.True(got[1] > 1f && got[1] < 2f, $"q1 expected in (1,2); got {got[1]}");
        //  - query 2 (qAbs=2, window=2) attends to keys {1,2} only — v0 is EXCLUDED — so the
        //    output is a convex combo of v1=2 and v2=3, hence > 2.
        Assert.True(got[2] > 2f && got[2] < 3f, $"q2 (window excludes v0) expected in (2,3); got {got[2]}");
    }

    // ----------------------------------------------------------------------------------------
    // 2. local_window_size = -1 (disabled) and a large window both reproduce full causal,
    //    i.e. equal the existing no-window result.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void WindowDisabledOrLarge_EqualsFullCausal()
    {
        int s = 4, headDim = 2, numHeads = 2, kvHeads = 1;
        float[] q = { 1, 0, 0, 1,  0.5f, 0.5f, 1, 1,  0.2f, 0.9f, 0.7f, 0.3f,  1, 0.4f, 0.6f, 0.8f };
        float[] k = { 1, 0,  0, 1,  0.5f, 0.5f,  0.3f, 0.9f };
        float[] v = { 2, 3,  5, 7,  1, 1,  4, 8 };

        float[] noWindow = RunUnpacked(q, k, v, s, numHeads, kvHeads, headDim, localWindow: -1);
        float[] bigWindow = RunUnpacked(q, k, v, s, numHeads, kvHeads, headDim, localWindow: 100);

        // A window ≥ S can never exclude any causal key, so it must equal the disabled case.
        Close(noWindow, bigWindow, 1e-6f);

        // And both equal the manual full-causal reference (per head; the single kv head is shared).
        float scale = 1f / MathF.Sqrt(headDim);
        var kh = new float[s, headDim];
        var vh = new float[s, headDim];
        for (int t = 0; t < s; t++)
        for (int d = 0; d < headDim; d++) { kh[t, d] = k[t * headDim + d]; vh[t, d] = v[t * headDim + d]; }
        for (int h = 0; h < numHeads; h++)
        {
            var qh = new float[s, headDim];
            for (int t = 0; t < s; t++)
            for (int d = 0; d < headDim; d++) qh[t, d] = q[(t * numHeads + h) * headDim + d];
            float[] r = RefWindow(qh, kh, vh, scale, pastOffset: 0, localWindow: -1);
            for (int t = 0; t < s; t++)
            for (int d = 0; d < headDim; d++)
            {
                float gv = noWindow[(t * numHeads + h) * headDim + d];
                Assert.True(MathF.Abs(r[t * headDim + d] - gv) <= 1e-4f,
                    $"head {h} pos {t} dim {d}: expected {r[t * headDim + d]}, got {gv}");
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // 3. Window of exactly 1 = attend to self only => output equals the matching V row.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void Window1_AttendsSelfOnly()
    {
        int s = 4, headDim = 1, numHeads = 1, kvHeads = 1;
        float[] q = { 0.3f, 0.7f, 1.1f, 0.2f };
        float[] k = { 0.5f, 0.2f, 0.8f, 0.4f };
        float[] v = { 1f, 2f, 3f, 4f };

        float[] got = RunUnpacked(q, k, v, s, numHeads, kvHeads, headDim, localWindow: 1);

        // Each query attends only to its own key => softmax over a single element => out == v[i].
        Close(v, got);
    }

    // ----------------------------------------------------------------------------------------
    // 4. Decode-with-past: the window excludes old cached keys. past has 3 keys, 1 new token
    //    (qAbs=3). local_window_size=2 => attend only to keys {2,3}; cached keys 0,1 excluded.
    // ----------------------------------------------------------------------------------------
    [Fact]
    public void DecodeWithPast_WindowExcludesOldCachedKeys()
    {
        int headDim = 2;
        // present query (S=1).
        float[] q = { 1, 0 };
        float[] kNew = { 1, 1 };
        float[] vNew = { 9, 9 };
        // past_key/value: [B, kv_heads, pastSeq=3, hd]. Keys 0,1 are "old" (outside window).
        float[] pk = { 5, 5,  6, 6,  0, 1 };   // key0,key1 huge dot with q would dominate if unmasked
        float[] pv = { 100, 100,  200, 200,  3, 4 };
        // seqlens_k: last valid key index = total-1 = 3.
        var ctx = Ctx(
            ("q", F(new[] { 1, 1, 2 }, q)),
            ("k", F(new[] { 1, 1, 2 }, kNew)),
            ("v", F(new[] { 1, 1, 2 }, vNew)),
            ("pk", F(new[] { 1, 1, 3, 2 }, pk)),
            ("pv", F(new[] { 1, 1, 3, 2 }, pv)),
            ("seqlens", I(new[] { 1 }, 3)),
            ("totlen", I(new[] { 1 }, 4)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention",
                new[] { "q", "k", "v", "pk", "pv", "seqlens", "totlen" },
                new[] { "out", "presk", "presv" },
                new Dictionary<string, object>
                {
                    ["num_heads"] = 1L, ["kv_num_heads"] = 1L, ["local_window_size"] = 2L,
                }), ctx);
        float[] got = ctx.Get("out").Span.ToArray();

        // Combined keys/values [past0,past1,past2,new]; query qAbs=3, window=2 => keys {2,3} only.
        var kAll = new[,] { { 5f, 5f }, { 6f, 6f }, { 0f, 1f }, { 1f, 1f } };
        var vAll = new[,] { { 100f, 100f }, { 200f, 200f }, { 3f, 4f }, { 9f, 9f } };
        float scale = 1f / MathF.Sqrt(headDim);
        var qh = new[,] { { q[0], q[1] } };
        float[] expected = RefWindow(qh, kAll, vAll, scale, pastOffset: 3, localWindow: 2);

        Close(expected, got);

        // The old cached values (100,200) must NOT leak in: with window=2 the output is a convex
        // combo of v[2]=(3,4) and v[3]=(9,9), so each component stays well below 10.
        Assert.True(got[0] < 10f && got[1] < 10f,
            $"old cached values leaked through the window: got ({got[0]},{got[1]})");

        // Cross-check: WITHOUT the window the huge old keys WOULD dominate (output near (100,100)),
        // confirming the window is what excludes them.
        var ctx2 = Ctx(
            ("q", F(new[] { 1, 1, 2 }, q)),
            ("k", F(new[] { 1, 1, 2 }, kNew)),
            ("v", F(new[] { 1, 1, 2 }, vNew)),
            ("pk", F(new[] { 1, 1, 3, 2 }, pk)),
            ("pv", F(new[] { 1, 1, 3, 2 }, pv)),
            ("seqlens", I(new[] { 1 }, 3)),
            ("totlen", I(new[] { 1 }, 4)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention",
                new[] { "q", "k", "v", "pk", "pv", "seqlens", "totlen" },
                new[] { "out", "presk", "presv" },
                new Dictionary<string, object>
                {
                    ["num_heads"] = 1L, ["kv_num_heads"] = 1L, ["local_window_size"] = -1L,
                }), ctx2);
        float[] full = ctx2.Get("out").Span.ToArray();
        Assert.True(full[0] > 50f,
            $"sanity: unwindowed decode should be dominated by old keys; got ({full[0]},{full[1]})");
    }
}
