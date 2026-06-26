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
/// Direct-kernel tests for the ONNXRuntime contrib attention ops
/// (<see cref="MultiHeadAttentionKernel"/> and <see cref="GroupQueryAttentionKernel"/>).
/// Each test builds the node + context by hand and checks against a reference
/// scaled-dot-product attention computed inline in the test.
/// </summary>
public class AttentionContribTests
{
    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    // Reference non-causal SDPA for a single batch, single head: Q,K,V each [S, headDim].
    // Returns [S, headDim].
    private static float[] RefSdpa(float[,] q, float[,] k, float[,] v, float scale, bool causal)
    {
        int s = q.GetLength(0), hd = q.GetLength(1), sk = k.GetLength(0), vhd = v.GetLength(1);
        var outp = new float[s * vhd];
        for (int i = 0; i < s; i++)
        {
            int limit = causal ? i : sk - 1;
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

    // ---- MultiHeadAttention --------------------------------------------------------------------

    [Fact]
    public void Mha_Tiny_TwoHeads_MatchesReferenceSdpa()
    {
        // B=1, S=2, hidden=4, num_heads=2 -> head_dim=2. Q=K=V (self-attention).
        // Token0 = [1,0, 0,1], Token1 = [0,1, 1,0].
        float[] data = { 1, 0, 0, 1, 0, 1, 1, 0 };
        var q = F(new[] { 1, 2, 4 }, data);
        var k = F(new[] { 1, 2, 4 }, data);
        var v = F(new[] { 1, 2, 4 }, data);

        var ctx = Ctx(("q", q), ("k", k), ("v", v));
        new MultiHeadAttentionKernel().Execute(
            Node("MultiHeadAttention", new[] { "q", "k", "v" }, new[] { "out" },
                new Dictionary<string, object> { ["num_heads"] = 2L }), ctx);

        float scale = 1f / MathF.Sqrt(2f);

        // Head 0 uses dims [0,1] of each token; head 1 uses dims [2,3].
        // Head0: Q/K/V tokens = [[1,0],[0,1]].
        var h0q = new float[,] { { 1, 0 }, { 0, 1 } };
        float[] r0 = RefSdpa(h0q, h0q, h0q, scale, causal: false);
        // Head1: Q/K/V tokens = [[0,1],[1,0]].
        var h1q = new float[,] { { 0, 1 }, { 1, 0 } };
        float[] r1 = RefSdpa(h1q, h1q, h1q, scale, causal: false);

        // Reassemble [S, hidden] = head0 dims then head1 dims per token.
        float[] expected =
        {
            r0[0], r0[1], r1[0], r1[1],   // token0
            r0[2], r0[3], r1[2], r1[3],   // token1
        };

        Assert.Equal(new[] { 1, 2, 4 }, ctx.GetTensor("out").Shape.Dimensions.ToArray());
        Close(expected, ctx.Get("out").Span.ToArray());
    }

    [Fact]
    public void Mha_ExplicitScale_OverridesDefault()
    {
        // Single head, B=1, S=2, hidden=2. With scale=0 path vs explicit scale.
        float[] data = { 1, 0, 0, 1 };
        var ctx = Ctx(("q", F(new[] { 1, 2, 2 }, data)), ("k", F(new[] { 1, 2, 2 }, data)),
            ("v", F(new[] { 1, 2, 2 }, data)));
        new MultiHeadAttentionKernel().Execute(
            Node("MultiHeadAttention", new[] { "q", "k", "v" }, new[] { "out" },
                new Dictionary<string, object> { ["num_heads"] = 1L, ["scale"] = 0.25f }), ctx);

        var qm = new float[,] { { 1, 0 }, { 0, 1 } };
        float[] expected = RefSdpa(qm, qm, qm, 0.25f, causal: false);
        Close(expected, ctx.Get("out").Span.ToArray());
    }

    [Fact]
    public void Mha_AdditiveMask_BlocksKey()
    {
        // Single head B=1 S=2 hidden=2. Mask [B,Sq,Skv] with -inf on key 1 forces
        // each query to attend only to key 0 -> output == V[0] for both queries.
        float[] data = { 1, 0, 0, 1 };
        float ninf = -1e9f;
        var ctx = Ctx(("q", F(new[] { 1, 2, 2 }, data)), ("k", F(new[] { 1, 2, 2 }, data)),
            ("v", F(new[] { 1, 2, 2 }, data)),
            ("mask", F(new[] { 1, 2, 2 }, 0f, ninf, 0f, ninf)));
        new MultiHeadAttentionKernel().Execute(
            Node("MultiHeadAttention", new[] { "q", "k", "v", "", "mask" }, new[] { "out" },
                new Dictionary<string, object> { ["num_heads"] = 1L }), ctx);

        // Both queries collapse onto V[0] = [1,0].
        Close(new[] { 1f, 0f, 1f, 0f }, ctx.Get("out").Span.ToArray());
    }

    [Fact]
    public void Mha_Packed5D_Throws()
    {
        var ctx = Ctx(("q", F(new[] { 1, 1, 3, 2, 2 }, new float[12])));
        Assert.Throws<ModelSharpException>(() =>
            new MultiHeadAttentionKernel().Execute(
                Node("MultiHeadAttention", new[] { "q" }, new[] { "out" },
                    new Dictionary<string, object> { ["num_heads"] = 2L }), ctx));
    }

    // ---- GroupQueryAttention -------------------------------------------------------------------

    [Fact]
    public void Gqa_FourQueryHeads_TwoKvHeads_SharedPerGroup_Causal()
    {
        // B=1, S=3, num_heads=4, kv_num_heads=2, head_dim=2.
        // group_size=2: query heads {0,1} -> kv head 0; query heads {2,3} -> kv head 1.
        int s = 3, headDim = 2, numHeads = 4, kvHeads = 2;
        var rnd = new Random(7);
        float[] q = new float[s * numHeads * headDim];
        float[] k = new float[s * kvHeads * headDim];
        float[] v = new float[s * kvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rnd.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rnd.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rnd.NextDouble() * 2 - 1);

        var ctx = Ctx(("q", F(new[] { 1, s, numHeads * headDim }, q)),
            ("k", F(new[] { 1, s, kvHeads * headDim }, k)),
            ("v", F(new[] { 1, s, kvHeads * headDim }, v)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention", new[] { "q", "k", "v" }, new[] { "out", "pk", "pv" },
                new Dictionary<string, object> { ["num_heads"] = 4L, ["kv_num_heads"] = 2L }), ctx);

        float scale = 1f / MathF.Sqrt(headDim);
        var outActual = ctx.Get("out").Span.ToArray();

        // Reference: for each query head h, kv head = h/2, causal SDPA.
        for (int h = 0; h < numHeads; h++)
        {
            int g = h / (numHeads / kvHeads);
            var qh = new float[s, headDim];
            var kh = new float[s, headDim];
            var vh = new float[s, headDim];
            for (int t = 0; t < s; t++)
            for (int d = 0; d < headDim; d++)
            {
                qh[t, d] = q[(t * numHeads + h) * headDim + d];
                kh[t, d] = k[(t * kvHeads + g) * headDim + d];
                vh[t, d] = v[(t * kvHeads + g) * headDim + d];
            }
            float[] r = RefSdpa(qh, kh, vh, scale, causal: true);
            for (int t = 0; t < s; t++)
            for (int d = 0; d < headDim; d++)
            {
                float got = outActual[(t * numHeads + h) * headDim + d];
                Assert.True(MathF.Abs(r[t * headDim + d] - got) <= 1e-4f,
                    $"head {h} pos {t} dim {d}: expected {r[t * headDim + d]}, got {got}");
            }
        }

        // present_key/value shape: [B, kv_num_heads, totalSeq, head_dim].
        Assert.Equal(new[] { 1, kvHeads, s, headDim }, ctx.GetTensor("pk").Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1, kvHeads, s, headDim }, ctx.GetTensor("pv").Shape.Dimensions.ToArray());
    }

    [Fact]
    public void Gqa_CausalMask_FirstQueryAttendsOnlyToFirstKey()
    {
        // num_heads=2, kv_num_heads=1, head_dim=1, S=2. The first query (pos 0) is
        // causal-masked to key 0 only, so out[0] == V[0] regardless of V[1].
        var ctx = Ctx(
            ("q", F(new[] { 1, 2, 2 }, 1, 1, 1, 1)),   // [B,S,num_heads*hd]
            ("k", F(new[] { 1, 2, 1 }, 0.5f, 0.5f)),   // [B,S,kv_heads*hd]
            ("v", F(new[] { 1, 2, 1 }, 3f, 9f)));
        new GroupQueryAttentionKernel().Execute(
            Node("GroupQueryAttention", new[] { "q", "k", "v" }, new[] { "out" },
                new Dictionary<string, object> { ["num_heads"] = 2L, ["kv_num_heads"] = 1L }), ctx);

        float[] o = ctx.Get("out").Span.ToArray();
        // pos0, both heads share kv head 0; only key0 visible -> V[0]=3.
        Assert.True(MathF.Abs(o[0] - 3f) < 1e-5f, $"head0 pos0 = {o[0]}");
        Assert.True(MathF.Abs(o[1] - 3f) < 1e-5f, $"head1 pos0 = {o[1]}");
        // pos1 sees both keys; with equal keys (0.5,0.5) softmax is uniform -> mean(3,9)=6.
        Assert.True(MathF.Abs(o[2] - 6f) < 1e-5f, $"head0 pos1 = {o[2]}");
        Assert.True(MathF.Abs(o[3] - 6f) < 1e-5f, $"head1 pos1 = {o[3]}");
    }

    [Fact]
    public void Gqa_NumHeadsNotMultipleOfKv_Throws()
    {
        var ctx = Ctx(("q", F(new[] { 1, 1, 6 }, new float[6])),
            ("k", F(new[] { 1, 1, 4 }, new float[4])), ("v", F(new[] { 1, 1, 4 }, new float[4])));
        Assert.Throws<ModelSharpException>(() =>
            new GroupQueryAttentionKernel().Execute(
                Node("GroupQueryAttention", new[] { "q", "k", "v" }, new[] { "out" },
                    new Dictionary<string, object> { ["num_heads"] = 3L, ["kv_num_heads"] = 2L }), ctx));
    }
}
