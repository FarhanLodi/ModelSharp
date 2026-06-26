using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Rnn;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class GruTests
{
    private static string Case(string name, string file) =>
        Path.Combine(AppContext.BaseDirectory, "assets", name, file);

    private static void RunOnnxRnnCase(string caseName, int numInputs)
    {
        ModelGraph graph = OnnxModelLoader.LoadModel(Case(caseName, "model.onnx"));
        using var engine = new ManagedCpuEngine(graph);

        Assert.Equal(numInputs, engine.Inputs.Count);
        var feeds = new Dictionary<string, NamedTensor>();
        for (int i = 0; i < numInputs; i++)
        {
            string name = engine.Inputs[i].Name;
            Tensor<float> t = OnnxModelLoader.LoadTensor(Case(caseName, $"test_data_set_0/input_{i}.pb"));
            feeds[name] = new NamedTensor(name, t);
        }

        Tensor<float> actual = engine.Run(feeds).Values.Single().Data;
        Tensor<float> expected = OnnxModelLoader.LoadTensor(Case(caseName, "test_data_set_0/output_0.pb"));

        Assert.Equal(expected.Length, actual.Length);
        float[] e = expected.Span.ToArray();
        float[] a = actual.Span.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) < 1e-4f, $"[{i}] expected {e[i]}, got {a[i]}");
    }

    [Fact]
    public void Gru_Defaults_Matches_Onnx_Reference() => RunOnnxRnnCase("test_gru_defaults", 3);

    [Fact]
    public void Gru_With_Initial_Bias_Matches_Onnx_Reference() => RunOnnxRnnCase("test_gru_with_initial_bias", 4);

    // ---------------------------------------------------------------------------------------------
    // Direct-kernel tests for the extended attributes: clip and the optional sequence_lens input.
    // These build a GraphNode / GraphContext by hand. Single direction (D=1) unless noted.
    // ---------------------------------------------------------------------------------------------

    private static float Sig(float v) => 1f / (1f + MathF.Exp(-v));

    /// <summary>Runs <see cref="GruKernel"/> for one direction and returns (Y, Y_h) as flat arrays.</summary>
    private static (float[] Y, float[] Yh) RunGru(
        int S, int Bt, int I, int H, string direction,
        float[] X, float[] W, float[] R,
        float[]? Bias, int[]? seqLens, float[]? initH,
        float clip, bool lbr)
    {
        int D = direction == "bidirectional" ? 2 : 1;
        var inputs = new List<string> { "X", "W", "R" };
        var values = new Dictionary<string, Tensor>
        {
            ["X"] = new Tensor<float>(new TensorShape(S, Bt, I), (float[])X.Clone()),
            ["W"] = new Tensor<float>(new TensorShape(D, 3 * H, I), (float[])W.Clone()),
            ["R"] = new Tensor<float>(new TensorShape(D, 3 * H, H), (float[])R.Clone()),
        };

        inputs.Add(Bias is not null ? "B" : "");
        if (Bias is not null) values["B"] = new Tensor<float>(new TensorShape(D, 6 * H), (float[])Bias.Clone());

        inputs.Add(seqLens is not null ? "seq" : "");
        if (seqLens is not null) values["seq"] = new Tensor<int>(new TensorShape(Bt), (int[])seqLens.Clone());

        inputs.Add(initH is not null ? "ih" : "");
        if (initH is not null) values["ih"] = new Tensor<float>(new TensorShape(D, Bt, H), (float[])initH.Clone());

        var attrs = new Dictionary<string, object> { ["direction"] = direction };
        if (clip > 0f) attrs["clip"] = clip;
        if (lbr) attrs["linear_before_reset"] = 1L;

        var node = new GraphNode("GRU", "gru", inputs, new[] { "Y", "Yh" }, attrs);
        var ctx = new GraphContext(values);
        new GruKernel().Execute(node, ctx);

        return (ctx.Get("Y").Span.ToArray(), ctx.Get("Yh").Span.ToArray());
    }

    /// <summary>Plain, independently written single-direction GRU reference following the ONNX spec.</summary>
    private static (float[] Y, float[] Yh) RefGru(
        int S, int Bt, int I, int H,
        float[] X, float[] W, float[] R, float[]? Bb,
        float[]? ih, int[]? seqLens, float clip, bool lbr, bool reverse)
    {
        float Clp(float v) => clip > 0f ? MathF.Max(-clip, MathF.Min(clip, v)) : v;

        var Y = new float[S * Bt * H];
        var hPrev = new float[Bt * H];
        if (ih is not null) Array.Copy(ih, hPrev, Bt * H);

        for (int step = 0; step < S; step++)
        {
            var hCur = (float[])hPrev.Clone();
            for (int b = 0; b < Bt; b++)
            {
                int len = seqLens is null ? S : seqLens[b];
                if (step >= len) continue;                 // padding: state carried, Y stays 0
                int t = reverse ? len - 1 - step : step;

                var z = new float[H];
                var rg = new float[H];
                for (int h = 0; h < H; h++)
                {
                    float gz = 0, gr = 0;
                    for (int k = 0; k < I; k++) { float xk = X[t * Bt * I + b * I + k]; gz += xk * W[(0 * H + h) * I + k]; gr += xk * W[(1 * H + h) * I + k]; }
                    for (int j = 0; j < H; j++) { float hj = hPrev[b * H + j]; gz += hj * R[(0 * H + h) * H + j]; gr += hj * R[(1 * H + h) * H + j]; }
                    if (Bb is not null) { gz += Bb[0 * H + h] + Bb[3 * H + h]; gr += Bb[1 * H + h] + Bb[4 * H + h]; }
                    gz = Clp(gz); gr = Clp(gr);
                    z[h] = Sig(gz); rg[h] = Sig(gr);
                }

                for (int h = 0; h < H; h++)
                {
                    float xWh = 0f;
                    for (int k = 0; k < I; k++) xWh += X[t * Bt * I + b * I + k] * W[(2 * H + h) * I + k];
                    float wbh = Bb is null ? 0f : Bb[2 * H + h];
                    float rbh = Bb is null ? 0f : Bb[5 * H + h];

                    float gh;
                    if (lbr)
                    {
                        float rec = rbh;
                        for (int j = 0; j < H; j++) rec += hPrev[b * H + j] * R[(2 * H + h) * H + j];
                        gh = xWh + wbh + rg[h] * rec;
                    }
                    else
                    {
                        float rec = 0f;
                        for (int j = 0; j < H; j++) rec += (rg[j] * hPrev[b * H + j]) * R[(2 * H + h) * H + j];
                        gh = xWh + rec + wbh + rbh;
                    }
                    gh = Clp(gh);
                    float ht = MathF.Tanh(gh);
                    float Ht = (1f - z[h]) * ht + z[h] * hPrev[b * H + h];
                    hCur[b * H + h] = Ht;
                    Y[(t * Bt + b) * H + h] = Ht;
                }
            }
            hPrev = hCur;
        }
        return (Y, hPrev);
    }

    private static void AssertClose(float[] expected, float[] actual, float tol = 2e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) < tol, $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    [Fact]
    public void Gru_Clip_Clamps_Gate_PreActivations()
    {
        // 1x1x1 cell, huge weights => unclipped gates saturate. clip=2 bounds every pre-activation.
        var X = new[] { 1f };
        var W = new[] { 100f, 100f, 100f };   // Wz, Wr, Wh
        var R = new[] { 0f, 0f, 0f };

        var (y, yh) = RunGru(1, 1, 1, 1, "forward", X, W, R, null, null, null, clip: 2f, lbr: false);

        float g = MathF.Min(2f, 100f);
        float z = Sig(g);
        float ht = MathF.Tanh(g);
        float Ht = (1f - z) * ht;             // z * hPrev(=0) drops out
        Assert.True(MathF.Abs(y[0] - Ht) < 1e-5f, $"clipped Y expected {Ht}, got {y[0]}");
        Assert.True(MathF.Abs(yh[0] - Ht) < 1e-5f);

        // Without clip: z -> 1, so Ht -> 0; clip must visibly change the result.
        var (yNo, _) = RunGru(1, 1, 1, 1, "forward", X, W, R, null, null, null, clip: 0f, lbr: false);
        Assert.True(MathF.Abs(yNo[0]) < 1e-4f, $"unclipped Y expected ~0, got {yNo[0]}");
        Assert.True(MathF.Abs(yNo[0] - y[0]) > 0.05f, "clip did not change the output");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gru_Clip_Matches_Reference(bool lbr)
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(lbr ? 67 : 71);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 6 - 3); return a; }
        var X = Rand(S * Bt * I); var W = Rand(3 * H * I); var R = Rand(3 * H * H); var Bb = Rand(6 * H);

        var (y, yh) = RunGru(S, Bt, I, H, "forward", X, W, R, Bb, null, null, clip: 1.5f, lbr: lbr);
        var (ry, ryh) = RefGru(S, Bt, I, H, X, W, R, Bb, null, null, clip: 1.5f, lbr: lbr, reverse: false);
        AssertClose(ry, y, 2e-5f); AssertClose(ryh, yh, 2e-5f);

        // The strong inputs guarantee clipping actually bites: removing it changes the result.
        var (y0, _) = RunGru(S, Bt, I, H, "forward", X, W, R, Bb, null, null, clip: 0f, lbr: lbr);
        bool differs = false;
        for (int i = 0; i < y.Length; i++) if (MathF.Abs(y[i] - y0[i]) > 1e-4f) differs = true;
        Assert.True(differs, "clip did not affect the output");
    }

    [Fact]
    public void Gru_SequenceLens_AllFull_Equals_NoSeqLens()
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(83);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var X = Rand(S * Bt * I); var W = Rand(3 * H * I); var R = Rand(3 * H * H); var Bb = Rand(6 * H);

        var (y0, yh0) = RunGru(S, Bt, I, H, "forward", X, W, R, Bb, null, null, 0f, false);
        var (y1, yh1) = RunGru(S, Bt, I, H, "forward", X, W, R, Bb, new[] { S, S }, null, 0f, false);
        AssertClose(y0, y1); AssertClose(yh0, yh1);
    }

    [Fact]
    public void Gru_SequenceLens_Masks_Padded_Timesteps()
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(97);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var W = Rand(3 * H * I); var R = Rand(3 * H * H); var Bb = Rand(6 * H);
        var X = Rand(S * Bt * I);
        var seq = new[] { 3, 2 };

        var (y, yh) = RunGru(S, Bt, I, H, "forward", X, W, R, Bb, seq, null, 0f, false);

        // Padded timestep (t=2) for batch 1 must be all zero in Y.
        for (int h = 0; h < H; h++)
            Assert.Equal(0f, y[(2 * Bt + 1) * H + h]);

        // Y_h for batch 1 holds the state at its last valid step (t=1).
        for (int h = 0; h < H; h++)
            Assert.True(MathF.Abs(yh[1 * H + h] - y[(1 * Bt + 1) * H + h]) < 1e-6f);

        // Changing batch 1's input at its padded step must not change any of batch 1's outputs.
        var X2 = (float[])X.Clone();
        for (int k = 0; k < I; k++) X2[2 * Bt * I + 1 * I + k] += 5f;   // perturb X[t=2, b=1]
        var (y2, yh2) = RunGru(S, Bt, I, H, "forward", X2, W, R, Bb, seq, null, 0f, false);
        for (int t = 0; t < S; t++)
            for (int h = 0; h < H; h++)
                Assert.True(MathF.Abs(y[(t * Bt + 1) * H + h] - y2[(t * Bt + 1) * H + h]) < 1e-6f, "padded input leaked into output");
        for (int h = 0; h < H; h++)
            Assert.True(MathF.Abs(yh[1 * H + h] - yh2[1 * H + h]) < 1e-6f);

        // Cross-check the full numbers against the independent reference.
        var (ry, ryh) = RefGru(S, Bt, I, H, X, W, R, Bb, null, seq, 0f, false, reverse: false);
        AssertClose(ry, y); AssertClose(ryh, yh);
    }

    [Fact]
    public void Gru_SequenceLens_Reverse_Matches_Reference()
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(101);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var W = Rand(3 * H * I); var R = Rand(3 * H * H); var Bb = Rand(6 * H);
        var X = Rand(S * Bt * I);
        var seq = new[] { 3, 2 };

        var (y, yh) = RunGru(S, Bt, I, H, "reverse", X, W, R, Bb, seq, null, 0f, false);
        var (ry, ryh) = RefGru(S, Bt, I, H, X, W, R, Bb, null, seq, 0f, false, reverse: true);
        AssertClose(ry, y); AssertClose(ryh, yh);

        // Padded tail (t=2) for batch 1 is still zero; in reverse, Y_h holds the final processed step t=0.
        for (int h = 0; h < H; h++)
        {
            Assert.Equal(0f, y[(2 * Bt + 1) * H + h]);
            Assert.True(MathF.Abs(yh[1 * H + h] - y[(0 * Bt + 1) * H + h]) < 1e-6f);
        }
    }
}
