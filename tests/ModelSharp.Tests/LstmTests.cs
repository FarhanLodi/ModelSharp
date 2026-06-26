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

public class LstmTests
{
    private static string Case(string name, string file) =>
        Path.Combine(AppContext.BaseDirectory, "assets", name, file);

    private static void RunOnnxLstmCase(string caseName, int numInputs)
    {
        ModelGraph graph = OnnxModelLoader.LoadModel(Case(caseName, "model.onnx"));
        using var engine = new ManagedCpuEngine(graph);

        // Feed the graph inputs in declared order (input_0, input_1, ...).
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
    public void Lstm_Defaults_Matches_Onnx_Reference() => RunOnnxLstmCase("test_lstm_defaults", 3);

    [Fact]
    public void Lstm_With_Initial_Bias_Matches_Onnx_Reference() => RunOnnxLstmCase("test_lstm_with_initial_bias", 4);

    // ---------------------------------------------------------------------------------------------
    // Direct-kernel tests for the extended attributes: peepholes (P), clip, input_forget and the
    // optional sequence_lens input. These build a GraphNode / GraphContext by hand so the new
    // optional inputs and attributes can be exercised in isolation. Single direction (D=1) unless
    // noted, so the independent reference below can stay simple.
    // ---------------------------------------------------------------------------------------------

    private static float Sig(float v) => 1f / (1f + MathF.Exp(-v));

    /// <summary>
    /// Runs <see cref="LstmKernel"/> for one direction and returns (Y, Y_h, Y_c) as flat arrays.
    /// Optional inputs are wired with empty "" slots when their array is null, matching how the
    /// ONNX loader represents omitted optional inputs.
    /// </summary>
    private static (float[] Y, float[] Yh, float[] Yc) RunLstm(
        int S, int Bt, int I, int H, string direction,
        float[] X, float[] W, float[] R,
        float[]? Bias, int[]? seqLens, float[]? initH, float[]? initC, float[]? P,
        float clip, bool inputForget)
    {
        int D = direction == "bidirectional" ? 2 : 1;
        var inputs = new List<string> { "X", "W", "R" };
        var values = new Dictionary<string, Tensor>
        {
            ["X"] = new Tensor<float>(new TensorShape(S, Bt, I), (float[])X.Clone()),
            ["W"] = new Tensor<float>(new TensorShape(D, 4 * H, I), (float[])W.Clone()),
            ["R"] = new Tensor<float>(new TensorShape(D, 4 * H, H), (float[])R.Clone()),
        };

        inputs.Add(Bias is not null ? "B" : "");
        if (Bias is not null) values["B"] = new Tensor<float>(new TensorShape(D, 8 * H), (float[])Bias.Clone());

        inputs.Add(seqLens is not null ? "seq" : "");
        if (seqLens is not null) values["seq"] = new Tensor<int>(new TensorShape(Bt), (int[])seqLens.Clone());

        inputs.Add(initH is not null ? "ih" : "");
        if (initH is not null) values["ih"] = new Tensor<float>(new TensorShape(D, Bt, H), (float[])initH.Clone());

        inputs.Add(initC is not null ? "ic" : "");
        if (initC is not null) values["ic"] = new Tensor<float>(new TensorShape(D, Bt, H), (float[])initC.Clone());

        inputs.Add(P is not null ? "P" : "");
        if (P is not null) values["P"] = new Tensor<float>(new TensorShape(D, 3 * H), (float[])P.Clone());

        var attrs = new Dictionary<string, object> { ["direction"] = direction };
        if (clip > 0f) attrs["clip"] = clip;
        if (inputForget) attrs["input_forget"] = 1L;

        var node = new GraphNode("LSTM", "lstm", inputs, new[] { "Y", "Yh", "Yc" }, attrs);
        var ctx = new GraphContext(values);
        new LstmKernel().Execute(node, ctx);

        return (ctx.Get("Y").Span.ToArray(), ctx.Get("Yh").Span.ToArray(), ctx.Get("Yc").Span.ToArray());
    }

    /// <summary>Plain, independently written single-direction LSTM reference following the ONNX spec.</summary>
    private static (float[] Y, float[] Yh, float[] Yc) RefLstm(
        int S, int Bt, int I, int H,
        float[] X, float[] W, float[] R, float[]? Bb,
        float[]? P, float[]? ih, float[]? ic,
        int[]? seqLens, float clip, bool inputForget, bool reverse)
    {
        float Clp(float v) => clip > 0f ? MathF.Max(-clip, MathF.Min(clip, v)) : v;

        var Y = new float[S * Bt * H];
        var hPrev = new float[Bt * H];
        var cPrev = new float[Bt * H];
        if (ih is not null) Array.Copy(ih, hPrev, Bt * H);
        if (ic is not null) Array.Copy(ic, cPrev, Bt * H);

        for (int step = 0; step < S; step++)
        {
            var hCur = (float[])hPrev.Clone();
            var cCur = (float[])cPrev.Clone();
            for (int b = 0; b < Bt; b++)
            {
                int len = seqLens is null ? S : seqLens[b];
                if (step >= len) continue;                 // padding: state carried, Y stays 0
                int t = reverse ? len - 1 - step : step;
                for (int h = 0; h < H; h++)
                {
                    float gi = 0, go = 0, gf = 0, gc = 0;
                    for (int k = 0; k < I; k++)
                    {
                        float xk = X[t * Bt * I + b * I + k];
                        gi += xk * W[(0 * H + h) * I + k];
                        go += xk * W[(1 * H + h) * I + k];
                        gf += xk * W[(2 * H + h) * I + k];
                        gc += xk * W[(3 * H + h) * I + k];
                    }
                    for (int j = 0; j < H; j++)
                    {
                        float hj = hPrev[b * H + j];
                        gi += hj * R[(0 * H + h) * H + j];
                        go += hj * R[(1 * H + h) * H + j];
                        gf += hj * R[(2 * H + h) * H + j];
                        gc += hj * R[(3 * H + h) * H + j];
                    }
                    if (Bb is not null)
                    {
                        gi += Bb[0 * H + h] + Bb[4 * H + h];
                        go += Bb[1 * H + h] + Bb[5 * H + h];
                        gf += Bb[2 * H + h] + Bb[6 * H + h];
                        gc += Bb[3 * H + h] + Bb[7 * H + h];
                    }
                    float cP = cPrev[b * H + h];
                    if (P is not null)
                    {
                        gi += P[0 * H + h] * cP;   // Pi
                        gf += P[2 * H + h] * cP;   // Pf
                    }
                    gi = Clp(gi); gf = Clp(gf); gc = Clp(gc);
                    float ft = Sig(gf);
                    float it = inputForget ? 1f - ft : Sig(gi);
                    float ct = MathF.Tanh(gc);
                    float Ct = ft * cP + it * ct;
                    if (P is not null) go += P[1 * H + h] * Ct;   // Po (.) C_t
                    go = Clp(go);
                    float ot = Sig(go);
                    float Ht = ot * MathF.Tanh(Ct);
                    cCur[b * H + h] = Ct;
                    hCur[b * H + h] = Ht;
                    Y[(t * Bt + b) * H + h] = Ht;
                }
            }
            hPrev = hCur; cPrev = cCur;
        }
        return (Y, hPrev, cPrev);
    }

    private static void AssertClose(float[] expected, float[] actual, float tol = 2e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) < tol, $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    [Fact]
    public void Lstm_Clip_Clamps_Gate_PreActivations()
    {
        // 1x1x1 cell, huge weights => unclipped gates saturate. clip=2 bounds every pre-activation.
        var X = new[] { 1f };
        var W = new[] { 100f, 100f, 100f, 100f };   // Wi, Wo, Wf, Wc
        var R = new[] { 0f, 0f, 0f, 0f };

        var (y, yh, yc) = RunLstm(1, 1, 1, 1, "forward", X, W, R, null, null, null, null, null, clip: 2f, inputForget: false);

        // Hand reference with each gate pre-activation clamped to +2.
        float g = MathF.Min(2f, 100f);
        float it = Sig(g), ft = Sig(g), ot = Sig(g);
        float Ct = ft * 0f + it * MathF.Tanh(g);
        float Ht = ot * MathF.Tanh(Ct);
        Assert.True(MathF.Abs(y[0] - Ht) < 1e-5f, $"clipped Y expected {Ht}, got {y[0]}");
        Assert.True(MathF.Abs(yh[0] - Ht) < 1e-5f);
        Assert.True(MathF.Abs(yc[0] - Ct) < 1e-5f);

        // Without clip the gates saturate to ~1 => Ht == tanh(1); clip must visibly change the result.
        var (yNo, _, _) = RunLstm(1, 1, 1, 1, "forward", X, W, R, null, null, null, null, null, clip: 0f, inputForget: false);
        Assert.True(MathF.Abs(yNo[0] - MathF.Tanh(1f)) < 1e-4f, $"unclipped Y expected tanh(1), got {yNo[0]}");
        Assert.True(MathF.Abs(yNo[0] - y[0]) > 0.05f, "clip did not change the output");
    }

    [Fact]
    public void Lstm_InputForget_Ignores_InputGate_Weights()
    {
        // With input_forget=1, i_t = 1 - f_t, so the input-gate weights (gate slot 0) must not matter.
        int S = 2, Bt = 1, I = 2, H = 2;
        var rng = new Random(7);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }

        var X = Rand(S * Bt * I);
        var W = Rand(4 * H * I);
        var R = Rand(4 * H * H);
        var Bb = Rand(8 * H);

        // A copy of W with the input-gate rows (rows [0, H)) overwritten with different values.
        var W2 = (float[])W.Clone();
        for (int h = 0; h < H; h++)
            for (int k = 0; k < I; k++)
                W2[(0 * H + h) * I + k] = (float)(rng.NextDouble() * 5 - 2.5);

        var (yA, yhA, ycA) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, null, null, null, null, 0f, inputForget: true);
        var (yB, yhB, ycB) = RunLstm(S, Bt, I, H, "forward", X, W2, R, Bb, null, null, null, null, 0f, inputForget: true);
        AssertClose(yA, yB);
        AssertClose(yhA, yhB);
        AssertClose(ycA, ycB);

        // Sanity: without coupling the same input-gate change DOES move the output.
        var (yC0, _, _) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, null, null, null, null, 0f, inputForget: false);
        var (yC1, _, _) = RunLstm(S, Bt, I, H, "forward", X, W2, R, Bb, null, null, null, null, 0f, inputForget: false);
        bool differs = false;
        for (int i = 0; i < yC0.Length; i++) if (MathF.Abs(yC0[i] - yC1[i]) > 1e-4f) differs = true;
        Assert.True(differs, "input-gate weights should matter when input_forget=0");
    }

    [Fact]
    public void Lstm_InputForget_Matches_Reference()
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(11);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var X = Rand(S * Bt * I); var W = Rand(4 * H * I); var R = Rand(4 * H * H); var Bb = Rand(8 * H);

        var (y, yh, yc) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, null, null, null, null, 0f, inputForget: true);
        var (ry, ryh, ryc) = RefLstm(S, Bt, I, H, X, W, R, Bb, null, null, null, null, 0f, inputForget: true, reverse: false);
        AssertClose(ry, y); AssertClose(ryh, yh); AssertClose(ryc, yc);
    }

    [Fact]
    public void Lstm_Peephole_Output_Gate_Matches_Hand_Reference()
    {
        // S=1 => C_{t-1}=0, so only the output-gate peephole Po (which uses the *current* C_t) acts.
        var X = new[] { 1f };
        var W = new[] { 0.5f, 0.5f, 0.5f, 0.5f };   // Wi, Wo, Wf, Wc
        var R = new[] { 0f, 0f, 0f, 0f };
        var P = new[] { 0f, 1.0f, 0f };             // Pi, Po, Pf

        var (y, _, yc) = RunLstm(1, 1, 1, 1, "forward", X, W, R, null, null, null, null, P, 0f, false);

        float it = Sig(0.5f), ft = Sig(0.5f);
        float Ct = it * MathF.Tanh(0.5f);          // ft * 0 + it * ctilde
        float ot = Sig(0.5f + 1.0f * Ct);          // output gate sees current C_t through Po
        float Ht = ot * MathF.Tanh(Ct);
        Assert.True(MathF.Abs(y[0] - Ht) < 1e-5f, $"expected {Ht}, got {y[0]}");
        Assert.True(MathF.Abs(yc[0] - Ct) < 1e-5f);

        // No peephole must differ (output gate would be Sig(0.5)).
        var (yNo, _, _) = RunLstm(1, 1, 1, 1, "forward", X, W, R, null, null, null, null, null, 0f, false);
        Assert.True(MathF.Abs(yNo[0] - y[0]) > 1e-3f, "peephole did not change the output gate");
    }

    [Fact]
    public void Lstm_Peephole_Matches_Reference_MultiStep()
    {
        // S>=2 so C_{t-1} is non-zero and the i/f peepholes (Pi, Pf) are exercised too.
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(23);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var X = Rand(S * Bt * I); var W = Rand(4 * H * I); var R = Rand(4 * H * H); var Bb = Rand(8 * H); var P = Rand(3 * H);

        var (y, yh, yc) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, null, null, null, P, 0f, false);
        var (ry, ryh, ryc) = RefLstm(S, Bt, I, H, X, W, R, Bb, P, null, null, null, 0f, false, reverse: false);
        AssertClose(ry, y, 2e-5f); AssertClose(ryh, yh, 2e-5f); AssertClose(ryc, yc, 2e-5f);

        // Removing the peepholes must change the result.
        var (y0, _, _) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, null, null, null, null, 0f, false);
        bool differs = false;
        for (int i = 0; i < y.Length; i++) if (MathF.Abs(y[i] - y0[i]) > 1e-4f) differs = true;
        Assert.True(differs, "peepholes did not affect the output");
    }

    [Fact]
    public void Lstm_SequenceLens_AllFull_Equals_NoSeqLens()
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(31);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var X = Rand(S * Bt * I); var W = Rand(4 * H * I); var R = Rand(4 * H * H); var Bb = Rand(8 * H);

        var (y0, yh0, yc0) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, null, null, null, null, 0f, false);
        var (y1, yh1, yc1) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, new[] { S, S }, null, null, null, 0f, false);
        AssertClose(y0, y1); AssertClose(yh0, yh1); AssertClose(yc0, yc1);
    }

    [Fact]
    public void Lstm_SequenceLens_Masks_Padded_Timesteps()
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(41);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var W = Rand(4 * H * I); var R = Rand(4 * H * H); var Bb = Rand(8 * H);
        var X = Rand(S * Bt * I);
        var seq = new[] { 3, 2 };   // batch 0 full length, batch 1 stops after t=1

        var (y, yh, yc) = RunLstm(S, Bt, I, H, "forward", X, W, R, Bb, seq, null, null, null, 0f, false);

        // Padded timestep (t=2) for batch 1 must be all zero in Y.
        for (int h = 0; h < H; h++)
            Assert.Equal(0f, y[(2 * Bt + 1) * H + h]);

        // Y_h / Y_c for batch 1 hold the state at its last valid step (t=1).
        for (int h = 0; h < H; h++)
        {
            Assert.True(MathF.Abs(yh[1 * H + h] - y[(1 * Bt + 1) * H + h]) < 1e-6f);
        }

        // Changing batch 1's input at its padded step must not change any of batch 1's outputs.
        var X2 = (float[])X.Clone();
        for (int k = 0; k < I; k++) X2[2 * Bt * I + 1 * I + k] += 5f;   // perturb X[t=2, b=1]
        var (y2, yh2, yc2) = RunLstm(S, Bt, I, H, "forward", X2, W, R, Bb, seq, null, null, null, 0f, false);
        for (int t = 0; t < S; t++)
            for (int h = 0; h < H; h++)
                Assert.True(MathF.Abs(y[(t * Bt + 1) * H + h] - y2[(t * Bt + 1) * H + h]) < 1e-6f, "padded input leaked into output");
        for (int h = 0; h < H; h++)
        {
            Assert.True(MathF.Abs(yh[1 * H + h] - yh2[1 * H + h]) < 1e-6f);
            Assert.True(MathF.Abs(yc[1 * H + h] - yc2[1 * H + h]) < 1e-6f);
        }

        // Sanity: batch 0 (full length) is unaffected and its last row is non-trivial.
        bool batch0Live = false;
        for (int h = 0; h < H; h++) if (MathF.Abs(y[(2 * Bt + 0) * H + h]) > 1e-6f) batch0Live = true;
        Assert.True(batch0Live, "batch 0 should still produce output at its final timestep");

        // Cross-check the full numbers against the independent reference.
        var (ry, ryh, ryc) = RefLstm(S, Bt, I, H, X, W, R, Bb, null, null, null, seq, 0f, false, reverse: false);
        AssertClose(ry, y); AssertClose(ryh, yh); AssertClose(ryc, yc);
    }

    [Fact]
    public void Lstm_SequenceLens_Reverse_Matches_Reference()
    {
        int S = 3, Bt = 2, I = 2, H = 2;
        var rng = new Random(53);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var W = Rand(4 * H * I); var R = Rand(4 * H * H); var Bb = Rand(8 * H);
        var X = Rand(S * Bt * I);
        var seq = new[] { 3, 2 };

        var (y, yh, yc) = RunLstm(S, Bt, I, H, "reverse", X, W, R, Bb, seq, null, null, null, 0f, false);
        var (ry, ryh, ryc) = RefLstm(S, Bt, I, H, X, W, R, Bb, null, null, null, seq, 0f, false, reverse: true);
        AssertClose(ry, y); AssertClose(ryh, yh); AssertClose(ryc, yc);

        // Padded tail (t=2) for batch 1 is still zero; in reverse, Y_h holds the final processed step t=0.
        for (int h = 0; h < H; h++)
        {
            Assert.Equal(0f, y[(2 * Bt + 1) * H + h]);
            Assert.True(MathF.Abs(yh[1 * H + h] - y[(0 * Bt + 1) * H + h]) < 1e-6f);
        }
    }
}
