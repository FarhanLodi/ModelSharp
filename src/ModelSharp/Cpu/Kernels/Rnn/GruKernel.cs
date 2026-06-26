using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Rnn;

/// <summary>
/// Gated Recurrent Unit (ONNX <c>GRU</c>). Supports forward / reverse / bidirectional, optional
/// bias and initial state, <c>linear_before_reset</c>, the <c>clip</c> attribute, the optional
/// <c>sequence_lens</c> input, and the Y / Y_h outputs. Default activations (Sigmoid, Tanh).
/// Gate weight order is ONNX zrh.
/// </summary>
public sealed class GruKernel : IKernel
{
    public string OpType => "GRU";

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Clamps <paramref name="v"/> into [-c, +c] — the ONNX gate-input clip.</summary>
    private static float Clip(float v, float c) => v > c ? c : (v < -c ? -c : v);

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> X = ctx.Get(node.Inputs[0]);   // [S, B, I]
        Tensor<float> W = ctx.Get(node.Inputs[1]);   // [D, 3H, I]  gate order zrh
        Tensor<float> R = ctx.Get(node.Inputs[2]);   // [D, 3H, H]
        Tensor<float>? B = Opt(node, ctx, 3);         // [D, 6H]
        Tensor<float>? initH = Opt(node, ctx, 5);     // [D, B, H]

        // Optional per-batch sequence lengths (ONNX dtype int32; read dtype-agnostically).
        long[]? seqLens = node.Inputs.Count > 4 && node.Inputs[4].Length > 0
            ? TensorInts.Read(ctx.GetTensor(node.Inputs[4]))   // [B]
            : null;

        System.ReadOnlySpan<int> xd = X.Shape.Dimensions;
        int S = xd[0], Bt = xd[1], I = xd[2];
        System.ReadOnlySpan<int> wd = W.Shape.Dimensions;
        int D = wd[0];
        int H = wd[1] / 3;

        string dir = Attr.Str(node, "direction", "forward");
        bool lbr = Attr.Int(node, "linear_before_reset", 0) != 0;
        float clip = Attr.Float(node, "clip", 0f);
        bool useClip = clip > 0f;                                   // absent / non-positive => no clip

        System.Span<float> xs = X.Span, ws = W.Span, rs = R.Span;
        System.Span<float> bs = B is null ? default : B.Span;
        System.Span<float> ih = initH is null ? default : initH.Span;

        var yAll = new float[S * D * Bt * H];
        var yH = new float[D * Bt * H];
        int wDir = 3 * H * I, rDir = 3 * H * H;

        for (int d = 0; d < D; d++)
        {
            bool reverse = dir == "reverse" || (dir == "bidirectional" && d == 1);
            var hPrev = new float[Bt * H];
            if (initH is not null) for (int i = 0; i < Bt * H; i++) hPrev[i] = ih[d * Bt * H + i];

            for (int step = 0; step < S; step++)
            {
                var hCur = new float[Bt * H];

                for (int b = 0; b < Bt; b++)
                {
                    int hBase = b * H;

                    // Per-batch sequence length. Beyond it the timestep is padding: carry the
                    // previous hidden state through and leave the Y slot at 0.
                    int len = seqLens is null ? S : (int)seqLens[b];
                    if (len < 0) len = 0; else if (len > S) len = S;
                    if (step >= len)
                    {
                        for (int h = 0; h < H; h++) hCur[hBase + h] = hPrev[hBase + h];
                        continue;
                    }

                    // Reverse walks the valid region [0, len) from high to low; padding stays at the tail.
                    int t = reverse ? len - 1 - step : step;
                    int xBase = t * Bt * I + b * I;

                    // Update gate z and reset gate r (need full vectors before the h gate).
                    var z = new float[H];
                    var rg = new float[H];
                    for (int h = 0; h < H; h++)
                    {
                        float gz = Bias(bs, B, d, H, 0, h);
                        float gr = Bias(bs, B, d, H, 1, h);
                        int wz = d * wDir + (0 * H + h) * I;
                        int wr = d * wDir + (1 * H + h) * I;
                        for (int k = 0; k < I; k++) { float xk = xs[xBase + k]; gz += xk * ws[wz + k]; gr += xk * ws[wr + k]; }
                        int rz = d * rDir + (0 * H + h) * H;
                        int rr = d * rDir + (1 * H + h) * H;
                        for (int j = 0; j < H; j++) { float hj = hPrev[hBase + j]; gz += hj * rs[rz + j]; gr += hj * rs[rr + j]; }
                        if (useClip) { gz = Clip(gz, clip); gr = Clip(gr, clip); }
                        z[h] = Sigmoid(gz);
                        rg[h] = Sigmoid(gr);
                    }

                    // Candidate h gate, then blend.
                    for (int h = 0; h < H; h++)
                    {
                        int wh = d * wDir + (2 * H + h) * I;
                        int rh = d * rDir + (2 * H + h) * H;
                        float xWh = 0f;
                        for (int k = 0; k < I; k++) xWh += xs[xBase + k] * ws[wh + k];
                        float wbh = B is null ? 0f : bs[d * 6 * H + 2 * H + h];
                        float rbh = B is null ? 0f : bs[d * 6 * H + 5 * H + h];

                        // Pre-activation argument to the candidate (g) activation.
                        float gh;
                        if (lbr)
                        {
                            float rec = rbh;
                            for (int j = 0; j < H; j++) rec += hPrev[hBase + j] * rs[rh + j];
                            gh = xWh + wbh + rg[h] * rec;
                        }
                        else
                        {
                            float rec = 0f;
                            for (int j = 0; j < H; j++) rec += (rg[j] * hPrev[hBase + j]) * rs[rh + j];
                            gh = xWh + rec + wbh + rbh;
                        }
                        if (useClip) gh = Clip(gh, clip);
                        float ht = MathF.Tanh(gh);

                        float Ht = (1f - z[h]) * ht + z[h] * hPrev[hBase + h];
                        hCur[hBase + h] = Ht;
                        yAll[((t * D + d) * Bt + b) * H + h] = Ht;
                    }
                }

                hPrev = hCur;
            }

            for (int i = 0; i < Bt * H; i++) yH[d * Bt * H + i] = hPrev[i];
        }

        if (node.Outputs.Count > 0 && node.Outputs[0].Length > 0)
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(S, D, Bt, H), yAll));
        if (node.Outputs.Count > 1 && node.Outputs[1].Length > 0)
            ctx.Set(node.Outputs[1], new Tensor<float>(new TensorShape(D, Bt, H), yH));
    }

    private static Tensor<float>? Opt(GraphNode node, GraphContext ctx, int idx)
        => node.Inputs.Count > idx && node.Inputs[idx].Length > 0 ? ctx.Get(node.Inputs[idx]) : null;

    /// <summary>Combined input+recurrence bias for gate (0=z,1=r) at unit h.</summary>
    private static float Bias(System.Span<float> bs, Tensor<float>? b, int d, int H, int gate, int h)
    {
        if (b is null) return 0f;
        int baseB = d * 6 * H;
        return bs[baseB + gate * H + h] + bs[baseB + (3 + gate) * H + h];
    }
}
