using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Rnn;

/// <summary>
/// Long Short-Term Memory (ONNX <c>LSTM</c>). Supports forward / reverse / bidirectional,
/// optional bias and initial states, optional peephole weights (the <c>P</c> input), the
/// <c>clip</c> and <c>input_forget</c> attributes, the optional <c>sequence_lens</c> input,
/// and the Y / Y_h / Y_c outputs (any subset, including omitted "" slots). Default activations
/// (Sigmoid, Tanh, Tanh). Gate weight order is ONNX iofc; peephole order is iof.
/// </summary>
public sealed class LstmKernel : IKernel
{
    public string OpType => "LSTM";

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Clamps <paramref name="v"/> into [-c, +c] — the ONNX gate-input clip.</summary>
    private static float Clip(float v, float c) => v > c ? c : (v < -c ? -c : v);

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> X = ctx.Get(node.Inputs[0]);   // [S, B, I]
        Tensor<float> W = ctx.Get(node.Inputs[1]);   // [D, 4H, I]   gate order iofc
        Tensor<float> R = ctx.Get(node.Inputs[2]);   // [D, 4H, H]
        Tensor<float>? B = Opt(node, ctx, 3);         // [D, 8H]      (Wb_iofc, Rb_iofc)
        Tensor<float>? initH = Opt(node, ctx, 5);     // [D, B, H]
        Tensor<float>? initC = Opt(node, ctx, 6);     // [D, B, H]
        Tensor<float>? P = Opt(node, ctx, 7);         // [D, 3H]      peephole order iof

        // Optional per-batch sequence lengths (ONNX dtype int32; read dtype-agnostically).
        long[]? seqLens = node.Inputs.Count > 4 && node.Inputs[4].Length > 0
            ? TensorInts.Read(ctx.GetTensor(node.Inputs[4]))   // [B]
            : null;

        System.ReadOnlySpan<int> xd = X.Shape.Dimensions;
        int S = xd[0], Bt = xd[1], I = xd[2];
        System.ReadOnlySpan<int> wd = W.Shape.Dimensions;
        int D = wd[0];
        int H = wd[1] / 4;

        string dir = Attr.Str(node, "direction", "forward");
        float clip = Attr.Float(node, "clip", 0f);
        bool useClip = clip > 0f;                                   // absent / non-positive => no clip
        bool inputForget = Attr.Int(node, "input_forget", 0) != 0;

        System.Span<float> xs = X.Span, ws = W.Span, rs = R.Span;
        System.Span<float> bs = B is null ? default : B.Span;
        System.Span<float> ih = initH is null ? default : initH.Span;
        System.Span<float> ic = initC is null ? default : initC.Span;
        System.Span<float> ps = P is null ? default : P.Span;

        var yAll = new float[S * D * Bt * H];   // Y   [S, D, B, H]
        var yH = new float[D * Bt * H];         // Y_h [D, B, H]
        var yC = new float[D * Bt * H];         // Y_c [D, B, H]

        int wDir = 4 * H * I, rDir = 4 * H * H, pDir = 3 * H;

        for (int d = 0; d < D; d++)
        {
            bool reverse = dir == "reverse" || (dir == "bidirectional" && d == 1);

            var hPrev = new float[Bt * H];
            var cPrev = new float[Bt * H];
            if (initH is not null) for (int i = 0; i < Bt * H; i++) hPrev[i] = ih[d * Bt * H + i];
            if (initC is not null) for (int i = 0; i < Bt * H; i++) cPrev[i] = ic[d * Bt * H + i];

            for (int step = 0; step < S; step++)
            {
                var hCur = new float[Bt * H];
                var cCur = new float[Bt * H];

                for (int b = 0; b < Bt; b++)
                {
                    int hBase = b * H;

                    // Per-batch sequence length. Beyond it the timestep is padding: carry the
                    // previous hidden/cell state through and leave the Y slot at 0.
                    int len = seqLens is null ? S : (int)seqLens[b];
                    if (len < 0) len = 0; else if (len > S) len = S;
                    if (step >= len)
                    {
                        for (int h = 0; h < H; h++)
                        {
                            hCur[hBase + h] = hPrev[hBase + h];
                            cCur[hBase + h] = cPrev[hBase + h];
                        }
                        continue;
                    }

                    // Reverse walks the valid region [0, len) from high to low; padding stays at the tail.
                    int t = reverse ? len - 1 - step : step;
                    int xBase = t * Bt * I + b * I;

                    for (int h = 0; h < H; h++)
                    {
                        float gi = Bias(bs, B, d, H, 0, h);
                        float go = Bias(bs, B, d, H, 1, h);
                        float gf = Bias(bs, B, d, H, 2, h);
                        float gc = Bias(bs, B, d, H, 3, h);

                        int wi = d * wDir + (0 * H + h) * I;
                        int wo = d * wDir + (1 * H + h) * I;
                        int wf = d * wDir + (2 * H + h) * I;
                        int wc = d * wDir + (3 * H + h) * I;
                        for (int k = 0; k < I; k++)
                        {
                            float xk = xs[xBase + k];
                            gi += xk * ws[wi + k];
                            go += xk * ws[wo + k];
                            gf += xk * ws[wf + k];
                            gc += xk * ws[wc + k];
                        }

                        int ri = d * rDir + (0 * H + h) * H;
                        int ro = d * rDir + (1 * H + h) * H;
                        int rf = d * rDir + (2 * H + h) * H;
                        int rc = d * rDir + (3 * H + h) * H;
                        for (int j = 0; j < H; j++)
                        {
                            float hj = hPrev[hBase + j];
                            gi += hj * rs[ri + j];
                            go += hj * rs[ro + j];
                            gf += hj * rs[rf + j];
                            gc += hj * rs[rc + j];
                        }

                        float cP = cPrev[hBase + h];

                        // Peepholes feed the cell state into the i / f gates (P order iof).
                        if (P is not null)
                        {
                            gi += ps[d * pDir + 0 * H + h] * cP;   // Pi (.) C_{t-1}
                            gf += ps[d * pDir + 2 * H + h] * cP;   // Pf (.) C_{t-1}
                        }

                        // Clip is applied to the input of the activations (after peepholes).
                        if (useClip) { gi = Clip(gi, clip); gf = Clip(gf, clip); gc = Clip(gc, clip); }

                        float ft = Sigmoid(gf);
                        // input_forget couples the gates: i_t = 1 - f_t (input-gate weights unused).
                        float it = inputForget ? 1f - ft : Sigmoid(gi);
                        float ctild = MathF.Tanh(gc);
                        float Ct = ft * cP + it * ctild;

                        // The output-gate peephole uses the *current* cell state C_t, then clip + activate.
                        if (P is not null) go += ps[d * pDir + 1 * H + h] * Ct;   // Po (.) C_t
                        if (useClip) go = Clip(go, clip);
                        float ot = Sigmoid(go);
                        float Ht = ot * MathF.Tanh(Ct);

                        cCur[hBase + h] = Ct;
                        hCur[hBase + h] = Ht;
                        yAll[((t * D + d) * Bt + b) * H + h] = Ht;
                    }
                }

                hPrev = hCur;
                cPrev = cCur;
            }

            for (int i = 0; i < Bt * H; i++)
            {
                yH[d * Bt * H + i] = hPrev[i];
                yC[d * Bt * H + i] = cPrev[i];
            }
        }

        if (node.Outputs.Count > 0 && node.Outputs[0].Length > 0)
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(S, D, Bt, H), yAll));
        if (node.Outputs.Count > 1 && node.Outputs[1].Length > 0)
            ctx.Set(node.Outputs[1], new Tensor<float>(new TensorShape(D, Bt, H), yH));
        if (node.Outputs.Count > 2 && node.Outputs[2].Length > 0)
            ctx.Set(node.Outputs[2], new Tensor<float>(new TensorShape(D, Bt, H), yC));
    }

    private static Tensor<float>? Opt(GraphNode node, GraphContext ctx, int idx)
        => node.Inputs.Count > idx && node.Inputs[idx].Length > 0 ? ctx.Get(node.Inputs[idx]) : null;

    /// <summary>Combined input+recurrence bias for gate <paramref name="gate"/> (0=i,1=o,2=f,3=c) at unit h.</summary>
    private static float Bias(System.Span<float> bs, Tensor<float>? b, int d, int H, int gate, int h)
    {
        if (b is null) return 0f;
        int baseB = d * 8 * H;
        return bs[baseB + gate * H + h] + bs[baseB + (4 + gate) * H + h];
    }
}
