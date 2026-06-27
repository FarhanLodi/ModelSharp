using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Rnn;

/// <summary>
/// Vanilla Elman recurrent unit (ONNX <c>RNN</c>):
/// <c>Ht = f(Xt·Wᵀ + Ht-1·Rᵀ + Wb + Rb)</c>. Supports forward / reverse / bidirectional,
/// optional bias <c>B</c> [D,2H], optional initial state <c>initial_h</c> [D,B,H], the optional
/// <c>sequence_lens</c> input, the <c>clip</c> attribute, and a single configurable activation via
/// the <c>activations</c> attribute (default Tanh; Relu / Sigmoid also handled). Emits Y [S,D,B,H]
/// and/or Y_h [D,B,H] (either may be an omitted "" output). Float32.
/// </summary>
public sealed class RnnKernel : IKernel
{
    public string OpType => "RNN";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> X = ctx.Get(node.Inputs[0]);   // [S, B, I]
        Tensor<float> W = ctx.Get(node.Inputs[1]);   // [D, H, I]
        Tensor<float> R = ctx.Get(node.Inputs[2]);   // [D, H, H]
        Tensor<float>? B = Opt(node, ctx, 3);         // [D, 2H]  (Wb, Rb)
        Tensor<float>? initH = Opt(node, ctx, 5);     // [D, B, H]

        long[]? seqLens = node.Inputs.Count > 4 && node.Inputs[4].Length > 0
            ? TensorInts.Read(ctx.GetTensor(node.Inputs[4]))   // [B]
            : null;

        ReadOnlySpan<int> xd = X.Shape.Dimensions;
        int S = xd[0], Bt = xd[1], I = xd[2];
        ReadOnlySpan<int> wd = W.Shape.Dimensions;
        int D = wd[0], H = wd[1];

        string dir = Attr.Str(node, "direction", "forward");
        float clip = Attr.Float(node, "clip", 0f);
        bool useClip = clip > 0f;
        // ONNX provides per-direction activation names; we honor the first (and its reverse-dir
        // counterpart for bidirectional). Default Tanh.
        string actName = FirstActivation(node);

        Span<float> xs = X.Span, ws = W.Span, rs = R.Span;
        Span<float> bs = B is null ? default : B.Span;
        Span<float> ih = initH is null ? default : initH.Span;

        var yAll = new float[S * D * Bt * H];   // Y   [S, D, B, H]
        var yH = new float[D * Bt * H];         // Y_h [D, B, H]

        int wDir = H * I, rDir = H * H, bDir = 2 * H;

        for (int d = 0; d < D; d++)
        {
            bool reverse = dir == "reverse" || (dir == "bidirectional" && d == 1);
            for (int b = 0; b < Bt; b++)
            {
                var h = new float[H];
                if (initH is not null)
                    for (int k = 0; k < H; k++) h[k] = ih[(d * Bt + b) * H + k];

                long len = seqLens is not null ? seqLens[b] : S;
                for (int ti = 0; ti < S; ti++)
                {
                    int t = reverse ? S - 1 - ti : ti;
                    if (t >= len)
                    {
                        // Past this batch's length: state frozen, Y entry left zero.
                        continue;
                    }

                    var hNew = new float[H];
                    for (int o = 0; o < H; o++)
                    {
                        float acc = 0f;
                        int wBase = d * wDir + o * I;
                        for (int i = 0; i < I; i++)
                            acc += xs[(t * Bt + b) * I + i] * ws[wBase + i];
                        int rBase = d * rDir + o * H;
                        for (int k = 0; k < H; k++)
                            acc += h[k] * rs[rBase + k];
                        if (B is not null)
                            acc += bs[d * bDir + o] + bs[d * bDir + H + o];
                        if (useClip) acc = acc > clip ? clip : (acc < -clip ? -clip : acc);
                        hNew[o] = Activate(actName, acc);
                    }
                    h = hNew;
                    for (int o = 0; o < H; o++)
                        yAll[((t * D + d) * Bt + b) * H + o] = h[o];
                }
                for (int o = 0; o < H; o++) yH[(d * Bt + b) * H + o] = h[o];
            }
        }

        if (node.Outputs.Count > 0 && node.Outputs[0].Length != 0)
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(S, D, Bt, H), yAll));
        if (node.Outputs.Count > 1 && node.Outputs[1].Length != 0)
            ctx.Set(node.Outputs[1], new Tensor<float>(new TensorShape(D, Bt, H), yH));
    }

    private static Tensor<float>? Opt(GraphNode node, GraphContext ctx, int idx)
        => node.Inputs.Count > idx && node.Inputs[idx].Length != 0 ? ctx.Get(node.Inputs[idx]) : null;

    private static string FirstActivation(GraphNode node)
    {
        if (node.Attributes.TryGetValue("activations", out object? v))
        {
            string? first = v switch
            {
                string[] sa when sa.Length > 0 => sa[0],
                string s => s,
                _ => null,
            };
            if (first is not null) return first;
        }
        return "Tanh";
    }

    private static float Activate(string name, float x) => name switch
    {
        "Relu" => x > 0f ? x : 0f,
        "Sigmoid" => 1f / (1f + MathF.Exp(-x)),
        "Tanh" => MathF.Tanh(x),
        _ => throw new ModelSharpException($"RNN: activation '{name}' is not supported."),
    };
}
