using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Loss;

/// <summary>
/// ONNX <c>NegativeLogLikelihoodLoss</c>: given per-class log-probabilities <c>input</c>
/// [N, C, d1..dk] and integer <c>target</c> [N, d1..dk], produces the loss
/// <c>loss[i] = -weight[target[i]] · input[i, target[i]]</c> (an <c>ignore_index</c> target yields
/// 0). The optional <c>weight</c> [C] re-weights classes. The <c>reduction</c> attribute selects
/// <c>none</c> (elementwise loss tensor), <c>sum</c>, or <c>mean</c> (default; weighted mean —
/// normalized by the sum of the applied weights). Float32 input; int target.
/// </summary>
public sealed class NegativeLogLikelihoodLossKernel : IKernel
{
    public string OpType => "NegativeLogLikelihoodLoss";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> input = ctx.Get(node.Inputs[0]);
        long[] target = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        Tensor<float>? weight = node.Inputs.Count > 2 && node.Inputs[2].Length != 0 ? ctx.Get(node.Inputs[2]) : null;

        string reduction = Attr.Str(node, "reduction", "mean");
        bool hasIgnore = node.Attributes.ContainsKey("ignore_index");
        long ignore = Attr.Int(node, "ignore_index", 0);

        ComputeNll(input, target, weight, reduction, hasIgnore, ignore,
            out float[] perElem, out int[] outDims, out float reduced);

        if (reduction == "none")
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(outDims), perElem));
        else
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(Array.Empty<int>()), new[] { reduced }));
    }

    /// <summary>
    /// Core NLL computation shared with <c>SoftmaxCrossEntropyLoss</c>. <paramref name="input"/> holds
    /// per-class scores already in log-prob form. Fills the per-(N,d…) loss <paramref name="perElem"/>
    /// (shape <paramref name="outDims"/>) and the scalar <paramref name="reduced"/>.
    /// </summary>
    internal static void ComputeNll(
        Tensor<float> input, long[] target, Tensor<float>? weight, string reduction,
        bool hasIgnore, long ignore,
        out float[] perElem, out int[] outDims, out float reduced)
    {
        ReadOnlySpan<int> id = input.Shape.Dimensions;
        int N = id[0], C = id[1];
        int inner = 1;
        for (int i = 2; i < id.Length; i++) inner *= id[i];

        outDims = new int[id.Length - 1];
        outDims[0] = N;
        for (int i = 2; i < id.Length; i++) outDims[i - 1] = id[i];

        Span<float> xs = input.Span;
        Span<float> ws = weight is null ? default : weight.Span;
        perElem = new float[N * inner];

        float lossSum = 0f, weightSum = 0f;
        for (int n = 0; n < N; n++)
        for (int p = 0; p < inner; p++)
        {
            int flat = n * inner + p;
            long t = target[flat];
            if (hasIgnore && t == ignore) { perElem[flat] = 0f; continue; }
            if (t < 0 || t >= C) throw new ModelSharpException($"NegativeLogLikelihoodLoss: target {t} out of range [0,{C}).");

            float w = weight is not null ? ws[(int)t] : 1f;
            // input index for class t at spatial position p: [n, t, p]
            float val = xs[(n * C + (int)t) * inner + p];
            float l = -w * val;
            perElem[flat] = l;
            lossSum += l;
            weightSum += w;
        }

        reduced = reduction switch
        {
            "sum" => lossSum,
            "mean" => weightSum != 0f ? lossSum / weightSum : 0f,
            "none" => 0f,
            _ => throw new ModelSharpException($"NegativeLogLikelihoodLoss: unsupported reduction '{reduction}'."),
        };
    }
}
