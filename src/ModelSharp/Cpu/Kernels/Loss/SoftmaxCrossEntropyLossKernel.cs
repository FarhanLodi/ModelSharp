using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Loss;

/// <summary>
/// ONNX <c>SoftmaxCrossEntropyLoss</c>: applies log-softmax over the class axis (1) of <c>scores</c>
/// [N, C, d1..dk], then computes the negative-log-likelihood against integer <c>labels</c>
/// [N, d1..dk]. Honors the optional class <c>weights</c> [C], <c>ignore_index</c>, and
/// <c>reduction</c> = none / sum / mean (default; weighted mean). Output 0 is the loss
/// (scalar for sum/mean, [N, d1..dk] for none); the optional output 1 is <c>log_prob</c>
/// (the full log-softmax tensor, same shape as <c>scores</c>). Float32 scores; int labels.
/// </summary>
public sealed class SoftmaxCrossEntropyLossKernel : IKernel
{
    public string OpType => "SoftmaxCrossEntropyLoss";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> scores = ctx.Get(node.Inputs[0]);
        long[] labels = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        Tensor<float>? weight = node.Inputs.Count > 2 && node.Inputs[2].Length != 0 ? ctx.Get(node.Inputs[2]) : null;

        string reduction = Attr.Str(node, "reduction", "mean");
        bool hasIgnore = node.Attributes.ContainsKey("ignore_index");
        long ignore = Attr.Int(node, "ignore_index", 0);

        // log-softmax over axis 1
        Tensor<float> logProb = LogSoftmaxAxis1(scores);

        NegativeLogLikelihoodLossKernel.ComputeNll(
            logProb, labels, weight, reduction, hasIgnore, ignore,
            out float[] perElem, out int[] outDims, out float reduced);

        if (reduction == "none")
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(outDims), perElem));
        else
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(Array.Empty<int>()), new[] { reduced }));

        if (node.Outputs.Count > 1 && node.Outputs[1].Length != 0)
            ctx.Set(node.Outputs[1], logProb);
    }

    private static Tensor<float> LogSoftmaxAxis1(Tensor<float> scores)
    {
        ReadOnlySpan<int> sd = scores.Shape.Dimensions;
        int N = sd[0], C = sd[1];
        int inner = 1;
        for (int i = 2; i < sd.Length; i++) inner *= sd[i];

        var outBuf = new float[scores.Span.Length];
        Span<float> xs = scores.Span;
        for (int n = 0; n < N; n++)
        for (int p = 0; p < inner; p++)
        {
            float max = float.NegativeInfinity;
            for (int c = 0; c < C; c++)
                max = MathF.Max(max, xs[(n * C + c) * inner + p]);
            float sum = 0f;
            for (int c = 0; c < C; c++)
                sum += MathF.Exp(xs[(n * C + c) * inner + p] - max);
            float logSum = max + MathF.Log(sum);
            for (int c = 0; c < C; c++)
            {
                int idx = (n * C + c) * inner + p;
                outBuf[idx] = xs[idx] - logSum;
            }
        }
        return new Tensor<float>(scores.Shape, outBuf);
    }
}
