using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>Global average pool (NCHW): mean over H×W → [N, C, 1, 1].</summary>
public sealed class GlobalAveragePoolKernel : IKernel
{
    public string OpType => "GlobalAveragePool";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];

        var y = new Tensor<float>(new TensorShape(N, C, 1, 1));
        System.Span<float> xs = x.Span, ys = y.Span;
        int xC = H * W, xN = C * xC;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            float s = 0f;
            int b = n * xN + c * xC;
            for (int i = 0; i < xC; i++) s += xs[b + i];
            ys[n * C + c] = s / xC;
        }

        ctx.Set(node.Outputs[0], y);
    }
}
