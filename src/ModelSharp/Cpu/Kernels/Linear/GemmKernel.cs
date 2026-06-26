using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Linear;

/// <summary>General matrix multiply: Y = α·op(A)·op(B) + β·C, with optional transpose and broadcastable C.</summary>
public sealed class GemmKernel : IKernel
{
    public string OpType => "Gemm";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> a = ctx.Get(node.Inputs[0]);
        Tensor<float> b = ctx.Get(node.Inputs[1]);
        bool hasC = node.Inputs.Count > 2 && node.Inputs[2].Length > 0;
        Tensor<float>? c = hasC ? ctx.Get(node.Inputs[2]) : null;

        float alpha = Attr.Float(node, "alpha", 1f);
        float beta = Attr.Float(node, "beta", 1f);
        bool transA = Attr.Int(node, "transA", 0) != 0;
        bool transB = Attr.Int(node, "transB", 0) != 0;

        System.ReadOnlySpan<int> ad = a.Shape.Dimensions;
        System.ReadOnlySpan<int> bd = b.Shape.Dimensions;
        int M = transA ? ad[1] : ad[0];
        int K = transA ? ad[0] : ad[1];
        int Kb = transB ? bd[1] : bd[0];
        int Ncol = transB ? bd[0] : bd[1];
        if (K != Kb) throw new ModelSharpException($"Gemm inner dimensions disagree: {K} vs {Kb}.");

        int aStride1 = ad[1], bStride1 = bd[1];
        var y = new Tensor<float>(new TensorShape(M, Ncol));
        System.Span<float> sa = a.Span, sb = b.Span, sy = y.Span;
        System.Span<float> sc = c is null ? default : c.Span;
        System.ReadOnlySpan<int> cd = c is null ? default : c.Shape.Dimensions;

        for (int i = 0; i < M; i++)
        for (int j = 0; j < Ncol; j++)
        {
            float sum = 0f;
            for (int kk = 0; kk < K; kk++)
            {
                float av = transA ? sa[kk * aStride1 + i] : sa[i * aStride1 + kk];
                float bv = transB ? sb[j * bStride1 + kk] : sb[kk * bStride1 + j];
                sum += av * bv;
            }
            float val = alpha * sum;
            if (c is not null) val += beta * BroadcastC(sc, cd, i, j);
            sy[i * Ncol + j] = val;
        }

        ctx.Set(node.Outputs[0], y);
    }

    private static float BroadcastC(System.Span<float> data, System.ReadOnlySpan<int> dims, int i, int j)
    {
        if (dims.Length == 0) return data[0];
        if (dims.Length == 1) return data[dims[0] == 1 ? 0 : j];
        int d0 = dims[0], d1 = dims[1];
        int ii = d0 == 1 ? 0 : i;
        int jj = d1 == 1 ? 0 : j;
        return data[ii * d1 + jj];
    }
}
