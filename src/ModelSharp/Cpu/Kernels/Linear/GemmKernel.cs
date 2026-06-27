using System;
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

        // Build a row-major [Ncol, K] copy of op(B) once, so every output row dots against a
        // contiguous weight row (SIMD-friendly) regardless of the transA/transB flags. Likewise
        // expose op(A) row i as a contiguous length-K vector. The transpose cost is O(K·Ncol),
        // amortised across all M rows.
        var bT = new float[(long)Ncol * K <= int.MaxValue ? Ncol * K : throw new ModelSharpException("Gemm too large")];
        {
            System.Span<float> sb = b.Span;
            for (int j = 0; j < Ncol; j++)
            for (int kk = 0; kk < K; kk++)
                bT[j * K + kk] = transB ? sb[j * bStride1 + kk] : sb[kk * bStride1 + j];
        }

        // op(A): when not transposed each row is already contiguous; when transposed we gather.
        float[]? aT = null;
        if (transA)
        {
            aT = new float[(long)M * K <= int.MaxValue ? M * K : throw new ModelSharpException("Gemm too large")];
            System.Span<float> sa = a.Span;
            for (int i = 0; i < M; i++)
            for (int kk = 0; kk < K; kk++)
                aT[i * K + kk] = sa[kk * aStride1 + i];
        }

        System.Memory<float> ma = a.Buffer, my = y.Buffer;
        System.Memory<float> mc = c is null ? default : c.Buffer;
        int[] cdArr = c is null ? System.Array.Empty<int>() : c.Shape.Dimensions.ToArray();
        float[]? aTLocal = aT;
        int aStride1Local = aStride1;
        bool hasCLocal = c is not null;

        MatMulParallel.For(M, (long)Ncol * K, i =>
        {
            System.ReadOnlySpan<float> sa = ma.Span;
            System.Span<float> sy = my.Span;
            System.ReadOnlySpan<float> sc = mc.Span;

            // Contiguous length-K view of op(A) row i.
            System.ReadOnlySpan<float> aRowSpan = aTLocal is not null
                ? aTLocal.AsSpan(i * K, K)
                : sa.Slice(i * aStride1Local, K);

            for (int j = 0; j < Ncol; j++)
            {
                float sum = MatMulParallel.Dot(aRowSpan, 0, bT.AsSpan(j * K, K), K);
                float val = alpha * sum;
                if (hasCLocal) val += beta * BroadcastC(sc, cdArr, i, j);
                sy[i * Ncol + j] = val;
            }
        });

        ctx.Set(node.Outputs[0], y);
    }

    private static float BroadcastC(System.ReadOnlySpan<float> data, System.ReadOnlySpan<int> dims, int i, int j)
    {
        if (dims.Length == 0) return data[0];
        if (dims.Length == 1) return data[dims[0] == 1 ? 0 : j];
        int d0 = dims[0], d1 = dims[1];
        int ii = d0 == 1 ? 0 : i;
        int jj = d1 == 1 ? 0 : j;
        return data[ii * d1 + jj];
    }
}
