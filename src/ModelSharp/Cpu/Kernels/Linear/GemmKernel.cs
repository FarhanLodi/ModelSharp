using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Cpu.Kernels.Llm;
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

        float[] aArr = KernelSimd.Array(a);
        float[] bArr = KernelSimd.Array(b);
        float[] yArr = KernelSimd.Array(y);

        // Present op(A) as row-major [M,K] (K contiguous) and op(B) as row-major [K,Ncol] (Ncol
        // contiguous) — exactly what BlockedGemm wants. A not-transposed input is already [M,K]
        // and a not-transposed B is already [K,Ncol], so only the transposed operands are copied.
        float[] opA; int lda;
        if (!transA) { opA = aArr; lda = aStride1; }            // a is [M,K], row stride = K
        else
        {
            opA = new float[(long)M * K <= int.MaxValue ? M * K : throw new ModelSharpException("Gemm too large")];
            for (int i = 0; i < M; i++)
                for (int kk = 0; kk < K; kk++)
                    opA[i * K + kk] = aArr[kk * aStride1 + i];  // a stored [K,M]
            lda = K;
        }

        float[] opB; int ldb;
        if (!transB) { opB = bArr; ldb = bStride1; }            // b is [K,Ncol], row stride = Ncol
        else
        {
            opB = new float[(long)K * Ncol <= int.MaxValue ? K * Ncol : throw new ModelSharpException("Gemm too large")];
            for (int kk = 0; kk < K; kk++)
                for (int j = 0; j < Ncol; j++)
                    opB[kk * Ncol + j] = bArr[j * bStride1 + kk]; // b stored [Ncol,K]
            ldb = Ncol;
        }

        // Raw Y = op(A) · op(B).
        BlockedGemm.Multiply(opA, 0, lda, opB, 0, ldb, yArr, 0, Ncol, M, Ncol, K);

        // Epilogue: Y = alpha·Y (+ beta·C). Skipped entirely in the common alpha==1, no-C case.
        if (alpha != 1f || hasC)
        {
            float[]? cArr = c is null ? null : KernelSimd.Array(c);
            int[] cdArr = c is null ? System.Array.Empty<int>() : c.Shape.Dimensions.ToArray();
            float alphaL = alpha, betaL = beta;
            bool hasCLocal = c is not null;
            MatMulParallel.For(M, Ncol, i =>
            {
                int row = i * Ncol;
                for (int j = 0; j < Ncol; j++)
                {
                    float val = alphaL * yArr[row + j];
                    if (hasCLocal) val += betaL * BroadcastC(cArr!, cdArr, i, j);
                    yArr[row + j] = val;
                }
            });
        }

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
