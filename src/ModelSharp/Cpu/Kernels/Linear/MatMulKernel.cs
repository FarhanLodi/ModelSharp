using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Cpu.Kernels.Llm;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Linear;

/// <summary>
/// Matrix multiply with full NumPy semantics: 2-D, stacked (A n-D × B 2-D), and batched
/// n-D × n-D with broadcasting of the leading (batch) dimensions. 1-D operands are promoted
/// per NumPy rules. This is the batched matmul that multi-head attention relies on.
/// </summary>
public sealed class MatMulKernel : IKernel
{
    public string OpType => "MatMul";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> a = ctx.Get(node.Inputs[0]);
        Tensor<float> b = ctx.Get(node.Inputs[1]);

        bool aWas1D = a.Shape.Rank == 1;
        bool bWas1D = b.Shape.Rank == 1;

        int[] adims = aWas1D ? new[] { 1, a.Shape[0] } : a.Shape.Dimensions.ToArray();
        int[] bdims = bWas1D ? new[] { b.Shape[0], 1 } : b.Shape.Dimensions.ToArray();

        int M = adims[adims.Length - 2], K = adims[adims.Length - 1];
        int Kb = bdims[bdims.Length - 2], N = bdims[bdims.Length - 1];
        if (K != Kb) throw new ModelSharpException($"MatMul inner dimensions disagree: {K} vs {Kb}.");

        // Batch dims = everything before the trailing [.., M, K] / [.., K, N].
        int[] aBatch = adims[..^2];
        int[] bBatch = bdims[..^2];
        int[] outBatch = Nd.BroadcastShape(aBatch, bBatch);
        int batchRank = outBatch.Length;
        int[] aBS = Nd.BroadcastStrides(aBatch, batchRank);   // strides in matrix units
        int[] bBS = Nd.BroadcastStrides(bBatch, batchRank);

        int aMat = M * K, bMat = K * N, oMat = M * N;
        int totalBatch = 1;
        foreach (int d in outBatch) totalBatch *= d;

        // Output shape = outBatch + [M, N], with NumPy 1-D promotions removed.
        var outDims = new List<int>(outBatch);
        if (!aWas1D) outDims.Add(M);
        if (!bWas1D) outDims.Add(N);
        var y = new Tensor<float>(new TensorShape(outDims.ToArray()));

        // Map each batch index to its A-matrix and B-matrix offsets (in matrix units) up front,
        // honoring NumPy batch broadcasting. Both A[M,K] and B[K,N] are already row-major and
        // contiguous along their natural axes, so the GEMM reads them in place (no transpose copy).
        var aOffOf = new int[totalBatch];
        var bOffOf = new int[totalBatch];
        {
            var coord = new int[batchRank];
            int aOff = 0, bOff = 0;
            for (int bi = 0; bi < totalBatch; bi++)
            {
                aOffOf[bi] = aOff;
                bOffOf[bi] = bOff;
                for (int ax = batchRank - 1; ax >= 0; ax--)
                {
                    coord[ax]++;
                    aOff += aBS[ax];
                    bOff += bBS[ax];
                    if (coord[ax] < outBatch[ax]) break;
                    coord[ax] = 0;
                    aOff -= aBS[ax] * outBatch[ax];
                    bOff -= bBS[ax] * outBatch[ax];
                }
            }
        }

        float[] aArr = KernelSimd.Array(a);
        float[] bArr = KernelSimd.Array(b);
        float[] yArr = KernelSimd.Array(y);

        // Parallelize over the flattened (batch × MR-row-tile × NR-col-tile) grid of disjoint
        // output tiles; each tile streams A[mr,K] / B[K,ncols] and writes Y[mr,ncols].
        int nr = BlockedGemm.NR;
        int mTiles = (M + BlockedGemm.MR - 1) / BlockedGemm.MR;
        int nTiles = (N + nr - 1) / nr;
        long tilesPerBatch = (long)mTiles * nTiles;
        long totalTiles = tilesPerBatch * totalBatch;
        long innerCost = (long)BlockedGemm.MR * nr * K;

        MatMulParallel.For(checked((int)totalTiles), innerCost, t =>
        {
            int bi = (int)(t / tilesPerBatch);
            int local = (int)(t - bi * tilesPerBatch);
            int mt = local / nTiles;
            int nt = local - mt * nTiles;
            int m0 = mt * BlockedGemm.MR;
            int n0 = nt * nr;
            BlockedGemm.Tile(
                aArr, aOffOf[bi] * aMat, /*lda*/ K,
                bArr, bOffOf[bi] * bMat, /*ldb*/ N,
                yArr, bi * oMat, /*ldc*/ N,
                m0, n0, Math.Min(BlockedGemm.MR, M - m0), Math.Min(nr, N - n0), K);
        });

        ctx.Set(node.Outputs[0], y);
    }
}
