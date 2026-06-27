using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
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

        // Capture the backing buffers (Memory<T>) so the parallel lambdas can re-acquire spans;
        // Span<T> is a ref struct and cannot be hoisted into a closure.
        System.Memory<float> ma = a.Buffer, my = y.Buffer;

        // Pre-transpose each distinct B batch matrix from [K,N] (column-strided) into row-major
        // [N,K] so the inner dot is fully contiguous and SIMD-friendly. The transposed cache is
        // indexed by the broadcast B-batch offset (bOff), so broadcast/shared B matrices are only
        // transposed once.
        var bTransposed = new float[totalBatch][];
        {
            var coordT = new int[batchRank];
            System.Span<float> sbAll = b.Span;
            int bOffT = 0;
            var seen = new System.Collections.Generic.Dictionary<int, float[]>();
            for (int bi = 0; bi < totalBatch; bi++)
            {
                if (!seen.TryGetValue(bOffT, out float[]? bt))
                {
                    bt = new float[bMat];
                    int bBase = bOffT * bMat;
                    for (int kk = 0; kk < K; kk++)
                    {
                        int src = bBase + kk * N;
                        for (int n = 0; n < N; n++) bt[n * K + kk] = sbAll[src + n];
                    }
                    seen[bOffT] = bt;
                }
                bTransposed[bi] = bt;

                for (int ax = batchRank - 1; ax >= 0; ax--)
                {
                    coordT[ax]++;
                    bOffT += bBS[ax];
                    if (coordT[ax] < outBatch[ax]) break;
                    coordT[ax] = 0;
                    bOffT -= bBS[ax] * outBatch[ax];
                }
            }
        }

        // Map each batch index to its A-matrix offset (in matrix units) up front so the per-row
        // work is embarrassingly parallel over flattened (batch × M) output rows.
        var aOffOf = new int[totalBatch];
        {
            var coordA = new int[batchRank];
            int aOff = 0;
            for (int bi = 0; bi < totalBatch; bi++)
            {
                aOffOf[bi] = aOff;
                for (int ax = batchRank - 1; ax >= 0; ax--)
                {
                    coordA[ax]++;
                    aOff += aBS[ax];
                    if (coordA[ax] < outBatch[ax]) break;
                    coordA[ax] = 0;
                    aOff -= aBS[ax] * outBatch[ax];
                }
            }
        }

        long totalRows = (long)totalBatch * M;
        MatMulParallel.For(checked((int)totalRows), (long)K * N, row =>
        {
            int bi = row / M;
            int m = row % M;
            System.ReadOnlySpan<float> sa = ma.Span;
            System.Span<float> sy = my.Span;
            float[] bt = bTransposed[bi];
            int aRow = aOffOf[bi] * aMat + m * K;
            int oRow = bi * oMat + m * N;
            for (int n = 0; n < N; n++)
                sy[oRow + n] = MatMulParallel.Dot(sa, aRow, bt.AsSpan(n * K, K), K);
        });

        ctx.Set(node.Outputs[0], y);
    }
}
