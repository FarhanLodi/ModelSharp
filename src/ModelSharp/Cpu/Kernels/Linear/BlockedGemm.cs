using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace ModelSharp.Cpu.Kernels.Linear;

/// <summary>
/// Register-tiled, multi-threaded float32 GEMM that computes the RAW product
/// <c>Y[M,N] = A[M,K] · B[K,N]</c> over flat row-major buffers with explicit element offsets and
/// row strides (lda/ldb/ldc), so callers
/// can pass sub-views (a batch slice, a weight channel block, an im2col panel) with zero copying.
/// <c>lda</c>/<c>ldb</c>/<c>ldc</c> are the per-matrix row strides.
///
/// <para><b>Why it is fast (the whole point — memory reuse).</b> The per-element dot path streams a
/// fresh A row and a fresh B row to produce ONE output scalar, so every loaded float is used once
/// and the kernel is bandwidth-bound. Here an <c>MR × NR</c> tile of the output is held in
/// <see cref="Vector{T}"/> registers across the entire K sweep: each k-step loads NRV B-vectors
/// (contiguous in N) and broadcasts MR A scalars, then issues MR·NRV vector FMAs. Each loaded B
/// vector feeds MR accumulators and each broadcast A scalar feeds NRV accumulators, raising the
/// reuse of every load from ~1 to ~MR FMAs. The accumulators never leave registers until the tile
/// is stored, so unlike a "dot-tile" there is no per-element horizontal reduction.</para>
///
/// <para>Pure-managed and AOT-safe: <see cref="Vector{T}"/> is variable width and lowers to
/// SSE/AVX/AVX-512 on x64 and NEON on ARM from a single source. MR=6, NRV=2 keeps 12 accumulator
/// vectors + 2 B vectors + 1 broadcast ≈ 15 SIMD registers, fitting AVX2's 16 (ARM/AVX-512 have
/// 32). No <c>System.Runtime.Intrinsics</c> platform code is used.</para>
///
/// <para>Scaling (alpha), accumulation (beta·C) and bias stay OUT of the kernel: callers apply
/// them in a cheap O(M·N) vectorized epilogue, keeping the kernel single-purpose and parity-stable.
/// The SIMD accumulation order differs from a left-to-right scalar sum by a few ULP, well inside
/// the engine's ~1e-2 ORT tolerance.</para>
/// </summary>
internal static class BlockedGemm
{
    /// <summary>Rows of the output held together in one register tile.</summary>
    public const int MR = 6;

    /// <summary>Number of <see cref="Vector{T}"/>-wide columns held together in one register tile.</summary>
    private const int NRV = 2;

    /// <summary>Columns of the output held together in one register tile (= NRV · vector width).</summary>
    public static int NR => NRV * Vector<float>.Count;

    /// <summary>
    /// Y[M,N] = A·B, threaded over the 2-D grid of disjoint MR×NR output tiles (so parallelism is
    /// available whether M or N is the large axis). Overwrites <paramref name="y"/>.
    /// </summary>
    public static void Multiply(
        float[] a, int aOff, int lda,
        float[] b, int bOff, int ldb,
        float[] y, int yOff, int ldc,
        int M, int N, int K)
    {
        if (M <= 0 || N <= 0) return;

        // Optional native AVX-512 fast path (libms_kernels.so). Falls through to the managed
        // kernel below when unavailable/disabled, strided, or too small. Used by the conv 1×1 /
        // im2col GEMM and any other full-matrix caller.
        if (Native.NativeGemm.TryMultiply(a, aOff, lda, b, bOff, ldb, y, yOff, ldc, M, N, K))
            return;

        int nr = NR;
        int nTiles = (N + nr - 1) / nr;
        int mTiles = (M + MR - 1) / MR;
        long total = (long)mTiles * nTiles;
        long innerCost = (long)MR * nr * K;

        MatMulParallel.For(checked((int)total), innerCost, t =>
        {
            int mt = t / nTiles;
            int nt = t - mt * nTiles;
            int m0 = mt * MR;
            int n0 = nt * nr;
            Tile(a, aOff, lda, b, bOff, ldb, y, yOff, ldc,
                 m0, n0, Math.Min(MR, M - m0), Math.Min(nr, N - n0), K);
        });
    }

    /// <summary>
    /// Computes one output tile <c>Y[m0..m0+mr, n0..n0+ncols] = A · B</c>. <paramref name="mr"/> ≤ MR
    /// and <paramref name="ncols"/> ≤ NR. Full interior tiles take the register-resident fast path;
    /// M/N remainders take a correct vectorized edge path. Safe to call from a parallel loop as long
    /// as tiles owning distinct (m0,n0) regions are dispatched (they write disjoint output).
    /// </summary>
    public static void Tile(
        float[] a, int aOff, int lda,
        float[] b, int bOff, int ldb,
        float[] y, int yOff, int ldc,
        int m0, int n0, int mr, int ncols, int K)
    {
        if (mr == MR && ncols == NR)
            FullTile(a, aOff + m0 * lda, lda, b, bOff + n0, ldb, y, yOff + m0 * ldc + n0, ldc, K);
        else
            EdgeTile(a, aOff + m0 * lda, lda, b, bOff + n0, ldb, y, yOff + m0 * ldc + n0, ldc, mr, ncols, K);
    }

    /// <summary>Register-resident 6 × (NRV·W) micro-kernel. 12 accumulators stay in registers across
    /// the whole K loop; each B vector feeds 6 FMAs and each broadcast A scalar feeds 2.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FullTile(
        float[] a, int a0Row, int lda,
        float[] b, int b0, int ldb,
        float[] y, int y0, int ldc, int K)
    {
        int w = Vector<float>.Count;
        int a1Row = a0Row + lda, a2Row = a0Row + 2 * lda, a3Row = a0Row + 3 * lda,
            a4Row = a0Row + 4 * lda, a5Row = a0Row + 5 * lda;

        Vector<float> c00 = default, c01 = default, c10 = default, c11 = default,
                      c20 = default, c21 = default, c30 = default, c31 = default,
                      c40 = default, c41 = default, c50 = default, c51 = default;

        int bo = b0;
        for (int k = 0; k < K; k++)
        {
            var b0v = new Vector<float>(b, bo);
            var b1v = new Vector<float>(b, bo + w);
            Vector<float> va;
            va = new Vector<float>(a[a0Row + k]); c00 += va * b0v; c01 += va * b1v;
            va = new Vector<float>(a[a1Row + k]); c10 += va * b0v; c11 += va * b1v;
            va = new Vector<float>(a[a2Row + k]); c20 += va * b0v; c21 += va * b1v;
            va = new Vector<float>(a[a3Row + k]); c30 += va * b0v; c31 += va * b1v;
            va = new Vector<float>(a[a4Row + k]); c40 += va * b0v; c41 += va * b1v;
            va = new Vector<float>(a[a5Row + k]); c50 += va * b0v; c51 += va * b1v;
            bo += ldb;
        }

        int r0 = y0, r1 = y0 + ldc, r2 = y0 + 2 * ldc, r3 = y0 + 3 * ldc, r4 = y0 + 4 * ldc, r5 = y0 + 5 * ldc;
        c00.CopyTo(y, r0); c01.CopyTo(y, r0 + w);
        c10.CopyTo(y, r1); c11.CopyTo(y, r1 + w);
        c20.CopyTo(y, r2); c21.CopyTo(y, r2 + w);
        c30.CopyTo(y, r3); c31.CopyTo(y, r3 + w);
        c40.CopyTo(y, r4); c41.CopyTo(y, r4 + w);
        c50.CopyTo(y, r5); c51.CopyTo(y, r5 + w);
    }

    /// <summary>Arbitrary mr×ncols edge tile (M/N remainders, GEMV, small shapes). Accumulates into a
    /// tiny stack buffer, still vectorizing the B loads over the contiguous N axis.</summary>
    private static void EdgeTile(
        float[] a, int a0Row, int lda,
        float[] b, int b0, int ldb,
        float[] y, int y0, int ldc, int mr, int ncols, int K)
    {
        int w = Vector<float>.Count;
        int nr = NR;
        Span<float> acc = stackalloc float[MR * (NRV * 16)]; // upper bound (max W = 16, AVX-512)
        acc = acc.Slice(0, mr * nr);
        acc.Clear();

        int bo = b0;
        for (int k = 0; k < K; k++)
        {
            int arow = a0Row + k;
            for (int i = 0; i < mr; i++)
            {
                float av = a[arow + i * lda];
                if (av == 0f) continue;
                var avv = new Vector<float>(av);
                int dst = i * nr;
                int n = 0;
                int last = ncols - w;
                for (; n <= last; n += w)
                {
                    var cur = new Vector<float>(acc.Slice(dst + n, w));
                    (cur + avv * new Vector<float>(b, bo + n)).CopyTo(acc.Slice(dst + n, w));
                }
                for (; n < ncols; n++) acc[dst + n] += av * b[bo + n];
            }
            bo += ldb;
        }

        for (int i = 0; i < mr; i++)
        {
            int yr = y0 + i * ldc;
            int src = i * nr;
            for (int n = 0; n < ncols; n++) y[yr + n] = acc[src + n];
        }
    }
}
