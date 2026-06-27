using System;
using ILGPU;

namespace ModelSharp.Gpu;

/// <summary>
/// Device-side kernels for the non-elementwise GPU ops (broadcasting binary ops, matmul, conv2d).
/// These are plain static methods that ILGPU JIT-compiles to PTX / OpenCL / CPU. They are kept
/// separate from the engine plumbing in <see cref="IlgpuEngine"/> so the host-side orchestration
/// (stride precomputation, buffer management) stays readable. All data is float32 and row-major.
/// </summary>
internal static class GpuKernels
{
    /// <summary>Op selector for <see cref="BroadcastBinaryK"/>: 0=Add, 1=Sub, 2=Mul, 3=Div.</summary>
    internal const int OpAdd = 0, OpSub = 1, OpMul = 2, OpDiv = 3;

    // --- Extra native unary float ops (one thread per element; mirror the CPU UnaryKernel/Activation kernels) ---

    /// <summary>Sign: -1, 0, or +1 (NaN preserved). Mirrors the CPU <c>SignKernel</c>.</summary>
    internal static void SignK(Index1D i, ArrayView<float> a, ArrayView<float> y)
        => y[i] = a[i] > 0f ? 1f : (a[i] < 0f ? -1f : a[i]);

    /// <summary>Floor (round toward −∞). Mirrors the CPU <c>FloorKernel</c>.</summary>
    internal static void FloorK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = MathF.Floor(a[i]);

    /// <summary>Ceiling (round toward +∞). Mirrors the CPU <c>CeilKernel</c>.</summary>
    internal static void CeilK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = MathF.Ceiling(a[i]);

    /// <summary>Round-half-to-even (banker's rounding). Mirrors the CPU <c>RoundKernel</c>.</summary>
    internal static void RoundK(Index1D i, ArrayView<float> a, ArrayView<float> y)
        => y[i] = MathF.Round(a[i], MidpointRounding.ToEven);

    /// <summary>Softplus: <c>log(1 + exp(x))</c>. Mirrors the CPU <c>SoftplusKernel</c>.</summary>
    internal static void SoftplusK(Index1D i, ArrayView<float> a, ArrayView<float> y)
        => y[i] = MathF.Log(1f + MathF.Exp(a[i]));

    /// <summary>Mish: <c>x · tanh(softplus(x))</c>. Mirrors the CPU <c>MishKernel</c>.</summary>
    internal static void MishK(Index1D i, ArrayView<float> a, ArrayView<float> y)
        => y[i] = a[i] * MathF.Tanh(MathF.Log(1f + MathF.Exp(a[i])));

    /// <summary>HardSwish: <c>x · relu6(x+3) / 6</c>. Mirrors the CPU <c>HardSwishKernel</c>.</summary>
    internal static void HardSwishK(Index1D i, ArrayView<float> a, ArrayView<float> y)
    {
        float v = a[i] + 3f;
        v = v < 0f ? 0f : (v > 6f ? 6f : v);
        y[i] = a[i] * v / 6f;
    }

    /// <summary>HardSigmoid: <c>clip(α·x + β, 0, 1)</c>. Mirrors the CPU <c>HardSigmoidKernel</c>.</summary>
    internal static void HardSigmoidK(Index1D i, ArrayView<float> a, ArrayView<float> y, float alpha, float beta)
    {
        float v = alpha * a[i] + beta;
        y[i] = v < 0f ? 0f : (v > 1f ? 1f : v);
    }

    /// <summary>ELU: <c>x</c> if x ≥ 0 else <c>α·(exp(x) − 1)</c>. Mirrors the CPU <c>EluKernel</c>.</summary>
    internal static void EluK(Index1D i, ArrayView<float> a, ArrayView<float> y, float alpha)
        => y[i] = a[i] >= 0f ? a[i] : alpha * (MathF.Exp(a[i]) - 1f);

    /// <summary>SELU: <c>γ·x</c> if x &gt; 0 else <c>γ·α·(exp(x) − 1)</c>. Mirrors the CPU <c>SeluKernel</c>.</summary>
    internal static void SeluK(Index1D i, ArrayView<float> a, ArrayView<float> y, float alpha, float gamma)
        => y[i] = a[i] > 0f ? gamma * a[i] : gamma * (alpha * (MathF.Exp(a[i]) - 1f));

    /// <summary>Clip: clamp each element to <c>[lo, hi]</c>. Mirrors the CPU <c>ClipKernel</c>.</summary>
    internal static void ClipK(Index1D i, ArrayView<float> a, ArrayView<float> y, float lo, float hi)
    {
        float v = a[i];
        y[i] = v < lo ? lo : (v > hi ? hi : v);
    }

    /// <summary>Reduce op selector for <see cref="ReduceOpK"/>: 0=Max, 1=Min, 2=Prod.</summary>
    internal const int RedMax = 0, RedMin = 1, RedProd = 2;

    /// <summary>
    /// Axis reduction by Max / Min / Prod (ONNX <c>ReduceMax</c>/<c>ReduceMin</c>/<c>ReduceProd</c>), selected by
    /// <paramref name="op"/>. Same indexing/fold scheme as <see cref="ReduceK"/> (which does Sum/Mean): one thread
    /// per output element, <paramref name="outBase"/> giving the all-reduced-coords-zero input offset and the
    /// reduced coordinates enumerated row-major via <paramref name="redOutStrides"/>/<paramref name="redStrides"/>.
    /// The accumulator is seeded with the op identity (Prod→1, Max→−∞, Min→+∞) and folded in the same order as the
    /// CPU reduce kernels, so float results match.
    /// </summary>
    internal static void ReduceOpK(
        Index1D idx,
        ArrayView<float> x,
        ArrayView<float> y,
        ArrayView<int> outBase,
        ArrayView<int> redOutStrides,
        ArrayView<int> redStrides,
        int numRed,
        int redCount,
        int op)
    {
        int o = idx.X;
        int baseOff = outBase[o];
        float acc = op == RedProd ? 1f : (op == RedMax ? float.NegativeInfinity : float.PositiveInfinity);
        for (int r = 0; r < redCount; r++)
        {
            int rem = r;
            int off = baseOff;
            for (int k = 0; k < numRed; k++)
            {
                int st = redOutStrides[k];
                int c = rem / st;
                rem -= c * st;
                off += c * redStrides[k];
            }
            float v = x[off];
            if (op == RedMax) acc = v > acc ? v : acc;
            else if (op == RedMin) acc = v < acc ? v : acc;
            else acc *= v;
        }
        y[o] = acc;
    }

    /// <summary>Variadic-fold op selector for <see cref="VariadicFoldK"/>: 0=Min, 1=Max, 2=Sum/Mean.</summary>
    internal const int VarMin = 0, VarMax = 1, VarSum = 2;

    /// <summary>
    /// Folds ONE input of a variadic elementwise op (ONNX <c>Min</c>/<c>Max</c>/<c>Sum</c>/<c>Mean</c>) into a
    /// running accumulator buffer, with NumPy-style broadcasting of the input over the output shape. One thread per
    /// output element; the host calls this once per input (seeding <paramref name="acc"/> with the op's identity
    /// first), matching the CPU <c>VariadicElementwiseKernel</c> fold order. <paramref name="op"/> selects the
    /// combine; Mean's divide-by-count is applied by a separate scalar pass on the host side.
    /// </summary>
    internal static void VariadicFoldK(
        Index1D i,
        ArrayView<float> acc,
        ArrayView<float> input,
        ArrayView<int> outStrides,
        ArrayView<int> inStrides,
        int rank,
        int op)
    {
        int rem = i.X;
        int off = 0;
        for (int ax = 0; ax < rank; ax++)
        {
            int st = outStrides[ax];
            int c = rem / st;
            rem -= c * st;
            off += c * inStrides[ax];
        }
        float a = acc[i], x = input[off];
        if (op == VarMin) acc[i] = x < a ? x : a;
        else if (op == VarMax) acc[i] = x > a ? x : a;
        else acc[i] = a + x;
    }

    /// <summary>Equal-shape elementwise divide (the broadcasting fast-path twin of Add/Sub/Mul).</summary>
    internal static void DivK(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> y)
        => y[i] = a[i] / b[i];

    /// <summary>
    /// NumPy-style broadcasting binary op. One thread per output element. The output's linear index
    /// is decomposed into per-axis coordinates via the precomputed row-major <paramref name="outStrides"/>,
    /// and those coordinates are re-projected into each operand through its broadcast strides
    /// (<paramref name="sA"/>/<paramref name="sB"/>, which are 0 on broadcast axes). Mirrors
    /// <c>BroadcastBinaryKernel</c> / <c>Nd</c> on the CPU side.
    /// </summary>
    internal static void BroadcastBinaryK(
        Index1D i,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> y,
        ArrayView<int> outStrides,
        ArrayView<int> sA,
        ArrayView<int> sB,
        int rank,
        int op)
    {
        int rem = i.X;
        int aOff = 0, bOff = 0;
        for (int ax = 0; ax < rank; ax++)
        {
            int st = outStrides[ax];
            int c = rem / st;
            rem -= c * st;
            aOff += c * sA[ax];
            bOff += c * sB[ax];
        }

        float va = a[aOff], vb = b[bOff];
        float r;
        if (op == OpAdd) r = va + vb;
        else if (op == OpSub) r = va - vb;
        else if (op == OpMul) r = va * vb;
        else r = va / vb;
        y[i] = r;
    }

    /// <summary>
    /// Batched matrix multiply (NumPy matmul semantics). One thread per output element across
    /// [batch, M, N]. The leading batch dimension is flattened on the host; for each flattened
    /// batch index the operand base offsets (already broadcast-resolved, in element units) are
    /// supplied via <paramref name="aBatchOff"/>/<paramref name="bBatchOff"/>. Mirrors <c>MatMulKernel</c>.
    /// </summary>
    internal static void MatMulK(
        Index1D idx,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> y,
        ArrayView<int> aBatchOff,
        ArrayView<int> bBatchOff,
        int M,
        int K,
        int N)
    {
        int gid = idx.X;
        int oMat = M * N;
        int bi = gid / oMat;
        int rem = gid - bi * oMat;
        int m = rem / N;
        int n = rem - m * N;

        int aRow = aBatchOff[bi] + m * K;
        int bBase = bBatchOff[bi];
        float sum = 0f;
        for (int kk = 0; kk < K; kk++)
            sum += a[aRow + kk] * b[bBase + kk * N + n];
        y[gid] = sum;
    }

    /// <summary>
    /// Integer GEMM for ONNX <c>MatMulInteger</c>: <c>Y = (A - a_zp) @ (B - b_zp)</c> with int32 accumulation.
    /// One thread per output element across [batch, M, N]. A and B arrive as int32 device buffers (the uint8/int8
    /// operands are widened to int32 on the host). Per-flattened-batch operand base offsets (element units, already
    /// broadcast-resolved) come via <paramref name="aBatchOff"/>/<paramref name="bBatchOff"/>, matching the float
    /// <see cref="MatMulK"/> layout. Zero points are supplied as int32 buffers with a stride selector
    /// (<paramref name="aZpStride"/> = 0 for per-tensor / absent, 1 for per-row of A; <paramref name="bZpStride"/>
    /// = 0 for per-tensor / absent, 1 for per-column of B) so the same kernel covers absent / per-tensor / per-row /
    /// per-column without branching on a null buffer. Mirrors the CPU <c>MatMulIntegerKernel</c> exactly: each
    /// product is computed as a 32-bit difference times a 32-bit difference; the running sum is int32 (wrapping on
    /// overflow exactly as the CPU kernel's <c>(int)</c> cast of its int64 accumulator does for in-range models).
    /// </summary>
    internal static void MatMulIntegerK(
        Index1D idx,
        ArrayView<int> a,
        ArrayView<int> b,
        ArrayView<int> y,
        ArrayView<int> aBatchOff,
        ArrayView<int> bBatchOff,
        ArrayView<int> aZp,
        ArrayView<int> bZp,
        int aZpStride,
        int bZpStride,
        int M,
        int K,
        int N)
    {
        int gid = idx.X;
        int oMat = M * N;
        int bi = gid / oMat;
        int rem = gid - bi * oMat;
        int m = rem / N;
        int n = rem - m * N;

        int aRow = aBatchOff[bi] + m * K;
        int bBase = bBatchOff[bi];
        int az = aZp[m * aZpStride];
        int bz = bZp[n * bZpStride];

        int sum = 0;
        for (int kk = 0; kk < K; kk++)
            sum += (a[aRow + kk] - az) * (b[bBase + kk * N + n] - bz);
        y[gid] = sum;
    }

    /// <summary>Tile edge for the shared-memory blocked int8 GEMM (<see cref="MatMulIntegerTiledK"/>). 16×16 threads/group.</summary>
    internal const int IntTile = 16;

    /// <summary>
    /// Shared-memory-tiled integer GEMM for one (already broadcast-resolved) batch of ONNX <c>MatMulInteger</c>:
    /// <c>Y[m,n] = Σ_k A'[m,k] · B'[k,n]</c> over int32 operands, where the zero points have ALREADY been
    /// subtracted on the host into <paramref name="a"/>/<paramref name="b"/> (so <c>A' = A − a_zp</c>,
    /// <c>B' = B − b_zp</c>). A 2-D grouped launch of <see cref="IntTile"/>×<see cref="IntTile"/> threads computes one
    /// output tile per group: each k-step stages an <c>IntTile×IntTile</c> block of A' and B' into shared memory
    /// (coalesced, one element per thread), barrier-syncs, then accumulates the tile's partial dot product. The
    /// running sum is <b>int32</b> (wrapping on overflow) — bit-identical to the naïve <see cref="MatMulIntegerK"/>
    /// (and to the CPU <c>MatMulIntegerKernel</c>'s <c>(int)</c> cast of its int64 accumulator for in-range models),
    /// since pre-subtracting the (int32) zero point and summing int32 products is the same arithmetic, just blocked.
    /// Out-of-range threads (M/N not a multiple of the tile) read zeros into shared memory and skip the store.
    /// <paramref name="aBase"/>/<paramref name="bBase"/>/<paramref name="yBase"/> are this batch's element offsets.
    /// </summary>
    internal static void MatMulIntegerTiledK(
        ArrayView<int> a,
        ArrayView<int> b,
        ArrayView<int> y,
        int aBase,
        int bBase,
        int yBase,
        int M,
        int K,
        int N)
    {
        // Shared staging tiles for A' [row, k] and B' [k, col], laid out row-major as flat 1-D shared memory
        // (index = localRow * IntTile + localCol) to keep the indexing unambiguous across backends.
        var aTile = SharedMemory.Allocate1D<int>(IntTile * IntTile);
        var bTile = SharedMemory.Allocate1D<int>(IntTile * IntTile);

        int ty = Group.IdxY;
        int tx = Group.IdxX;
        int row = Grid.IdxY * IntTile + ty;   // output m
        int col = Grid.IdxX * IntTile + tx;   // output n

        int sum = 0;
        int numTiles = (K + IntTile - 1) / IntTile;
        for (int t = 0; t < numTiles; t++)
        {
            int kA = t * IntTile + tx;   // column of A' this thread stages
            int kB = t * IntTile + ty;   // row of B' this thread stages

            aTile[ty * IntTile + tx] = (row < M && kA < K) ? a[aBase + row * K + kA] : 0;
            bTile[ty * IntTile + tx] = (kB < K && col < N) ? b[bBase + kB * N + col] : 0;

            Group.Barrier();

            for (int kk = 0; kk < IntTile; kk++)
                sum += aTile[ty * IntTile + kk] * bTile[kk * IntTile + tx];

            Group.Barrier();
        }

        if (row < M && col < N)
            y[yBase + row * N + col] = sum;
    }

    /// <summary>
    /// Direct 2-D convolution (NCHW). One thread per output element across [N, Cout, OutH, OutW],
    /// supporting stride, padding, dilation, groups and an optional bias. Mirrors <c>ConvKernel</c>.
    /// </summary>
    internal static void Conv2DK(
        Index1D idx,
        ArrayView<float> x,
        ArrayView<float> w,
        ArrayView<float> bias,
        ArrayView<float> y,
        ConvParams p)
    {
        int gid = idx.X;
        int yC = p.OutH * p.OutW;
        int yN = p.Cout * yC;

        int n = gid / yN;
        int r = gid - n * yN;
        int oc = r / yC;
        int r2 = r - oc * yC;
        int oy = r2 / p.OutW;
        int ox = r2 - oy * p.OutW;

        int g = oc / p.OutPerGroup;

        int xN = p.C * p.H * p.W;
        int xC = p.H * p.W;
        int wO = p.CinPerGroup * p.KH * p.KW;
        int wC = p.KH * p.KW;

        float sum = p.HasBias != 0 ? bias[oc] : 0f;
        for (int icg = 0; icg < p.CinPerGroup; icg++)
        {
            int ic = g * p.CinPerGroup + icg;
            int xBase = n * xN + ic * xC;
            int wBase = oc * wO + icg * wC;
            for (int ky = 0; ky < p.KH; ky++)
            {
                int iy = oy * p.SH - p.PadTop + ky * p.DH;
                if (iy < 0 || iy >= p.H) continue;
                int xRow = xBase + iy * p.W;
                int wRow = wBase + ky * p.KW;
                for (int kx = 0; kx < p.KW; kx++)
                {
                    int ix = ox * p.SW - p.PadLeft + kx * p.DW;
                    if (ix < 0 || ix >= p.W) continue;
                    sum += x[xRow + ix] * w[wRow + kx];
                }
            }
        }

        y[gid] = sum;
    }

    /// <summary>
    /// General N-D axis permutation (ONNX <c>Transpose</c>). One thread per output element: the output's
    /// linear index is decomposed into per-axis coordinates via the precomputed row-major
    /// <paramref name="outStrides"/>, and each coordinate is re-projected into the source through
    /// <paramref name="srcStrides"/> (where <c>srcStrides[i] = inStrides[perm[i]]</c>). Mirrors the CPU
    /// <c>TransposeKernel</c>.
    /// </summary>
    internal static void TransposeK(
        Index1D i,
        ArrayView<float> x,
        ArrayView<float> y,
        ArrayView<int> outStrides,
        ArrayView<int> srcStrides,
        int rank)
    {
        int rem = i.X;
        int src = 0;
        for (int ax = 0; ax < rank; ax++)
        {
            int st = outStrides[ax];
            int c = rem / st;
            rem -= c * st;
            src += c * srcStrides[ax];
        }
        y[i] = x[src];
    }

    /// <summary>
    /// Softmax along one axis, numerically stabilized by max-subtraction. One thread per (outer, inner)
    /// "row": the <paramref name="axisSize"/> elements of a row are <paramref name="inner"/> apart in
    /// memory. Computes the per-row max, then the exponentials and their sum, then divides — matching the
    /// accumulation order of the CPU <c>SoftmaxKernel</c>.
    /// </summary>
    internal static void SoftmaxK(
        Index1D idx,
        ArrayView<float> x,
        ArrayView<float> y,
        int axisSize,
        int inner)
    {
        int gid = idx.X;
        int o = gid / inner;
        int q = gid - o * inner;
        int baseIdx = o * axisSize * inner + q;

        float mx = x[baseIdx];
        for (int s = 1; s < axisSize; s++)
        {
            float v = x[baseIdx + s * inner];
            if (v > mx) mx = v;
        }

        float sum = 0f;
        for (int s = 0; s < axisSize; s++)
        {
            float e = MathF.Exp(x[baseIdx + s * inner] - mx);
            y[baseIdx + s * inner] = e;
            sum += e;
        }

        for (int s = 0; s < axisSize; s++)
            y[baseIdx + s * inner] = y[baseIdx + s * inner] / sum;
    }

    /// <summary>
    /// Axis reduction by summation (ONNX <c>ReduceSum</c> with <paramref name="divisor"/> = 1, or
    /// <c>ReduceMean</c> with <paramref name="divisor"/> = reduced-element count). One thread per output
    /// element: <paramref name="outBase"/> gives the input offset where every reduced coordinate is zero,
    /// and the reduced coordinates are then enumerated in row-major order (<paramref name="redOutStrides"/>
    /// over the reduced axes, <paramref name="redStrides"/> their input strides) — the same fold order as
    /// the CPU reduce kernels, so float results match.
    /// </summary>
    internal static void ReduceK(
        Index1D idx,
        ArrayView<float> x,
        ArrayView<float> y,
        ArrayView<int> outBase,
        ArrayView<int> redOutStrides,
        ArrayView<int> redStrides,
        int numRed,
        int redCount,
        float divisor)
    {
        int o = idx.X;
        int baseOff = outBase[o];
        float acc = 0f;
        for (int r = 0; r < redCount; r++)
        {
            int rem = r;
            int off = baseOff;
            for (int k = 0; k < numRed; k++)
            {
                int st = redOutStrides[k];
                int c = rem / st;
                rem -= c * st;
                off += c * redStrides[k];
            }
            acc += x[off];
        }
        y[o] = acc / divisor;
    }

    /// <summary>
    /// Layer normalization over the trailing <paramref name="norm"/> elements of each group:
    /// <c>y = (x − mean)/√(var + ε)·scale (+ bias)</c>. One thread per group of <paramref name="norm"/>
    /// contiguous elements. The per-group mean, then the summed squared deviations, then the normalized
    /// scale/bias apply are accumulated in the same order as the CPU <c>LayerNormalizationKernel</c>, so the
    /// float results match. <paramref name="hasBias"/> selects whether <paramref name="bias"/> is added.
    /// </summary>
    internal static void LayerNormK(
        Index1D idx,
        ArrayView<float> x,
        ArrayView<float> scale,
        ArrayView<float> bias,
        ArrayView<float> y,
        int norm,
        float eps,
        int hasBias)
    {
        int b = idx.X * norm;

        float mean = 0f;
        for (int i = 0; i < norm; i++) mean += x[b + i];
        mean /= norm;

        float varSum = 0f;
        for (int i = 0; i < norm; i++) { float d = x[b + i] - mean; varSum += d * d; }
        float inv = 1f / MathF.Sqrt(varSum / norm + eps);

        for (int i = 0; i < norm; i++)
        {
            float nval = (x[b + i] - mean) * inv;
            y[b + i] = nval * scale[i] + (hasBias != 0 ? bias[i] : 0f);
        }
    }

    /// <summary>
    /// Generic gather-by-offset: <c>y[i] = x[srcOffsets[i]]</c>. One thread per output element. Used for both
    /// ONNX <c>Gather</c> (axis indexing) and the float fast-path of <c>Slice</c>, where the host precomputes
    /// the per-output-element source offset (element units) matching the corresponding CPU kernel's layout.
    /// </summary>
    internal static void GatherK(
        Index1D idx,
        ArrayView<float> x,
        ArrayView<float> y,
        ArrayView<int> srcOffsets)
    {
        int i = idx.X;
        y[i] = x[srcOffsets[i]];
    }

    /// <summary>
    /// Elementwise <c>Pow</c> (base^exp) with NumPy-style broadcasting, sharing the broadcast-stride
    /// decomposition of <see cref="BroadcastBinaryK"/>. One thread per output element. Mirrors the CPU
    /// <c>PowKernel</c> (which is <c>MathF.Pow</c>).
    /// </summary>
    internal static void PowK(
        Index1D i,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> y,
        ArrayView<int> outStrides,
        ArrayView<int> sA,
        ArrayView<int> sB,
        int rank)
    {
        int rem = i.X;
        int aOff = 0, bOff = 0;
        for (int ax = 0; ax < rank; ax++)
        {
            int st = outStrides[ax];
            int c = rem / st;
            rem -= c * st;
            aOff += c * sA[ax];
            bOff += c * sB[ax];
        }
        y[i] = MathF.Pow(a[aOff], b[bOff]);
    }

    /// <summary>
    /// Block-wise n-bit (INT4/INT8) quantized matmul — the Microsoft contrib op <c>MatMulNBits</c>.
    /// Computes <c>Y[m, n] = Σ_k A[m, k] · W[n, k]</c> where <c>W[n, k] = (q − zp[n, b]) · scale[n, b]</c>
    /// is the on-the-fly dequantization of the packed weight row <c>n</c>, block <c>b = k / blockSize</c>.
    /// One thread per output element across the flattened <c>[M, N]</c> output (M = product of A's leading
    /// dims). The packed weights <paramref name="b"/> (<c>[N · nBlocksPerRow · blobSize]</c> bytes, widened
    /// to int per byte), per-(row,block) <paramref name="scales"/> (<c>[N · nBlocksPerRow]</c>), and zero
    /// points all live on the device.
    ///
    /// <para><b>Zero point selection (branchless via <paramref name="zpMode"/>):</b>
    /// 0 = default symmetric <c>2^(bits-1)</c> (the <paramref name="zpFloat"/>/<paramref name="zpPacked"/>
    /// buffers are unused 1-element sentinels); 1 = float per-(row,block) in <paramref name="zpFloat"/>
    /// indexed <c>n·nBlocksPerRow + b</c>; 2 = packed n-bit in <paramref name="zpPacked"/>, the same
    /// least-significant-first nibble packing as <paramref name="b"/>, row stride
    /// <paramref name="zpRowBytes"/>. Mirrors the CPU <c>MatMulNBitsKernel</c> bit-for-bit in dequant
    /// semantics (nibble order, default zp, accumulation order); float accumulation differs only by the
    /// usual GPU vs CPU rounding.</para>
    /// </summary>
    internal static void MatMulNBitsK(
        Index1D idx,
        ArrayView<float> a,
        ArrayView<int> b,
        ArrayView<float> scales,
        ArrayView<float> zpFloat,
        ArrayView<int> zpPacked,
        ArrayView<float> y,
        MatMulNBitsParams p)
    {
        int gid = idx.X;
        int N = p.N;
        int K = p.K;
        int m = gid / N;
        int n = gid - m * N;

        int bits = p.Bits;
        int mask = (1 << bits) - 1;
        float defaultZp = 1 << (bits - 1);
        int bRowBase = n * p.NBlocksPerRow * p.BlobSize;
        int scaleRowBase = n * p.NBlocksPerRow;
        int zpRowBase = n * p.ZpRowBytes;
        int aRow = m * K;

        float sum = 0f;
        for (int bk = 0; bk < p.NBlocksPerRow; bk++)
        {
            float scale = scales[scaleRowBase + bk];

            float zp;
            if (p.ZpMode == 1) zp = zpFloat[scaleRowBase + bk];
            else if (p.ZpMode == 2) zp = UnpackNBitK(zpPacked, zpRowBase, bk, bits, mask);
            else zp = defaultZp;

            int blobBase = bRowBase + bk * p.BlobSize;
            int kStart = bk * p.BlockSize;
            int kEnd = kStart + p.BlockSize;
            if (kEnd > K) kEnd = K;
            for (int k = kStart; k < kEnd; k++)
            {
                int inBlock = k - kStart;
                int q = UnpackNBitK(b, blobBase, inBlock, bits, mask);
                float w = (q - zp) * scale;
                sum += a[aRow + k] * w;
            }
        }
        y[gid] = sum;
    }

    /// <summary>
    /// Unpacks the <paramref name="index"/>-th <paramref name="bits"/>-bit value from a byte-per-int region
    /// starting at <paramref name="baseByte"/>, least-significant-first (bits=4: even index = low nibble,
    /// odd = high nibble; bits=8: one value per byte). Mirrors the CPU <c>MatMulNBitsKernel.UnpackNBit</c>.
    /// </summary>
    private static int UnpackNBitK(ArrayView<int> data, int baseByte, int index, int bits, int mask)
    {
        if (bits == 8) return data[baseByte + index];
        int byteOff = baseByte + (index >> 1);
        int shift = (index & 1) * 4;
        return (data[byteOff] >> shift) & mask;
    }

    /// <summary>
    /// In-place RoPE on a <c>[B, S, heads·headDim]</c> float buffer (NeoX half-split or GPT-J interleaved),
    /// one thread per (b, s, head, j) rotary pair. The token at sequence index <c>s</c> uses cos/sin cache
    /// row <c>pastSeq + s</c> (clamped to <paramref name="maxPos"/>−1); cos/sin caches are
    /// <c>[maxPos, half]</c>. Mirrors the CPU GQA <c>ApplyRotary</c>: <paramref name="interleaved"/> ≠ 0
    /// pairs <c>(2j, 2j+1)</c>, else <c>(j, j+rotHalf)</c>. Channels beyond the rotary span are untouched
    /// (this kernel only launches over the rotHalf pairs). The grid is
    /// <c>[batch · seq · heads · rotHalf]</c>.
    /// </summary>
    internal static void RotaryK(
        Index1D idx,
        ArrayView<float> buf,
        ArrayView<float> cos,
        ArrayView<float> sin,
        int seq,
        int heads,
        int headDim,
        int rotHalf,
        int half,
        int pastSeq,
        int maxPos,
        int interleaved)
    {
        int gid = idx.X;
        int j = gid % rotHalf;
        int t = gid / rotHalf;
        int h = t % heads;
        int t2 = t / heads;
        int s = t2 % seq;
        int b = t2 / seq;

        int pos = pastSeq + s;
        if (pos >= maxPos) pos = maxPos - 1;
        int cacheBase = pos * half;
        float c = cos[cacheBase + j];
        float sn = sin[cacheBase + j];

        int rowBase = ((b * seq + s) * heads + h) * headDim;
        int i0, i1;
        if (interleaved != 0) { i0 = rowBase + 2 * j; i1 = i0 + 1; }
        else { i0 = rowBase + j; i1 = rowBase + j + rotHalf; }
        float va = buf[i0];
        float vb = buf[i1];
        buf[i0] = va * c - vb * sn;
        buf[i1] = vb * c + va * sn;
    }

    /// <summary>
    /// Copies a step's per-token K (or V) — laid out <c>[B, kvSeq, kvNumHeads·headDim]</c> — into a present
    /// cache laid out <c>[B, kvNumHeads, totalSeq, headDim]</c> at sequence offset <paramref name="pastSeq"/>.
    /// One thread per (b, s, g, d) element. Mirrors the CPU GQA's new-K/V scatter. (Past K/V, when present,
    /// is staged into the cache by a separate device→device copy on the host side.)
    /// </summary>
    internal static void GqaScatterKvK(
        Index1D idx,
        ArrayView<float> src,
        ArrayView<float> present,
        int kvSeq,
        int kvNumHeads,
        int headDim,
        int totalSeq,
        int pastSeq)
    {
        int gid = idx.X;
        int d = gid % headDim;
        int t = gid / headDim;
        int g = t % kvNumHeads;
        int t2 = t / kvNumHeads;
        int s = t2 % kvSeq;
        int b = t2 / kvSeq;

        int kvHid = kvNumHeads * headDim;
        int srcOff = (b * kvSeq + s) * kvHid + g * headDim + d;
        int dstOff = ((b * kvNumHeads + g) * totalSeq + pastSeq + s) * headDim + d;
        present[dstOff] = src[srcOff];
    }

    /// <summary>
    /// Grouped-query attention core: for each (b, h, qi) computes causal scaled-dot-product attention of the
    /// query against the cached present-K/V and writes the context into <paramref name="outBuf"/>
    /// (<c>[B, qSeq, numHeads·headDim]</c>). One thread per (b, h, qi) — i.e. per output row of headDim
    /// elements. Present K/V are <c>[B, kvNumHeads, totalSeq, headDim]</c>; query head <c>h</c> reads KV head
    /// <c>h / groupSize</c> (repeat-KV). Causal bound: key positions <c>0 … min(pastSeq+qi, seqBound[b])</c>.
    /// Scores are computed, max-subtracted, exponentiated and normalized inline (no scratch buffer — the
    /// two-pass online form), then the V-weighted sum is accumulated — matching the CPU GQA accumulation
    /// order. <paramref name="seqBounds"/> is the per-batch last-valid-key index (<c>seqlens_k</c>), or a
    /// single-element buffer of <c>totalSeq−1</c> when absent (selected by <paramref name="hasSeqBound"/>).
    /// </summary>
    internal static void GqaAttentionK(
        Index1D idx,
        ArrayView<float> q,
        ArrayView<float> presentK,
        ArrayView<float> presentV,
        ArrayView<float> outBuf,
        ArrayView<int> seqBounds,
        int qSeq,
        int numHeads,
        int kvNumHeads,
        int headDim,
        int totalSeq,
        int pastSeq,
        int groupSize,
        float scale,
        int hasSeqBound)
    {
        int gid = idx.X;
        int qi = gid % qSeq;
        int t = gid / qSeq;
        int h = t % numHeads;
        int b = t / numHeads;

        int g = h / groupSize;
        int qHid = numHeads * headDim;
        int qRow = (b * qSeq + qi) * qHid + h * headDim;

        int seqBound = hasSeqBound != 0 ? seqBounds[b] : totalSeq - 1;
        int kLimit = pastSeq + qi;
        if (kLimit > seqBound) kLimit = seqBound;

        // Pass 1: max score.
        float mx = float.NegativeInfinity;
        for (int kj = 0; kj <= kLimit; kj++)
        {
            int kBase = ((b * kvNumHeads + g) * totalSeq + kj) * headDim;
            float dot = 0f;
            for (int d = 0; d < headDim; d++) dot += q[qRow + d] * presentK[kBase + d];
            float sc = dot * scale;
            if (sc > mx) mx = sc;
        }

        // Pass 2: sum of exps.
        float sum = 0f;
        for (int kj = 0; kj <= kLimit; kj++)
        {
            int kBase = ((b * kvNumHeads + g) * totalSeq + kj) * headDim;
            float dot = 0f;
            for (int d = 0; d < headDim; d++) dot += q[qRow + d] * presentK[kBase + d];
            sum += MathF.Exp(dot * scale - mx);
        }
        float inv = sum > 0f ? 1f / sum : 0f;

        // Pass 3: V-weighted sum.
        int outRow = (b * qSeq + qi) * qHid + h * headDim;
        for (int d = 0; d < headDim; d++) outBuf[outRow + d] = 0f;
        for (int kj = 0; kj <= kLimit; kj++)
        {
            int kBase = ((b * kvNumHeads + g) * totalSeq + kj) * headDim;
            float dot = 0f;
            for (int d = 0; d < headDim; d++) dot += q[qRow + d] * presentK[kBase + d];
            float w = MathF.Exp(dot * scale - mx) * inv;
            int vBase = ((b * kvNumHeads + g) * totalSeq + kj) * headDim;
            for (int d = 0; d < headDim; d++) outBuf[outRow + d] += w * presentV[vBase + d];
        }
    }

    /// <summary>
    /// Elementwise <c>Where</c>: <c>cond != 0 ? x : y</c>, with NumPy-style broadcasting over all three
    /// operands. <paramref name="cond"/> is supplied as a float buffer (0/1). One thread per output element.
    /// Mirrors the CPU <c>WhereKernel</c> (float value path).
    /// </summary>
    internal static void WhereK(
        Index1D i,
        ArrayView<float> cond,
        ArrayView<float> x,
        ArrayView<float> y,
        ArrayView<float> outBuf,
        ArrayView<int> outStrides,
        ArrayView<int> sC,
        ArrayView<int> sX,
        ArrayView<int> sY,
        int rank)
    {
        int rem = i.X;
        int cOff = 0, xOff = 0, yOff = 0;
        for (int ax = 0; ax < rank; ax++)
        {
            int st = outStrides[ax];
            int c = rem / st;
            rem -= c * st;
            cOff += c * sC[ax];
            xOff += c * sX[ax];
            yOff += c * sY[ax];
        }
        outBuf[i] = cond[cOff] != 0f ? x[xOff] : y[yOff];
    }
}

/// <summary>
/// Blittable bundle of scalar <c>MatMulNBits</c> layout parameters passed by value to
/// <see cref="GpuKernels.MatMulNBitsK"/>. Packed into a struct because the kernel needs more scalars than the
/// 15-arg <c>LoadAutoGroupedStreamKernel</c> generic delegate arity allows.
/// </summary>
public readonly struct MatMulNBitsParams
{
    /// <summary>Flattened A rows (product of A's leading dims).</summary>
    public readonly int M;
    /// <summary>Contraction dimension.</summary>
    public readonly int K;
    /// <summary>Output columns (weight rows).</summary>
    public readonly int N;
    /// <summary>Quantization bit width (4 or 8).</summary>
    public readonly int Bits;
    /// <summary>Block size along K.</summary>
    public readonly int BlockSize;
    /// <summary>Blocks per weight row = ceil(K / BlockSize).</summary>
    public readonly int NBlocksPerRow;
    /// <summary>Packed bytes per block = ceil(BlockSize · Bits / 8).</summary>
    public readonly int BlobSize;
    /// <summary>Packed zero-point bytes per weight row = ceil(NBlocksPerRow · Bits / 8).</summary>
    public readonly int ZpRowBytes;
    /// <summary>Zero-point selector: 0 = default symmetric, 1 = float per-(row,block), 2 = packed n-bit.</summary>
    public readonly int ZpMode;

    /// <summary>Bundles all MatMulNBits scalar parameters.</summary>
    public MatMulNBitsParams(int m, int k, int n, int bits, int blockSize,
        int nBlocksPerRow, int blobSize, int zpRowBytes, int zpMode)
    {
        M = m; K = k; N = n; Bits = bits; BlockSize = blockSize;
        NBlocksPerRow = nBlocksPerRow; BlobSize = blobSize; ZpRowBytes = zpRowBytes; ZpMode = zpMode;
    }
}

/// <summary>
/// Blittable bundle of scalar Conv2D parameters passed by value to <see cref="GpuKernels.Conv2DK"/>.
/// Packed into a struct because the kernel needs more scalars than a generic delegate arity allows.
/// </summary>
public readonly struct ConvParams
{
    /// <summary>Batch size.</summary>
    public readonly int N;
    /// <summary>Total input channels (NCHW C).</summary>
    public readonly int C;
    /// <summary>Input height.</summary>
    public readonly int H;
    /// <summary>Input width.</summary>
    public readonly int W;
    /// <summary>Output channels.</summary>
    public readonly int Cout;
    /// <summary>Input channels per group (weight dim 1).</summary>
    public readonly int CinPerGroup;
    /// <summary>Kernel height.</summary>
    public readonly int KH;
    /// <summary>Kernel width.</summary>
    public readonly int KW;
    /// <summary>Output height.</summary>
    public readonly int OutH;
    /// <summary>Output width.</summary>
    public readonly int OutW;
    /// <summary>Output channels per group.</summary>
    public readonly int OutPerGroup;
    /// <summary>Stride along height.</summary>
    public readonly int SH;
    /// <summary>Stride along width.</summary>
    public readonly int SW;
    /// <summary>Dilation along height.</summary>
    public readonly int DH;
    /// <summary>Dilation along width.</summary>
    public readonly int DW;
    /// <summary>Top padding.</summary>
    public readonly int PadTop;
    /// <summary>Left padding.</summary>
    public readonly int PadLeft;
    /// <summary>Non-zero when a bias buffer is supplied.</summary>
    public readonly int HasBias;

    /// <summary>Bundles all Conv2D scalar parameters.</summary>
    public ConvParams(
        int n, int c, int h, int w, int cout, int cinPerGroup, int kH, int kW,
        int outH, int outW, int outPerGroup, int sH, int sW, int dH, int dW,
        int padTop, int padLeft, int hasBias)
    {
        N = n; C = c; H = h; W = w; Cout = cout; CinPerGroup = cinPerGroup;
        KH = kH; KW = kW; OutH = outH; OutW = outW; OutPerGroup = outPerGroup;
        SH = sH; SW = sW; DH = dH; DW = dW; PadTop = padTop; PadLeft = padLeft;
        HasBias = hasBias;
    }
}
