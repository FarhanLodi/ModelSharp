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
