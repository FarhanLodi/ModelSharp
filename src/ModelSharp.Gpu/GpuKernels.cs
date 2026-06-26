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
