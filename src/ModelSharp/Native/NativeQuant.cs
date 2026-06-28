using System;
using System.Runtime.InteropServices;

namespace ModelSharp.Native;

/// <summary>
/// Optional native fast paths for the <c>MatMulNBits</c> quantized matmul, backed by
/// <c>libms_kernels.so</c>. Two independent, <b>opt-in</b> accelerators are offered:
/// <list type="bullet">
/// <item><description><b>Accurate n-bit</b> (<c>ms_qgemm_nbits</c>): dequantizes weights to fp32
/// and matches the managed kernel's result. Enabled by <c>MODELSHARP_NATIVE_QUANT</c>.</description></item>
/// <item><description><b>W4A8 VNNI</b> (<c>ms_qgemm_w4a8</c>): quantizes activations to int8 and
/// uses AVX512-VNNI. <b>Approximate</b> (~0.5% rel-L2 error) but fast. Enabled by
/// <c>MODELSHARP_W4A8</c>; additionally requires VNNI support.</description></item>
/// </list>
/// Both default <b>OFF</b>, so existing behaviour and tests are unaffected unless explicitly
/// enabled. Like <see cref="NativeGemm"/> this is a guarded accelerator: when unavailable, the
/// sizes don't match, or the flag is off, <see cref="TryMatMulNBits"/> returns <c>false</c> and the
/// caller runs the pure-managed kernel.
/// </summary>
/// <remarks>
/// The DllImport resolver is registered once per assembly by <see cref="NativeGemm"/>; this type
/// reuses it (referencing <see cref="NativeGemm.Available"/> forces that static ctor to run) and
/// never calls <c>SetDllImportResolver</c> again.
/// </remarks>
internal static class NativeQuant
{
    private const string Lib = "ms_kernels";

    private static readonly bool _available;   // library loaded + AVX-512
    private static readonly bool _hasVnni;     // library loaded + AVX512-VNNI

    /// <summary>Accurate native n-bit path. Defaults to (library + AVX-512 present AND
    /// <c>MODELSHARP_NATIVE_QUANT</c> in {1,on,true}). Settable for tests/benchmarks.</summary>
    public static bool NativeNBitsEnabled { get; set; }

    /// <summary>Fast approximate W4A8 path. Defaults to (library + AVX-512 + VNNI present AND
    /// <c>MODELSHARP_W4A8</c> in {1,on,true}). Settable for tests/benchmarks.</summary>
    public static bool W4A8Enabled { get; set; }

    static NativeQuant()
    {
        try
        {
            // Force NativeGemm's static ctor to run so its DllImportResolver is registered for
            // this assembly. We must not call SetDllImportResolver again ourselves.
            _ = NativeGemm.Available;

            _available = Probe() && ms_has_avx512() != 0;
            _hasVnni = _available && ms_has_vnni() != 0;
        }
        catch { _available = false; _hasVnni = false; }

        NativeNBitsEnabled = _available && IsOn(Environment.GetEnvironmentVariable("MODELSHARP_NATIVE_QUANT"));
        W4A8Enabled = _available && _hasVnni && IsOn(Environment.GetEnvironmentVariable("MODELSHARP_W4A8"));
    }

    /// <summary>True when the library loaded and advertises AVX-512 (diagnostics).</summary>
    public static bool Available => _available;

    /// <summary>True when the library loaded and advertises AVX512-VNNI (diagnostics).</summary>
    public static bool HasVnni => _hasVnni;

    private static bool IsOn(string? v) =>
        v is "1" ||
        string.Equals(v, "on", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(v, "true", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Computes <c>y[M,N] = a[M,K] · dequant(bq[N,K])ᵀ</c> natively, overwriting <paramref name="y"/>,
    /// and returns <c>true</c>. Returns <c>false</c> (touching nothing) when no native path is
    /// enabled, the dims are non-positive, or any buffer is too small for the implied extents — in
    /// which case the caller must run the managed kernel.
    /// <para><c>zeroPoints</c> are packed n-bit zero points (one per (row, block)), or <c>null</c>
    /// for the symmetric default <c>2^(bits-1)</c>. The native ABI does not accept float zero
    /// points; callers with float zero points must not use this path.</para>
    /// </summary>
    public static bool TryMatMulNBits(
        float[] a, byte[] bq, float[] scales, byte[]? zeroPoints, float[] y,
        int M, int N, int K, int bits, int blockSize)
    {
        bool useW4A8 = W4A8Enabled;
        bool useNBits = !useW4A8 && NativeNBitsEnabled;
        if (!useW4A8 && !useNBits) return false;

        if (M <= 0 || N <= 0 || K <= 0 || blockSize <= 0) return false;
        if (bits != 4 && bits != 8) return false;

        int nBlocksPerRow = (K + blockSize - 1) / blockSize;
        int blobSize = (blockSize * bits + 7) / 8;

        // Bounds sanity: native reads/writes exactly these contiguous extents.
        if ((long)M * K > a.Length) return false;
        if ((long)N * nBlocksPerRow * blobSize > bq.Length) return false;
        if ((long)N * nBlocksPerRow > scales.Length) return false;
        if ((long)M * N > y.Length) return false;
        if (zeroPoints is not null)
        {
            int zpRowBytes = (nBlocksPerRow * bits + 7) / 8;
            if ((long)N * zpRowBytes > zeroPoints.Length) return false;
        }

        unsafe
        {
            fixed (float* pa = a, pScales = scales, py = y)
            fixed (byte* pbq = bq, pzp = zeroPoints)
            {
                if (useW4A8)
                    ms_qgemm_w4a8(pa, pbq, pScales, pzp, py, M, N, K, bits, blockSize);
                else
                    ms_qgemm_nbits(pa, pbq, pScales, pzp, py, M, N, K, bits, blockSize);
            }
        }
        return true;
    }

    private static bool Probe()
    {
        try { _ = ms_has_avx512(); return true; } catch { return false; }
    }

    [DllImport(Lib, EntryPoint = "ms_has_avx512")]
    private static extern int ms_has_avx512();

    [DllImport(Lib, EntryPoint = "ms_has_vnni")]
    private static extern int ms_has_vnni();

    [DllImport(Lib, EntryPoint = "ms_qgemm_nbits")]
    private static extern unsafe void ms_qgemm_nbits(
        float* a, byte* bq, float* scales, byte* zero_points, float* y,
        int M, int N, int K, int bits, int block_size);

    [DllImport(Lib, EntryPoint = "ms_qgemm_w4a8")]
    private static extern unsafe void ms_qgemm_w4a8(
        float* a, byte* bq, float* scales, byte* zero_points, float* y,
        int M, int N, int K, int bits, int block_size);
}
