using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace ModelSharp.Native;

/// <summary>
/// Optional native fast path for fp32 GEMM, backed by <c>libms_kernels.so</c> (built from
/// <c>native/</c>). This is a <b>guarded accelerator, never a dependency</b>: when the shared
/// library is missing, the CPU lacks AVX-512, the buffers are strided sub-views, or the op is
/// tiny, <see cref="TryMultiply"/> returns <c>false</c> and the caller runs the pure-managed
/// kernel. So ModelSharp stays portable (Windows/macOS/Linux, x86/ARM) and only opts into native
/// speed where it is actually available.
/// </summary>
/// <remarks>
/// Enabled automatically when the library loads and advertises AVX-512. Disable with the
/// environment variable <c>MODELSHARP_NATIVE=0</c> (or <c>off</c>/<c>false</c>), or set
/// <see cref="Enabled"/> programmatically. Override the library path with
/// <c>MODELSHARP_NATIVE_LIB=/path/to/libms_kernels.so</c>.
/// <para><b>Deployment caveat:</b> the shipped <c>native/Makefile</c> builds with
/// <c>-march=native</c>, so the resulting <c>.so</c> is tuned to the build host's ISA. Ship a
/// <c>.so</c> built for the target CPU family (or a portable <c>-mavx512f ...</c> build); never
/// hand an AVX-512 build to a CPU without AVX-512.</para>
/// </remarks>
internal static class NativeGemm
{
    private const string Lib = "ms_kernels";

    /// <summary>Below this many multiply-accumulates the P/Invoke + thread spin-up overhead is not
    /// worth it, so we keep the op on the managed path.</summary>
    private const long MinWork = 1L << 15;

    private static readonly bool _available;
    private static readonly bool _hasAvx512;

    /// <summary>Whether the native GEMM path is active. Defaults to (library present AND AVX-512 AND
    /// not disabled via env). Settable for benchmarks/tests that want to force the managed path.</summary>
    public static bool Enabled { get; set; }

    static NativeGemm()
    {
        try
        {
            NativeLibrary.SetDllImportResolver(typeof(NativeGemm).Assembly, Resolve);
            _available = Probe();
            _hasAvx512 = _available && ms_has_avx512() != 0;
        }
        catch { _available = false; _hasAvx512 = false; }

        string? flag = Environment.GetEnvironmentVariable("MODELSHARP_NATIVE");
        bool killed = flag is "0" ||
                      string.Equals(flag, "false", StringComparison.OrdinalIgnoreCase) ||
                      string.Equals(flag, "off", StringComparison.OrdinalIgnoreCase);
        Enabled = _available && _hasAvx512 && !killed;
    }

    /// <summary>True when the native library loaded and advertises AVX-512 (regardless of the
    /// enable flag); useful for diagnostics.</summary>
    public static bool Available => _available && _hasAvx512;

    /// <summary>
    /// Computes <c>Y[M,N] = A[M,K] · B[K,N]</c> natively, overwriting <paramref name="y"/>, and
    /// returns <c>true</c>. Returns <c>false</c> (touching nothing) when the native path is
    /// unavailable/disabled, the strides are not the natural contiguous ones (the native kernel
    /// has no sub-row stride support), or the work is below <see cref="MinWork"/>.
    /// </summary>
    public static bool TryMultiply(
        float[] a, int aOff, int lda,
        float[] b, int bOff, int ldb,
        float[] y, int yOff, int ldc,
        int M, int N, int K)
    {
        if (!Enabled) return false;
        if (M <= 0 || N <= 0 || K <= 0) return false;
        // Native kernel assumes contiguous row-major tiles (lda=K, ldb=N, ldc=N), no sub-row views.
        if (lda != K || ldb != N || ldc != N) return false;
        if ((long)M * N * K < MinWork) return false;
        // Bounds sanity: the native kernel reads/writes these contiguous extents from the offsets.
        if ((long)aOff + (long)M * K > a.Length) return false;
        if ((long)bOff + (long)K * N > b.Length) return false;
        if ((long)yOff + (long)M * N > y.Length) return false;

        unsafe
        {
            fixed (float* pa = &a[aOff], pb = &b[bOff], py = &y[yOff])
                ms_sgemm_f32(pa, pb, py, M, N, K);
        }
        return true;
    }

    private static bool Probe()
    {
        try { _ = ms_has_avx512(); return true; } catch { return false; }
    }

    private static IntPtr Resolve(string name, Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        var candidates = new List<string>();
        string? envLib = Environment.GetEnvironmentVariable("MODELSHARP_NATIVE_LIB");
        if (!string.IsNullOrEmpty(envLib)) candidates.Add(envLib);

        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && !string.IsNullOrEmpty(dir); i++)
        {
            candidates.Add(Path.Combine(dir, "libms_kernels.so"));
            candidates.Add(Path.Combine(dir, "native", "build", "libms_kernels.so"));
            dir = Path.GetDirectoryName(dir.TrimEnd(Path.DirectorySeparatorChar));
        }

        foreach (var c in candidates)
            if (File.Exists(c) && NativeLibrary.TryLoad(c, out var h)) return h;
        return NativeLibrary.TryLoad("libms_kernels.so", out var g) ? g : IntPtr.Zero;
    }

    [DllImport(Lib, EntryPoint = "ms_has_avx512")]
    private static extern int ms_has_avx512();

    [DllImport(Lib, EntryPoint = "ms_sgemm_f32")]
    private static extern unsafe void ms_sgemm_f32(float* a, float* b, float* c, int m, int n, int k);
}
