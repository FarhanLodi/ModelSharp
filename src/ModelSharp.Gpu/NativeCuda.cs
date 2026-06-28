using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace ModelSharp.Gpu;

/// <summary>
/// P/Invoke binding to the native cuBLAS GPU fast path, backed by <c>libms_cuda.so</c> (built from
/// <c>native/</c>). Like <c>ModelSharp.Native.NativeGemm</c>, this is a <b>guarded accelerator, never a
/// dependency</b>: when the shared library is missing or no CUDA device is present, <see cref="Available"/>
/// is <c>false</c> and callers stay on the managed/ILGPU path.
/// </summary>
/// <remarks>
/// <para>The native library uses the CUDA <b>runtime-API primary context</b>. ILGPU creates its own
/// driver-API context, so device pointers from one are <b>not</b> valid in the other. Never hand an ILGPU
/// buffer pointer to <see cref="SgemmDev"/> / <see cref="SgemmResidentB"/>; use the host copy-in/out
/// <see cref="Sgemm(float[],float[],float[],int,int,int)"/> across the context boundary, or keep an
/// allocation resident entirely on the cuBLAS side via <see cref="Alloc"/>/<see cref="Upload"/>.</para>
/// <para>Override the library path with <c>MODELSHARP_CUDA_LIB=/path/to/libms_cuda.so</c>.</para>
/// </remarks>
public static class NativeCuda
{
    private const string Lib = "ms_cuda";

    private static readonly bool _available;
    private static string? _deviceName;

    static NativeCuda()
    {
        try
        {
            NativeLibrary.SetDllImportResolver(typeof(NativeCuda).Assembly, Resolve);
            _available = ms_cuda_available() == 1;
            // Opt-in TF32 Tensor Cores for cuBLAS (fp32 in/out, ~5-8x on Ada, ~1e-3 accuracy).
            if (_available &&
                Environment.GetEnvironmentVariable("MODELSHARP_TF32") is "1" or "on" or "true" or "True" or "ON" or "TRUE")
                ms_cuda_set_tf32(1);
        }
        catch { _available = false; }
    }

    /// <summary>Enable/disable TF32 Tensor Core math for all cuBLAS GEMMs (single, resident, batched).
    /// fp32 inputs/outputs, Tensor Core compute (~5-8x on Ada), ~1e-3 relative accuracy. Default off.</summary>
    public static void SetTf32(bool enable) { if (_available) ms_cuda_set_tf32(enable ? 1 : 0); }

    /// <summary>True when TF32 Tensor Core math is currently enabled.</summary>
    public static bool Tf32Enabled => _available && ms_cuda_get_tf32() != 0;

    /// <summary>True when <c>libms_cuda.so</c> loaded and a CUDA device is present.</summary>
    public static bool Available => _available;

    /// <summary>Name of the active CUDA device (e.g. "NVIDIA GeForce RTX 4090"), or "" if unavailable.</summary>
    public static string DeviceName
    {
        get
        {
            if (_deviceName != null) return _deviceName;
            try
            {
                IntPtr p = ms_cuda_device_name();
                _deviceName = p == IntPtr.Zero ? "" : (Marshal.PtrToStringAnsi(p) ?? "");
            }
            catch { _deviceName = ""; }
            return _deviceName;
        }
    }

    // ---- copy-in/out single GEMM: row-major C[M,N] = A[M,K] * B[K,N] ----

    /// <summary>
    /// Computes <c>C[M,N] = A[M,K] · B[K,N]</c> (row-major) on the GPU via cuBLAS, copying the operands in
    /// and the result out each call. Context-safe (host memory crosses the boundary). Throws if the native
    /// path is unavailable — callers should gate on <see cref="Available"/>.
    /// </summary>
    public static void Sgemm(float[] a, float[] b, float[] c, int M, int N, int K)
    {
        if (!_available) throw new InvalidOperationException("cuBLAS native path unavailable.");
        if ((long)a.Length < (long)M * K) throw new ArgumentException("A too small.", nameof(a));
        if ((long)b.Length < (long)K * N) throw new ArgumentException("B too small.", nameof(b));
        if ((long)c.Length < (long)M * N) throw new ArgumentException("C too small.", nameof(c));
        unsafe
        {
            fixed (float* pa = a, pb = b, pc = c)
                ms_cuda_sgemm(pa, pb, pc, M, N, K);
        }
    }

    // ---- resident device API (all pointers belong to the cuBLAS primary context) ----

    /// <summary>Allocate <paramref name="nFloats"/> floats of device memory; returns an opaque device handle.</summary>
    public static IntPtr Alloc(long nFloats) => ms_cuda_alloc((nuint)nFloats);

    /// <summary>Upload a host [rows,cols] row-major matrix to a fresh device allocation; returns the handle.</summary>
    public static IntPtr Upload(float[] host, int rows, int cols)
    {
        unsafe { fixed (float* p = host) return ms_cuda_upload(p, rows, cols); }
    }

    /// <summary>Copy <paramref name="nFloats"/> from host into an existing device allocation. Returns 0 on success.</summary>
    public static int UploadInto(IntPtr dev, float[] host, long nFloats)
    {
        unsafe { fixed (float* p = host) return ms_cuda_upload_into(dev, p, (nuint)nFloats); }
    }

    /// <summary>Download <paramref name="nFloats"/> from a device allocation into host. Returns 0 on success.</summary>
    public static int Download(IntPtr dev, float[] host, long nFloats)
    {
        unsafe { fixed (float* p = host) return ms_cuda_download(dev, p, (nuint)nFloats); }
    }

    /// <summary>Free a device allocation obtained from <see cref="Alloc"/>/<see cref="Upload"/>.</summary>
    public static void Free(IntPtr dev) => ms_cuda_free(dev);

    /// <summary>Enqueue-only resident GEMM: <c>C=A·B</c> with all operands already on the device. Returns 0 on success.</summary>
    public static int SgemmDev(IntPtr dA, IntPtr dB, IntPtr dC, int M, int N, int K) =>
        ms_cuda_sgemm_dev(dA, dB, dC, M, N, K);

    /// <summary>GEMM with host A, host C, but a resident device B (a persistent weight). Copies A in / C out.</summary>
    public static void SgemmResidentB(float[] a, IntPtr dB, float[] c, int M, int N, int K)
    {
        unsafe
        {
            fixed (float* pa = a, pc = c)
                ms_cuda_sgemm_with_resident_b(pa, dB, pc, M, N, K);
        }
    }

    // ---- context-aware path (operates inside a caller-supplied driver CUcontext) ----

    /// <summary>
    /// Enqueue-only GEMM <c>C[M,N]=A[M,K]·B[K,N]</c> (row-major) where the device pointers belong to the
    /// driver context <paramref name="cuContext"/> (e.g. ILGPU's <c>Accelerator.NativePtr</c>), run on
    /// <paramref name="cuStream"/> (ILGPU's <c>DefaultStream.StreamPtr</c>, or <see cref="IntPtr.Zero"/> for
    /// the context default). This is the context-safe way to call cuBLAS directly on ILGPU buffers — it
    /// pushes the context, uses a per-context cached cuBLAS handle, and pops. Returns 0 on success.
    /// </summary>
    public static int SgemmCtx(IntPtr cuContext, IntPtr cuStream, IntPtr dA, IntPtr dB, IntPtr dC, int M, int N, int K)
    {
        int rc = ms_cuda_sgemm_ctx(cuContext, cuStream, dA, dB, dC, M, N, K);
        CtxCalls++;
        LastCtxStatus = rc;
        return rc;
    }

    /// <summary>Diagnostics: number of <see cref="SgemmCtx"/> invocations this process.</summary>
    public static long CtxCalls;
    /// <summary>Diagnostics: status returned by the most recent <see cref="SgemmCtx"/> (0 = success).</summary>
    public static int LastCtxStatus;

    /// <summary>
    /// Strided-batched GEMM inside <paramref name="cuContext"/>: for each b in [0,batch),
    /// <c>C[b]=A[b]·B[b]</c> where the b-th matrix starts at base + b·stride (strides in elements).
    /// <paramref name="strideB"/> (or A) may be 0 for a shared/broadcast operand. Returns 0 on success.
    /// </summary>
    public static int SgemmStridedBatchedCtx(IntPtr cuContext, IntPtr cuStream, IntPtr dA, IntPtr dB, IntPtr dC,
        int M, int N, int K, long strideA, long strideB, long strideC, int batch) =>
        ms_cuda_sgemm_strided_batched_ctx(cuContext, cuStream, dA, dB, dC, M, N, K, strideA, strideB, strideC, batch);

    /// <summary>
    /// Destroy the cached cuBLAS handle bound to <paramref name="cuContext"/>. Must be called while that context
    /// is still alive (e.g. just before ILGPU disposes its accelerator), otherwise a later CUDA context that
    /// reuses the same pointer value would receive a handle bound to the destroyed context. No-op if uncached.
    /// </summary>
    public static void ReleaseContext(IntPtr cuContext)
    {
        if (_available) ms_cuda_release_context(cuContext);
    }

    /// <summary>Synchronize a specific stream (or the device if <see cref="IntPtr.Zero"/>). Returns 0 on success.</summary>
    public static int SyncStream(IntPtr cuStream) => ms_cuda_sync_stream(cuStream);

    /// <summary>Block until all enqueued device work completes. Returns 0 on success.</summary>
    public static int Sync() => ms_cuda_sync();

    /// <summary>Release the cuBLAS handle / runtime context.</summary>
    public static void Shutdown() => ms_cuda_shutdown();

    // ---- resolver ----

    private static IntPtr Resolve(string name, Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        var candidates = new List<string>();
        string? envLib = Environment.GetEnvironmentVariable("MODELSHARP_CUDA_LIB");
        if (!string.IsNullOrEmpty(envLib)) candidates.Add(envLib);

        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && !string.IsNullOrEmpty(dir); i++)
        {
            candidates.Add(Path.Combine(dir, "libms_cuda.so"));
            candidates.Add(Path.Combine(dir, "native", "build", "libms_cuda.so"));
            dir = Path.GetDirectoryName(dir.TrimEnd(Path.DirectorySeparatorChar));
        }

        foreach (var c in candidates)
            if (File.Exists(c) && NativeLibrary.TryLoad(c, out var h)) return h;
        return NativeLibrary.TryLoad("libms_cuda.so", out var g) ? g : IntPtr.Zero;
    }

    // ---- raw imports ----

    [DllImport(Lib)] private static extern int ms_cuda_available();
    [DllImport(Lib)] private static extern IntPtr ms_cuda_device_name();
    [DllImport(Lib)] private static extern unsafe void ms_cuda_sgemm(float* a, float* b, float* c, int M, int N, int K);
    [DllImport(Lib)] private static extern IntPtr ms_cuda_alloc(nuint nFloats);
    [DllImport(Lib)] private static extern unsafe IntPtr ms_cuda_upload(float* host, int rows, int cols);
    [DllImport(Lib)] private static extern unsafe int ms_cuda_upload_into(IntPtr dev, float* host, nuint nFloats);
    [DllImport(Lib)] private static extern unsafe int ms_cuda_download(IntPtr dev, float* host, nuint nFloats);
    [DllImport(Lib)] private static extern void ms_cuda_free(IntPtr dev);
    [DllImport(Lib)] private static extern int ms_cuda_sgemm_dev(IntPtr dA, IntPtr dB, IntPtr dC, int M, int N, int K);
    [DllImport(Lib)] private static extern unsafe void ms_cuda_sgemm_with_resident_b(float* a, IntPtr dB, float* c, int M, int N, int K);
    [DllImport(Lib)] private static extern int ms_cuda_sync();
    [DllImport(Lib)] private static extern void ms_cuda_shutdown();
    [DllImport(Lib)] private static extern int ms_cuda_sgemm_ctx(IntPtr cuContext, IntPtr cuStream, IntPtr dA, IntPtr dB, IntPtr dC, int M, int N, int K);
    [DllImport(Lib)] private static extern int ms_cuda_sync_stream(IntPtr cuStream);
    [DllImport(Lib)] private static extern int ms_cuda_sgemm_strided_batched_ctx(IntPtr cuContext, IntPtr cuStream, IntPtr dA, IntPtr dB, IntPtr dC, int M, int N, int K, long strideA, long strideB, long strideC, int batch);
    [DllImport(Lib)] private static extern void ms_cuda_release_context(IntPtr cuContext);
    [DllImport(Lib)] private static extern void ms_cuda_set_tf32(int enable);
    [DllImport(Lib)] private static extern int ms_cuda_get_tf32();
}
