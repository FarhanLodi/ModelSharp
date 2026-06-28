using System;
using System.Runtime.InteropServices;

namespace ModelSharp.Bench;

/// <summary>
/// P/Invoke bindings to the native kernel library (native/build/libms_kernels.so).
/// Mirrors native/ms_kernels.h exactly. Loaded lazily; <see cref="Available"/> is
/// false (with no throw) when the .so has not been built, so the bench can still
/// report the managed baseline alone.
/// </summary>
internal static class NativeKernels
{
    private const string Lib = "ms_kernels"; // resolved via NativeLibrary probing below

    static NativeKernels()
    {
        NativeLibrary.SetDllImportResolver(typeof(NativeKernels).Assembly, (name, asm, path) =>
        {
            if (name != Lib) return IntPtr.Zero;
            foreach (var cand in new[]
            {
                "native/build/libms_kernels.so",
                "../native/build/libms_kernels.so",
                "../../native/build/libms_kernels.so",
                "../../../native/build/libms_kernels.so",
                "../../../../native/build/libms_kernels.so",
            })
            {
                if (NativeLibrary.TryLoad(System.IO.Path.GetFullPath(cand), out var h)) return h;
            }
            return NativeLibrary.TryLoad("libms_kernels.so", out var g) ? g : IntPtr.Zero;
        });
    }

    public static bool Available
    {
        get { try { _ = BuildInfo(); return true; } catch { return false; } }
    }

    public static string BuildInfo() => Marshal.PtrToStringAnsi(ms_build_info()) ?? "?";

    [DllImport(Lib, EntryPoint = "ms_build_info")] private static extern IntPtr ms_build_info();
    [DllImport(Lib, EntryPoint = "ms_has_avx512")] public static extern int HasAvx512();
    [DllImport(Lib, EntryPoint = "ms_has_vnni")] public static extern int HasVnni();

    [DllImport(Lib, EntryPoint = "ms_sgemm_f32")]
    public static extern unsafe void Sgemm(float* a, float* b, float* c, int m, int n, int k);

    [DllImport(Lib, EntryPoint = "ms_qgemm_nbits")]
    public static extern unsafe void QGemmNBits(float* a, byte* bq, float* scales, byte* zeroPoints,
        float* y, int m, int n, int k, int bits, int blockSize);

    [DllImport(Lib, EntryPoint = "ms_attention_f32")]
    public static extern unsafe void Attention(float* q, float* k, float* v, float* outp,
        int bh, int sq, int sk, int d, float scale, int causal);

    [DllImport(Lib, EntryPoint = "ms_conv2d_f32")]
    public static extern unsafe void Conv2d(float* x, float* w, float* bias, float* y,
        int n, int cin, int h, int w_, int cout, int kh, int kw,
        int strideH, int strideW, int padH, int padW, int dilH, int dilW, int groups);
}
