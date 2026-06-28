using System;
using System.Diagnostics;
using ModelSharp.Cpu.Kernels.Linear;

namespace ModelSharp.Bench;

/// <summary>
/// Head-to-head: ModelSharp's managed BlockedGemm vs the native AVX-512 kernel
/// (native/build/libms_kernels.so), same shapes, parity-checked. This is the
/// number that decides whether the native rewrite is worth it.
/// </summary>
internal static class Program
{
    private static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "probe") { EngineCublasProbe.Run(); return; }

        // The "managed" column must be genuinely managed: disable the native seam that
        // BlockedGemm.Multiply now uses, so we measure managed vs native cleanly. The "native"
        // column calls the native kernel directly via NativeKernels.
        ModelSharp.Native.NativeGemm.Enabled = false;

        Console.WriteLine($"Cores: {Environment.ProcessorCount}   ServerGC: {System.Runtime.GCSettings.IsServerGC}");
        if (NativeKernels.Available)
            Console.WriteLine($"Native: {NativeKernels.BuildInfo()}  AVX512={NativeKernels.HasAvx512()}  VNNI={NativeKernels.HasVnni()}");
        else
            Console.WriteLine("Native: NOT FOUND (run `make` in native/). Showing managed only.");
        Console.WriteLine();

        (int M, int N, int K)[] shapes =
        {
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (8, 4096, 4096),      // LLM MLP-ish (batch token x hidden)
            (1, 4096, 4096),      // decode GEMV
        };

        Console.WriteLine($"{"shape",-20} {"managed GF/s",13} {"native GF/s",13} {"speedup",9} {"max rel err",13}");
        Console.WriteLine(new string('-', 72));

        foreach (var (M, N, K) in shapes)
            RunSgemm(M, N, K);

        Console.WriteLine();
        Console.WriteLine("(GFLOP/s = 2*M*N*K / median-time. Both paths multi-threaded; set OMP_NUM_THREADS to vary native.)");

        VerifyWiring();

        CudaBench.Run();
    }

    // Proves the engine seam routes to native: with NativeGemm.Enabled, BlockedGemm.Multiply
    // (the conv/im2col entry) should run at native speed and match the managed result.
    private static void VerifyWiring()
    {
        Console.WriteLine();
        Console.WriteLine("=== engine wiring (BlockedGemm.Multiply seam) ===");
        const int M = 1024, N = 1024, K = 1024;
        var rng = new Random(7);
        var a = new float[M * K]; var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var yMan = new float[M * N]; var yNat = new float[M * N];
        double flops = 2.0 * M * N * K;

        ModelSharp.Native.NativeGemm.Enabled = false;
        BlockedGemm.Multiply(a, 0, K, b, 0, N, yMan, 0, N, M, N, K);
        double manMs = TimeMs(() => BlockedGemm.Multiply(a, 0, K, b, 0, N, yMan, 0, N, M, N, K));

        ModelSharp.Native.NativeGemm.Enabled = ModelSharp.Native.NativeGemm.Available;
        BlockedGemm.Multiply(a, 0, K, b, 0, N, yNat, 0, N, M, N, K);
        double natMs = TimeMs(() => BlockedGemm.Multiply(a, 0, K, b, 0, N, yNat, 0, N, M, N, K));

        Console.WriteLine($"NativeGemm.Available = {ModelSharp.Native.NativeGemm.Available}");
        Console.WriteLine($"BlockedGemm.Multiply managed : {flops / (manMs * 1e6):0} GF/s");
        Console.WriteLine($"BlockedGemm.Multiply wired   : {flops / (natMs * 1e6):0} GF/s  (max rel err {MaxRelErr(yMan, yNat):0.0e+0})");
        Console.WriteLine(flops / (natMs * 1e6) > flops / (manMs * 1e6) * 1.5
            ? "=> seam routes to native ✓"
            : "=> seam NOT accelerating (lib missing or disabled)");
    }

    private static void RunSgemm(int M, int N, int K)
    {
        var rng = new Random(12345);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var yManaged = new float[M * N];
        var yNative = new float[M * N];

        double flops = 2.0 * M * N * K;
        long reps = Math.Max(1, (long)(2e9 / flops)); // ~aim for a couple GFLOP of work per timed rep

        // ---- managed ----
        BlockedGemm.Multiply(a, 0, K, b, 0, N, yManaged, 0, N, M, N, K); // warmup
        double mManagedMs = TimeMs(() =>
        {
            for (long r = 0; r < reps; r++)
                BlockedGemm.Multiply(a, 0, K, b, 0, N, yManaged, 0, N, M, N, K);
        }) / reps;
        double managedGf = flops / (mManagedMs * 1e6);

        // ---- native ----
        double nativeGf = double.NaN, maxRel = double.NaN;
        if (NativeKernels.Available)
        {
            unsafe
            {
                fixed (float* pa = a, pb = b, py = yNative)
                {
                    NativeKernels.Sgemm(pa, pb, py, M, N, K); // warmup
                    var samples = new double[5];
                    for (int s = 0; s < samples.Length; s++)
                    {
                        var sw = Stopwatch.StartNew();
                        for (long r = 0; r < reps; r++)
                            NativeKernels.Sgemm(pa, pb, py, M, N, K);
                        sw.Stop();
                        samples[s] = sw.Elapsed.TotalMilliseconds;
                    }
                    Array.Sort(samples);
                    double mNativeMs = samples[samples.Length / 2] / reps;
                    nativeGf = flops / (mNativeMs * 1e6);
                }
            }
            maxRel = MaxRelErr(yManaged, yNative);
        }

        string speedup = double.IsNaN(nativeGf) ? "-" : $"{nativeGf / managedGf:0.00}x";
        string nat = double.IsNaN(nativeGf) ? "-" : $"{nativeGf:0}";
        string err = double.IsNaN(maxRel) ? "-" : $"{maxRel:0.0e+0}";
        Console.WriteLine($"{M}x{N}x{K,-12} {managedGf,13:0} {nat,13} {speedup,9} {err,13}");
    }

    private static double TimeMs(Action body)
    {
        // median of 5
        var samples = new double[5];
        for (int i = 0; i < samples.Length; i++)
        {
            var sw = Stopwatch.StartNew();
            body();
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    // Error relative to the output RMS magnitude — the standard GEMM metric. Per-element
    // |ref|+eps denominators flag harmless catastrophic-cancellation noise on near-zero outputs.
    private static double MaxRelErr(float[] reference, float[] test)
    {
        double sumSq = 0;
        for (int i = 0; i < reference.Length; i++) sumSq += (double)reference[i] * reference[i];
        double rms = Math.Sqrt(sumSq / reference.Length) + 1e-9;
        double max = 0;
        for (int i = 0; i < reference.Length; i++)
            max = Math.Max(max, Math.Abs(reference[i] - test[i]) / rms);
        return max;
    }
}
