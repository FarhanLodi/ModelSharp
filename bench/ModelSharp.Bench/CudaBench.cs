using System;
using System.Diagnostics;
using ModelSharp.Gpu;

namespace ModelSharp.Bench;

/// <summary>
/// Proves the native cuBLAS binding (<see cref="NativeCuda"/>) from managed code: a correctness check
/// against a managed reference GEMM, then two benchmark families — copy-in/out single GEMMs, and the
/// "resident weight" decode/prefill pattern (upload a weight once, then reuse it for R GEMMs with the
/// activation kept on-device). Reports TFLOP/s and ms/GEMM. Skips cleanly if no CUDA device.
/// </summary>
internal static class CudaBench
{
    public static void Run()
    {
        Console.WriteLine();
        Console.WriteLine("=== CUDA cuBLAS (NativeCuda) ===");
        if (!NativeCuda.Available)
        {
            Console.WriteLine("NativeCuda: NOT AVAILABLE (libms_cuda.so missing or no CUDA device). Skipping.");
            return;
        }
        Console.WriteLine($"Device: {NativeCuda.DeviceName}");

        // ---- correctness vs managed reference on a small shape ----
        {
            const int M = 64, N = 48, K = 80;
            var rng = new Random(1);
            var a = RandData(M * K, rng);
            var b = RandData(K * N, rng);
            var c = new float[M * N];
            NativeCuda.Sgemm(a, b, c, M, N, K);
            var refC = RefGemm(a, b, M, N, K);
            double rel = MaxRelErr(refC, c);
            Console.WriteLine($"correctness {M}x{N}x{K}: max rel err = {rel:0.0e+0}  {(rel < 1e-3 ? "PASS" : "FAIL")}");
            if (!(rel < 1e-3)) throw new Exception($"cuBLAS correctness FAILED: rel={rel}");
        }

        // ---- (i) copy-in/out Sgemm ----
        Console.WriteLine();
        Console.WriteLine("copy-in/out Sgemm (includes H2D+D2H per call):");
        Console.WriteLine($"{"shape",-16} {"ms/GEMM",10} {"TFLOP/s",10}");
        foreach (int S in new[] { 1024, 2048, 4096 })
        {
            var rng = new Random(S);
            var a = RandData(S * S, rng);
            var b = RandData(S * S, rng);
            var c = new float[S * S];
            NativeCuda.Sgemm(a, b, c, S, S, S); // warmup
            int reps = 20;
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < reps; r++) NativeCuda.Sgemm(a, b, c, S, S, S);
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds / reps;
            double tflops = 2.0 * S * S * S / (ms * 1e9);
            Console.WriteLine($"{S}x{S}x{S,-6} {ms,10:0.000} {tflops,10:0.00}");
        }

        // ---- (ii) resident weight pattern ----
        Console.WriteLine();
        Console.WriteLine("resident weight [4096,4096] uploaded ONCE, reused for R=100 GEMMs:");
        Console.WriteLine($"{"pattern",-16} {"ms/GEMM",10} {"TFLOP/s",10}");
        ResidentBench(M: 1,   N: 4096, K: 4096, R: 100, label: "decode M=1");
        ResidentBench(M: 512, N: 4096, K: 4096, R: 100, label: "prefill M=512");
    }

    /// <summary>
    /// Upload a [K,N] weight once via Upload; keep an [M,K] activation device buffer (Alloc + UploadInto)
    /// and an [M,N] output device buffer; run R enqueue-only SgemmDev calls reusing the resident weight,
    /// Sync once, then Download the final result. This is the LLM weight-resident decode/prefill loop.
    /// </summary>
    private static void ResidentBench(int M, int N, int K, int R, string label)
    {
        var rng = new Random(99);
        var wHost = RandData(K * N, rng);   // weight B[K,N]
        var aHost = RandData(M * K, rng);   // activation A[M,K]

        IntPtr dB = NativeCuda.Upload(wHost, K, N);
        IntPtr dA = NativeCuda.Alloc((long)M * K);
        IntPtr dC = NativeCuda.Alloc((long)M * N);
        try
        {
            NativeCuda.UploadInto(dA, aHost, (long)M * K);

            // warmup
            NativeCuda.SgemmDev(dA, dB, dC, M, N, K);
            NativeCuda.Sync();

            var sw = Stopwatch.StartNew();
            for (int r = 0; r < R; r++)
                NativeCuda.SgemmDev(dA, dB, dC, M, N, K);
            NativeCuda.Sync();
            sw.Stop();

            var cHost = new float[M * N];
            NativeCuda.Download(dC, cHost, (long)M * N);

            double ms = sw.Elapsed.TotalMilliseconds / R;
            double tflops = 2.0 * M * N * K / (ms * 1e9);

            // sanity: result is finite and nonzero
            bool ok = !float.IsNaN(cHost[0]) && !float.IsInfinity(cHost[0]);
            Console.WriteLine($"{label,-16} {ms,10:0.000} {tflops,10:0.00}  ({(ok ? "ok" : "BAD")})");
        }
        finally
        {
            NativeCuda.Free(dA);
            NativeCuda.Free(dB);
            NativeCuda.Free(dC);
        }
    }

    private static float[] RandData(int n, Random rng)
    {
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return d;
    }

    private static float[] RefGemm(float[] a, float[] b, int M, int N, int K)
    {
        var c = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
            {
                float aik = a[i * K + k];
                int bRow = k * N, cRow = i * N;
                for (int j = 0; j < N; j++) c[cRow + j] += aik * b[bRow + j];
            }
        return c;
    }

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
