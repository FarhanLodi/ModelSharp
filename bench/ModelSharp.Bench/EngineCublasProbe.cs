using System;
using System.Collections.Generic;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Bench;

/// <summary>
/// Validates the exact primitives the IlgpuEngine cuBLAS seam relies on: extract ILGPU's driver
/// CUcontext (Accelerator.NativePtr) and a buffer's device pointer (LoadEffectiveAddressAsPtr),
/// then run cuBLAS on those resident ILGPU buffers via NativeCuda.SgemmCtx — verifying correctness
/// and that the context-aware path actually executes (status 0), not a silent fallback.
/// </summary>
internal static class EngineCublasProbe
{
    public static void Run()
    {
        RunPrimitiveProbe();
        RunResidencyBenchmark();
    }

    private static void RunPrimitiveProbe()
    {
        Console.WriteLine();
        Console.WriteLine("=== ILGPU+cuBLAS resident-context probe ===");
        if (!NativeCuda.Available) { Console.WriteLine("NativeCuda unavailable — skip."); return; }

        using var ctx = Context.Create(b => b.Cuda());
        var cudaDev = ctx.GetCudaDevices().Count > 0 ? ctx.GetCudaDevice(0) : null;
        if (cudaDev == null) { Console.WriteLine("No CUDA device in ILGPU — skip."); return; }
        using Accelerator acc = cudaDev.CreateAccelerator(ctx);
        Console.WriteLine($"ILGPU accel: {acc.Name}  type={acc.AcceleratorType}");

        const int M = 1024, N = 1024, K = 1024;
        var rng = new Random(3);
        var ha = new float[M * K]; var hb = new float[K * N];
        for (int i = 0; i < ha.Length; i++) ha[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < hb.Length; i++) hb[i] = (float)(rng.NextDouble() * 2 - 1);

        using var da = acc.Allocate1D(ha);   // resident ILGPU buffers
        using var db = acc.Allocate1D(hb);
        using var dy = acc.Allocate1D<float>(M * N);
        acc.Synchronize();

        IntPtr cuCtx = acc.NativePtr;
        IntPtr strm = (acc.DefaultStream as CudaStream)?.StreamPtr ?? IntPtr.Zero;
        IntPtr pA = da.View.BaseView.LoadEffectiveAddressAsPtr();
        IntPtr pB = db.View.BaseView.LoadEffectiveAddressAsPtr();
        IntPtr pY = dy.View.BaseView.LoadEffectiveAddressAsPtr();
        Console.WriteLine($"CUcontext={cuCtx:X}  stream={strm:X}  dA={pA:X}");

        long before = NativeCuda.CtxCalls;
        int rc = NativeCuda.SgemmCtx(cuCtx, strm, pA, pB, pY, M, N, K);
        NativeCuda.SyncStream(strm);
        var hy = dy.GetAsArray1D();   // read back through ILGPU — proves same-context buffer

        // CPU reference
        double maxRel = ReferenceCheck(ha, hb, hy, M, N, K);
        Console.WriteLine($"SgemmCtx status={rc} (0=ok)  calls={NativeCuda.CtxCalls - before}  max rel err={maxRel:0.0e+0}");

        // timing of the resident-context kernel (no copies, ILGPU buffers reused)
        const int reps = 50;
        var sw = Stopwatch.StartNew();
        for (int r = 0; r < reps; r++) NativeCuda.SgemmCtx(cuCtx, strm, pA, pB, pY, M, N, K);
        NativeCuda.SyncStream(strm);
        sw.Stop();
        double msPer = sw.Elapsed.TotalMilliseconds / reps;
        double tflops = 2.0 * M * N * K / (msPer * 1e9);
        Console.WriteLine($"resident cuBLAS on ILGPU buffers: {msPer:0.000} ms/GEMM  {tflops:0.0} TFLOP/s");
        Console.WriteLine(rc == 0 && maxRel < 1e-2
            ? "=> cuBLAS runs IN ILGPU's context on resident buffers ✓"
            : "=> FAILED (status nonzero or wrong result)");
    }

    // End-to-end engine benchmark: a [K,N] weight as a graph INITIALIZER, repeated Run() with varying
    // [M,K] activations (decode/serving pattern). Compares re-upload-every-Run vs resident weight, and
    // the ILGPU kernel vs the cuBLAS context path — i.e. whether the GPU win actually lands end-to-end.
    private static void RunResidencyBenchmark()
    {
        Console.WriteLine();
        Console.WriteLine("=== engine weight-residency benchmark (weight = initializer) ===");
        if (!NativeCuda.Available) { Console.WriteLine("NativeCuda unavailable — skip."); return; }

        const int M = 8, K = 4096, N = 4096;   // small-batch decode/prefill against a 64 MB resident weight
        var rng = new Random(5);
        var w = new float[K * N];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);

        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "x", "w" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["w"] = new Tensor<float>(new TensorShape(new[] { K, N }), w) },
        };

        var x = new float[M * K];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        Dictionary<string, NamedTensor> Feeds() => new()
        {
            ["x"] = new NamedTensor("x", new Tensor<float>(new TensorShape(new[] { M, K }), (float[])x.Clone())),
        };

        Measure("re-upload weight + ILGPU kernel", cache: false, cublas: false, graph, Feeds);
        Measure("re-upload weight + cuBLAS      ", cache: false, cublas: true,  graph, Feeds);
        Measure("resident weight  + ILGPU kernel", cache: true,  cublas: false, graph, Feeds);
        Measure("resident weight  + cuBLAS      ", cache: true,  cublas: true,  graph, Feeds);
    }

    private static void Measure(string label, bool cache, bool cublas, ModelGraph graph,
                                Func<Dictionary<string, NamedTensor>> feeds)
    {
        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        if (!gpu.IsHardwareGpu) { Console.WriteLine($"  {label}: not hardware GPU — skip."); return; }
        gpu.CacheWeights = cache;
        gpu.UseCublas = cublas;

        for (int i = 0; i < 5; i++) gpu.Run(feeds());   // warmup (JIT, first weight upload)
        const int reps = 30;
        var samples = new double[reps];
        for (int r = 0; r < reps; r++)
        {
            var sw = Stopwatch.StartNew();
            gpu.Run(feeds());
            sw.Stop();
            samples[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        Console.WriteLine($"  {label} : {samples[reps / 2]:0.000} ms/Run (median)");
    }

    private static double ReferenceCheck(float[] a, float[] b, float[] y, int M, int N, int K)
    {
        double sumSq = 0; for (int i = 0; i < y.Length; i++) sumSq += (double)y[i] * y[i];
        double rms = Math.Sqrt(sumSq / y.Length) + 1e-9;
        double max = 0;
        var rng = new Random(11);
        for (int s = 0; s < 64; s++) // spot-check 64 random outputs
        {
            int m = rng.Next(M), n = rng.Next(N);
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)a[m * K + k] * b[k * N + n];
            max = Math.Max(max, Math.Abs(acc - y[m * N + n]) / rms);
        }
        return max;
    }
}
