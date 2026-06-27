using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ModelSharp.Cpu;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// B5 — native <c>MatMulInteger</c> throughput benchmark. The native on-device int8 GEMM (the GPU engine's
/// <c>MatMulInteger</c> case → <see cref="GpuKernels.MatMulIntegerK"/>) is timed against the same op run on the
/// managed CPU engine — i.e. the slow double-precision <c>MatMulIntegerKernel</c> that the GPU engine used to fall
/// back to before this kernel existed. This is the GEMM hot-spot behind a quantized-LLM decode step.
///
/// <para>Correctness is asserted (the native int32 result is element-exact vs the CPU reference); the speedup is
/// only LOGGED — no ratio is hard-asserted because it is entirely hardware-dependent. CUDA-gated: skips (green)
/// when no hardware GPU is present.</para>
/// </summary>
[Collection("CudaGpu")]
public class GpuMatMulIntegerPerfTests
{
    private readonly ITestOutputHelper _out;

    public GpuMatMulIntegerPerfTests(ITestOutputHelper output) => _out = output;

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch { return false; }
    }

    private static double TimeMs(Action run, int iters)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) run();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildIntGemm(int M, int K, int N, int seed)
    {
        var rnd = new Random(seed);
        var aq = new byte[M * K];
        for (int i = 0; i < aq.Length; i++) aq[i] = (byte)rnd.Next(0, 256);
        var bq = new sbyte[K * N];
        for (int i = 0; i < bq.Length; i++) bq[i] = (sbyte)rnd.Next(-128, 128);

        var graph = new ModelGraph
        {
            Inputs = new[] { "A", "B" },
            Outputs = new[] { "Y" },
            Nodes = new[] { new GraphNode("MatMulInteger", "mmi", new[] { "A", "B", "az", "bz" }, new[] { "Y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["az"] = new Tensor<byte>(new TensorShape(), new byte[] { 128 }),
                ["bz"] = new Tensor<sbyte>(new TensorShape(), new sbyte[] { 0 }),
            },
        };
        var feeds = new Dictionary<string, NamedTensor>
        {
            ["A"] = new NamedTensor("A", new Tensor<byte>(new TensorShape(new[] { M, K }), aq)),
            ["B"] = new NamedTensor("B", new Tensor<sbyte>(new TensorShape(new[] { K, N }), bq)),
        };
        return (graph, feeds);
    }

    private void Benchmark(string what, int M, int K, int N, int seed, int gpuIters, int cpuIters)
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine($"{what}: no CUDA device; skipping.");
            return;
        }

        var (graph, feeds) = BuildIntGemm(M, K, N, seed);

        // Shared GPU box: skip cleanly (don't fail) if a co-tenant process holds VRAM so the large int8
        // buffers can't be allocated — same treatment as "no CUDA available".
        try
        {
            using var gpuTiled = new IlgpuEngine(graph, preferCpu: false);
            Assert.True(gpuTiled.IsHardwareGpu, $"{what}: expected hardware GPU, got '{gpuTiled.AcceleratorName}'");
            using var gpuNaive = new IlgpuEngine(graph, preferCpu: false) { UseNaiveIntGemm = true };
            using var cpu = new ManagedCpuEngine(graph);

            // Correctness: BOTH the tiled and the naïve int32 results must equal the CPU reference exactly,
            // and therefore each other (bit-exact int32 wraparound semantics).
            int[] cOut = ((Tensor<int>)cpu.Run(feeds)["Y"].Tensor).Span.ToArray();
            int[] tiledOut = ((Tensor<int>)gpuTiled.Run(feeds)["Y"].Tensor).Span.ToArray();
            int[] naiveOut = ((Tensor<int>)gpuNaive.Run(feeds)["Y"].Tensor).Span.ToArray();
            Assert.Equal(cOut, tiledOut);
            Assert.Equal(cOut, naiveOut);

            gpuTiled.Run(feeds);                              // warmup (kernel JIT)
            double tiledMs = TimeMs(() => gpuTiled.Run(feeds), gpuIters);
            gpuNaive.Run(feeds);                              // warmup
            double naiveMs = TimeMs(() => gpuNaive.Run(feeds), gpuIters);
            cpu.Run(feeds);                                   // warmup
            double cpuMs = TimeMs(() => cpu.Run(feeds), cpuIters);

            double gop = 2.0 * M * N * K / 1e9;               // multiply-add per output element
            _out.WriteLine($"{what}  MatMulInteger {M}x{K}x{N} on '{gpuTiled.AcceleratorName}':");
            _out.WriteLine($"  tiled GPU : {tiledMs,9:F3} ms  ({gop / (tiledMs / 1000.0),8:F1} GOP/s)");
            _out.WriteLine($"  naive GPU : {naiveMs,9:F3} ms  ({gop / (naiveMs / 1000.0),8:F1} GOP/s)");
            _out.WriteLine($"  CPU ref   : {cpuMs,9:F3} ms  ({gop / (cpuMs / 1000.0),8:F1} GOP/s)");
            _out.WriteLine($"  speedup tiled-vs-naive : {naiveMs / tiledMs:F2}x");
            _out.WriteLine($"  speedup tiled-vs-CPUref: {cpuMs / tiledMs:F1}x");
        }
        catch (Exception ex) when (ex.Message.Contains("out of memory"))
        {
            _out.WriteLine($"{what}: GPU out of memory (co-tenant holding VRAM); skipping. [{ex.Message}]");
        }
    }

    /// <summary>Square 2048³ INT8 GEMM — a generic large-tile throughput point.</summary>
    [Fact]
    public void Cuda_MatMulInteger_2048_Benchmark()
        => Benchmark("Square2048", M: 2048, K: 2048, N: 2048, seed: 11, gpuIters: 10, cpuIters: 2);

    /// <summary>
    /// LLM-decode-shaped INT8 GEMM: a [seq=32, 4096] activation against a [4096, 4096] projection weight — the
    /// shape a 7B-class attention/MLP projection lowers to at INT8. This is the per-layer GEMM the native kernel
    /// must accelerate for a practical quantized run.
    /// </summary>
    [Fact]
    public void Cuda_MatMulInteger_Llm4096_Benchmark()
        => Benchmark("Llm[32,4096]x[4096,4096]", M: 32, K: 4096, N: 4096, seed: 23, gpuIters: 20, cpuIters: 2);
}
