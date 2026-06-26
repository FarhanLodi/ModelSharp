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
/// B4 — performance benchmark. Hardware-gated micro-benchmarks comparing the CUDA <see cref="IlgpuEngine"/>
/// against the managed CPU engine on a large MatMul and a Conv2D. Each is warmed up once, then timed over
/// several iterations; ms, GFLOP/s and the GPU/CPU speedup are logged via <see cref="ITestOutputHelper"/>.
/// Correctness is asserted (CUDA matches CPU within tolerance); timings are only LOGGED — no speedup is
/// asserted because hardware varies. Skips (green) when no hardware CUDA device is present.
/// </summary>
[Collection("CudaGpu")]
public class GpuPerfTests
{
    private readonly ITestOutputHelper _out;

    public GpuPerfTests(ITestOutputHelper output) => _out = output;

    private static Tensor<float> T(int[] dims, float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch
        {
            return false;
        }
    }

    private static float[] RandData(int n, int seed)
    {
        var rnd = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rnd.NextDouble() * 2 - 1);
        return data;
    }

    private static double TimeMs(Action run, int iters)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) run();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    [Fact]
    public void Cuda_MatMul_1024_Benchmark()
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine("Cuda_MatMul_1024_Benchmark: no CUDA device; skipping.");
            return;
        }

        const int M = 1024, K = 1024, N = 1024;
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("MatMul", "mm", new[] { "a", "b" }, new[] { "y" }) },
        };
        var feeds = new Dictionary<string, NamedTensor>
        {
            ["a"] = new NamedTensor("a", T(new[] { M, K }, RandData(M * K, 1))),
            ["b"] = new NamedTensor("b", T(new[] { K, N }, RandData(K * N, 2))),
        };

        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"expected hardware GPU, got '{gpu.AcceleratorName}'");
        using var cpu = new ManagedCpuEngine(graph);

        // Correctness on a sample of the output.
        float[] gOut = gpu.Run(feeds)["y"].Data.Span.ToArray();
        float[] cOut = cpu.Run(feeds)["y"].Data.Span.ToArray();
        Assert.Equal(cOut.Length, gOut.Length);
        var rnd = new Random(99);
        for (int s = 0; s < 200; s++)
        {
            int i = rnd.Next(cOut.Length);
            // K=1024 summed in fp32 → allow a relative tolerance.
            float tol = 1e-2f * Math.Max(1f, MathF.Abs(cOut[i]));
            Assert.True(MathF.Abs(cOut[i] - gOut[i]) < tol, $"y[{i}] cpu={cOut[i]} gpu={gOut[i]}");
        }

        const int iters = 10;
        gpu.Run(feeds);                       // warmup (kernel JIT + caches)
        double gpuMs = TimeMs(() => gpu.Run(feeds), iters);

        cpu.Run(feeds);                       // warmup
        double cpuMs = TimeMs(() => cpu.Run(feeds), 3);

        double flop = 2.0 * M * N * K;        // multiply-add per output element
        double gpuGflops = flop / (gpuMs * 1e6);
        double cpuGflops = flop / (cpuMs * 1e6);

        _out.WriteLine($"MatMul {M}x{K}x{N} on '{gpu.AcceleratorName}':");
        _out.WriteLine($"  GPU: {gpuMs,8:F3} ms  ({gpuGflops,8:F1} GFLOP/s)");
        _out.WriteLine($"  CPU: {cpuMs,8:F3} ms  ({cpuGflops,8:F1} GFLOP/s)");
        _out.WriteLine($"  speedup (CPU/GPU): {cpuMs / gpuMs:F1}x");
    }

    [Fact]
    public void Cuda_Conv2D_Benchmark()
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine("Cuda_Conv2D_Benchmark: no CUDA device; skipping.");
            return;
        }

        // 1x32x64x64 input, 64 output channels, 3x3 kernel, pad 1 → 1x64x64x64 output.
        const int N = 1, Cin = 32, H = 64, Wd = 64, Cout = 64, KH = 3, KW = 3;
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Conv", "c", new[] { "x", "w" }, new[] { "y" },
                    new Dictionary<string, object> { ["pads"] = new long[] { 1, 1, 1, 1 } }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["w"] = T(new[] { Cout, Cin, KH, KW }, RandData(Cout * Cin * KH * KW, 5)),
            },
        };
        var feeds = new Dictionary<string, NamedTensor>
        {
            ["x"] = new NamedTensor("x", T(new[] { N, Cin, H, Wd }, RandData(N * Cin * H * Wd, 6))),
        };

        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"expected hardware GPU, got '{gpu.AcceleratorName}'");
        using var cpu = new ManagedCpuEngine(graph);

        float[] gOut = gpu.Run(feeds)["y"].Data.Span.ToArray();
        float[] cOut = cpu.Run(feeds)["y"].Data.Span.ToArray();
        Assert.Equal(cOut.Length, gOut.Length);
        var rnd = new Random(7);
        for (int s = 0; s < 200; s++)
        {
            int i = rnd.Next(cOut.Length);
            float tol = 1e-2f * Math.Max(1f, MathF.Abs(cOut[i]));
            Assert.True(MathF.Abs(cOut[i] - gOut[i]) < tol, $"y[{i}] cpu={cOut[i]} gpu={gOut[i]}");
        }

        const int iters = 10;
        gpu.Run(feeds);
        double gpuMs = TimeMs(() => gpu.Run(feeds), iters);
        cpu.Run(feeds);
        double cpuMs = TimeMs(() => cpu.Run(feeds), 3);

        int outH = H, outW = Wd; // pad 1, stride 1, 3x3 → same size
        double flop = 2.0 * N * Cout * outH * outW * Cin * KH * KW;
        double gpuGflops = flop / (gpuMs * 1e6);
        double cpuGflops = flop / (cpuMs * 1e6);

        _out.WriteLine($"Conv2D N{N} Cin{Cin} {H}x{Wd} -> Cout{Cout} {KH}x{KW} on '{gpu.AcceleratorName}':");
        _out.WriteLine($"  GPU: {gpuMs,8:F3} ms  ({gpuGflops,8:F1} GFLOP/s)");
        _out.WriteLine($"  CPU: {cpuMs,8:F3} ms  ({cpuGflops,8:F1} GFLOP/s)");
        _out.WriteLine($"  speedup (CPU/GPU): {cpuMs / gpuMs:F1}x");
    }
}
