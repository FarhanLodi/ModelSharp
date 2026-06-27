using System;
using System.Collections.Generic;
using System.Diagnostics;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Linear;
using ModelSharp.Cpu.Kernels.Quantize;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// Micro-benchmarks + correctness sanity for the multithreaded/SIMD CPU matmul kernels
/// (MatMul, MatMulNBits). These are not strict perf assertions (CI machines vary wildly); they
/// log timing for humans and assert the parallel/vectorized result still matches a plain serial
/// reference computed independently in-test. The existing op tests cover the full semantics; this
/// just guards that the fast path stays numerically faithful on a large, parallel-eligible shape.
/// </summary>
public class MatMulParallelPerfTests
{
    private readonly ITestOutputHelper _out;
    public MatMulParallelPerfTests(ITestOutputHelper output) => _out = output;

    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    [Fact]
    public void MatMul_Large_MatchesSerialReference_AndLogsTiming()
    {
        int M = 64, K = 512, N = 384;
        var rng = new Random(1234);
        var aData = new float[M * K];
        var bData = new float[K * N];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        var ctx = Ctx(
            ("A", Tensor<float>.FromArray(new TensorShape(M, K), aData)),
            ("B", Tensor<float>.FromArray(new TensorShape(K, N), bData)));
        var node = new GraphNode("MatMul", "n", new[] { "A", "B" }, new[] { "Y" },
            new Dictionary<string, object>());

        var sw = Stopwatch.StartNew();
        new MatMulKernel().Execute(node, ctx);
        sw.Stop();
        Tensor<float> y = ctx.Get("Y");
        _out.WriteLine($"MatMul [{M}x{K}]x[{K}x{N}] kernel: {sw.Elapsed.TotalMilliseconds:F3} ms");

        // Plain serial triple loop reference.
        var refY = new float[M * N];
        for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            float s = 0f;
            for (int k = 0; k < K; k++) s += aData[m * K + k] * bData[k * N + n];
            refY[m * N + n] = s;
        }

        ReadOnlySpan<float> got = y.Span;
        for (int i = 0; i < refY.Length; i++)
            Assert.True(MathF.Abs(refY[i] - got[i]) <= 1e-2f,
                $"index {i}: ref {refY[i]} got {got[i]}");
    }

    [Fact]
    public void MatMulNBits_Large_MatchesSerialReference_AndLogsTiming()
    {
        // bits=4, block_size=32. K=256 (8 blocks/row), N=512. Default symmetric zp = 8.
        int M = 32, K = 256, N = 512, blockSize = 32, bits = 4;
        int nBlocksPerRow = (K + blockSize - 1) / blockSize;
        int blobSize = (blockSize * bits + 7) / 8;          // 16 bytes per block
        int bRowBytes = nBlocksPerRow * blobSize;

        var rng = new Random(99);
        var aData = new float[M * K];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);

        var bBytes = new byte[N * bRowBytes];
        var scales = new float[N * nBlocksPerRow];
        var codes = new int[N * K];                          // keep the raw codes for the reference
        for (int n = 0; n < N; n++)
        {
            for (int blk = 0; blk < nBlocksPerRow; blk++)
                scales[n * nBlocksPerRow + blk] = 0.05f + 0.01f * ((n + blk) % 7);
            for (int k = 0; k < K; k++)
            {
                int q = rng.Next(0, 16);
                codes[n * K + k] = q;
                int blk = k / blockSize;
                int inBlk = k - blk * blockSize;
                int blobBase = n * bRowBytes + blk * blobSize;
                int byteOff = blobBase + (inBlk >> 1);
                if ((inBlk & 1) == 0) bBytes[byteOff] = (byte)((bBytes[byteOff] & 0xF0) | (q & 0xF));
                else bBytes[byteOff] = (byte)((bBytes[byteOff] & 0x0F) | ((q & 0xF) << 4));
            }
        }

        var ctx = Ctx(
            ("A", Tensor<float>.FromArray(new TensorShape(M, K), aData)),
            ("B", Tensor<byte>.FromArray(new TensorShape(N, nBlocksPerRow, blobSize), bBytes)),
            ("scales", Tensor<float>.FromArray(new TensorShape(N * nBlocksPerRow), scales)));
        var node = new GraphNode("MatMulNBits", "n", new[] { "A", "B", "scales" }, new[] { "Y" },
            new Dictionary<string, object>
            {
                ["N"] = (long)N, ["K"] = (long)K, ["bits"] = (long)bits, ["block_size"] = (long)blockSize,
            });

        var sw = Stopwatch.StartNew();
        new MatMulNBitsKernel().Execute(node, ctx);
        sw.Stop();
        Tensor<float> y = ctx.Get("Y");
        _out.WriteLine($"MatMulNBits [{M}x{K}]x[{N}x{K}] int4 kernel: {sw.Elapsed.TotalMilliseconds:F3} ms");

        // Serial reference: dequantize with default zp=8, then Y[m,n] = Σ A[m,k]·W[n,k].
        const float zp = 8f;
        var refY = new float[M * N];
        for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++)
        {
            float s = 0f;
            for (int k = 0; k < K; k++)
            {
                float scale = scales[n * nBlocksPerRow + (k / blockSize)];
                float w = (codes[n * K + k] - zp) * scale;
                s += aData[m * K + k] * w;
            }
            refY[m * N + n] = s;
        }

        ReadOnlySpan<float> got = y.Span;
        for (int i = 0; i < refY.Length; i++)
            Assert.True(MathF.Abs(refY[i] - got[i]) <= 1e-3f,
                $"index {i}: ref {refY[i]} got {got[i]}");
    }
}
