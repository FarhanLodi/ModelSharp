using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// B5 completion — a WHOLE GPT-2-style decoder layer composed on the GPU through the persistent on-device
/// KV-cache (<see cref="IlgpuEngine.DecodeLayerStep"/>): pre-LN → fused QKV projection → per-head split →
/// cache append → scaled-dot-product attention over the cached prefix → output projection → residual → pre-LN
/// → GELU MLP → residual. Each step is validated against a hand-written CPU reference (using the engine's own
/// A&amp;S Erf-based GELU so the activation matches bit-for-bit to tolerance). Also exercises the seq-0 /
/// zero-length-past case (the first decode step starts from an empty cache). Shares the serialized
/// <c>CudaGpu</c> collection and skips cleanly without CUDA, mirroring <see cref="GpuLlmTests"/>.
/// </summary>
[Collection("CudaGpu")]
public class GpuDecoderLayerTests
{
    private readonly ITestOutputHelper _out;
    public GpuDecoderLayerTests(ITestOutputHelper output) => _out = output;

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch { return false; }
    }

    private static Tensor<float> Rand(int[] dims, int seed, float lo = -0.5f, float hi = 0.5f)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (a, d) => a * d);
        float[] data = Enumerable.Range(0, n).Select(_ => lo + (float)rnd.NextDouble() * (hi - lo)).ToArray();
        return Tensor<float>.FromArray(new TensorShape(dims), data);
    }

    // --- CPU reference matching the engine's kernels exactly ---

    /// <summary>Same A&amp;S 7.1.26 Erf the GPU GeluK uses, so GELU matches.</summary>
    private static float Erf(float x)
    {
        float sign = x < 0f ? -1f : 1f;
        float ax = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * ax);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f)
                  * t * MathF.Exp(-ax * ax);
        return sign * y;
    }

    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x * 0.70710678f));

    private static float[] LayerNorm(float[] x, int rows, int n, float[] scale, float[] bias, float eps)
    {
        var y = new float[rows * n];
        for (int r = 0; r < rows; r++)
        {
            int b = r * n;
            float mean = 0f; for (int i = 0; i < n; i++) mean += x[b + i]; mean /= n;
            float vs = 0f; for (int i = 0; i < n; i++) { float d = x[b + i] - mean; vs += d * d; }
            float inv = 1f / MathF.Sqrt(vs / n + eps);
            for (int i = 0; i < n; i++) y[b + i] = (x[b + i] - mean) * inv * scale[i] + bias[i];
        }
        return y;
    }

    /// <summary>y[rows,N] = x[rows,K]·W[K,N] + b[N].</summary>
    private static float[] Linear(float[] x, int rows, int K, float[] w, int N, float[] b)
    {
        var y = new float[rows * N];
        for (int r = 0; r < rows; r++)
        for (int n = 0; n < N; n++)
        {
            float s = b[n];
            for (int k = 0; k < K; k++) s += x[r * K + k] * w[k * N + n];
            y[r * N + n] = s;
        }
        return y;
    }

    /// <summary>
    /// Full CPU reference for one decoder layer over the accumulated K/V history. <paramref name="histK"/>/
    /// <paramref name="histV"/> are the per-head cached prefixes [H, past, D]; this step's K/V are appended.
    /// </summary>
    private static float[] DecoderLayerCpu(
        float[] hidden, int S, int E, int H, int D,
        float[] wqkv, float[] bqkv, float[] wo, float[] bo,
        float[] wfc, float[] bfc, float[] wproj, float[] bproj,
        float[] ln1g, float[] ln1b, float[] ln2g, float[] ln2b, float eps,
        List<float[]> histK, List<float[]> histV) // each entry [H,1,D] head-major for one past token
    {
        int hidWidth = bfc.Length;
        float[] ln1 = LayerNorm(hidden, S, E, ln1g, ln1b, eps);
        float[] qkv = Linear(ln1, S, E, wqkv, 3 * E, bqkv); // [S,3E]

        // Split into Q/K/V [S,E] then per-head [H,S,D].
        // attention: for each head, scores over (past + S) keys.
        int past = histK.Count;
        int total = past + S;
        float invScale = 1f / MathF.Sqrt(D);
        var ctx = new float[H * S * D]; // [H,S,D]

        for (int h = 0; h < H; h++)
        {
            // assemble K_h, V_h [total, D]
            var kH = new float[total * D];
            var vH = new float[total * D];
            for (int t = 0; t < past; t++)
            for (int d = 0; d < D; d++)
            {
                kH[t * D + d] = histK[t][h * D + d];
                vH[t * D + d] = histV[t][h * D + d];
            }
            for (int s = 0; s < S; s++)
            for (int d = 0; d < D; d++)
            {
                kH[(past + s) * D + d] = qkv[s * (3 * E) + 1 * E + h * D + d];
                vH[(past + s) * D + d] = qkv[s * (3 * E) + 2 * E + h * D + d];
            }

            for (int s = 0; s < S; s++)
            {
                var scores = new float[total];
                for (int j = 0; j < total; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < D; d++)
                        dot += qkv[s * (3 * E) + 0 * E + h * D + d] * kH[j * D + d];
                    scores[j] = dot * invScale;
                }
                float mx = scores[0]; for (int j = 1; j < total; j++) if (scores[j] > mx) mx = scores[j];
                float sum = 0f; for (int j = 0; j < total; j++) { scores[j] = MathF.Exp(scores[j] - mx); sum += scores[j]; }
                for (int j = 0; j < total; j++) scores[j] /= sum;
                for (int d = 0; d < D; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < total; j++) acc += scores[j] * vH[j * D + d];
                    ctx[(h * S + s) * D + d] = acc;
                }
            }
        }

        // ctx [H,S,D] -> [S,E]
        var ctxSE = new float[S * E];
        for (int s = 0; s < S; s++)
        for (int h = 0; h < H; h++)
        for (int d = 0; d < D; d++)
            ctxSE[s * E + h * D + d] = ctx[(h * S + s) * D + d];

        float[] attnOut = Linear(ctxSE, S, E, wo, E, bo);
        var h2 = new float[S * E];
        for (int i = 0; i < S * E; i++) h2[i] = hidden[i] + attnOut[i];

        float[] ln2 = LayerNorm(h2, S, E, ln2g, ln2b, eps);
        float[] fc = Linear(ln2, S, E, wfc, hidWidth, bfc);
        var act = new float[fc.Length];
        for (int i = 0; i < fc.Length; i++) act[i] = Gelu(fc[i]);
        float[] mlp = Linear(act, S, hidWidth, wproj, E, bproj);
        var outp = new float[S * E];
        for (int i = 0; i < S * E; i++) outp[i] = h2[i] + mlp[i];
        return outp;
    }

    [Fact]
    public void Cuda_DecodeLayerStep_MultiStep_Matches_Cpu()
    {
        // Small but real layer shape: H heads × D dim = E embed; MLP hidden = 4E (GPT-2 ratio).
        const int H = 3, D = 4, S = 1, steps = 4, maxSeq = 8;
        const int E = H * D;          // 12
        const int Hid = 4 * E;        // 48
        const float eps = 1e-5f;

        if (!HardwareGpuAvailable())
        {
            _out.WriteLine("DecodeLayerStep: no CUDA device; skipping.");
            return;
        }

        // Weights (shared across steps).
        float[] wqkv = Rand(new[] { E, 3 * E }, 1).Span.ToArray();
        float[] bqkv = Rand(new[] { 3 * E }, 2).Span.ToArray();
        float[] wo = Rand(new[] { E, E }, 3).Span.ToArray();
        float[] bo = Rand(new[] { E }, 4).Span.ToArray();
        float[] wfc = Rand(new[] { E, Hid }, 5).Span.ToArray();
        float[] bfc = Rand(new[] { Hid }, 6).Span.ToArray();
        float[] wproj = Rand(new[] { Hid, E }, 7).Span.ToArray();
        float[] bproj = Rand(new[] { E }, 8).Span.ToArray();
        float[] ln1g = Rand(new[] { E }, 9, 0.8f, 1.2f).Span.ToArray();
        float[] ln1b = Rand(new[] { E }, 10).Span.ToArray();
        float[] ln2g = Rand(new[] { E }, 11, 0.8f, 1.2f).Span.ToArray();
        float[] ln2b = Rand(new[] { E }, 12).Span.ToArray();

        var w = new IlgpuEngine.DecoderLayerWeights
        {
            Ln1Scale = Tensor<float>.FromArray(new TensorShape(E), ln1g),
            Ln1Bias = Tensor<float>.FromArray(new TensorShape(E), ln1b),
            QkvWeight = Tensor<float>.FromArray(new TensorShape(E, 3 * E), wqkv),
            QkvBias = Tensor<float>.FromArray(new TensorShape(3 * E), bqkv),
            OutWeight = Tensor<float>.FromArray(new TensorShape(E, E), wo),
            OutBias = Tensor<float>.FromArray(new TensorShape(E), bo),
            Ln2Scale = Tensor<float>.FromArray(new TensorShape(E), ln2g),
            Ln2Bias = Tensor<float>.FromArray(new TensorShape(E), ln2b),
            FcWeight = Tensor<float>.FromArray(new TensorShape(E, Hid), wfc),
            FcBias = Tensor<float>.FromArray(new TensorShape(Hid), bfc),
            ProjWeight = Tensor<float>.FromArray(new TensorShape(Hid, E), wproj),
            ProjBias = Tensor<float>.FromArray(new TensorShape(E), bproj),
            Epsilon = eps,
        };

        // Engine only needs a trivial graph; we use the DecodeLayerStep seam + cache directly.
        var graph = new ModelGraph
        {
            Inputs = new[] { "q" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Relu", "noop", new[] { "q" }, new[] { "y" }) },
        };
        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"expected hardware GPU, got '{gpu.AcceleratorName}'.");
        _out.WriteLine($"DecodeLayerStep: hardware GPU '{gpu.AcceleratorName}' (IsHardwareGpu=true).");

        using GpuKvCache cache = gpu.CreateKvCache(H, maxSeq, D);

        // CPU-side accumulation of the cached per-head K/V (each token [H,D] head-major), recomputed from
        // the same QKV projection of each step's hidden so the reference attends over the identical prefix.
        var histK = new List<float[]>();
        var histV = new List<float[]>();

        for (int s = 0; s < steps; s++)
        {
            float[] hidden = Rand(new[] { S, E }, 100 + s).Span.ToArray();

            // GPU: whole layer through the on-device cache (seq-0 path on the very first step).
            Tensor<float> gpuOut = gpu.DecodeLayerStep(cache, Tensor<float>.FromArray(new TensorShape(S, E), hidden), w);
            Assert.Equal(s + 1, cache.SeqLen); // cache grew by exactly one token (append path, no realloc)

            // CPU reference for this step over the current history.
            float[] cpuOut = DecoderLayerCpu(hidden, S, E, H, D, wqkv, bqkv, wo, bo, wfc, bfc, wproj, bproj,
                ln1g, ln1b, ln2g, ln2b, eps, histK, histV);

            // Append this step's K/V to the CPU history (re-derive from the same QKV projection).
            float[] ln1 = LayerNorm(hidden, S, E, ln1g, ln1b, eps);
            float[] qkv = Linear(ln1, S, E, wqkv, 3 * E, bqkv);
            var kTok = new float[H * D];
            var vTok = new float[H * D];
            for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++)
            {
                kTok[h * D + d] = qkv[1 * E + h * D + d];
                vTok[h * D + d] = qkv[2 * E + h * D + d];
            }
            histK.Add(kTok);
            histV.Add(vTok);

            float[] g = gpuOut.Span.ToArray();
            Assert.Equal(cpuOut.Length, g.Length);
            float maxDiff = 0f;
            for (int i = 0; i < cpuOut.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(cpuOut[i] - g[i]));
            Assert.True(maxDiff < 1e-3f, $"DecodeLayerStep step {s}: max|Δ|={maxDiff} >= 1e-3");
            _out.WriteLine($"  step {s}: cache.SeqLen={cache.SeqLen}, layer output matches CPU (max|Δ|={maxDiff:E2}).");
        }

        cache.Reset();
        Assert.Equal(0, cache.SeqLen);
    }
}
