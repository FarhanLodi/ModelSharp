using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// GPU/CPU parity for the two heavyweight quantized-LLM contrib ops now running <b>natively on the device</b>:
/// <c>MatMulNBits</c> (block-wise INT4/INT8 weight-quantized matmul, dequantized inside the kernel) and
/// <c>GroupQueryAttention</c> (packed-QKV split, in-op rotary, repeat-KV causal attention, present-K/V — done
/// on-device). Each builds a small in-memory <see cref="ModelGraph"/> and asserts the <see cref="IlgpuEngine"/>
/// result matches the <see cref="ManagedCpuEngine"/> to a relative float tolerance (GPU float accumulation
/// differs slightly from the CPU reference, so parity is to ~1e-3 relative, not bit-exact).
///
/// <para>Two execution surfaces, mirroring <see cref="GpuQuantizedTests"/>:
/// <list type="bullet">
/// <item>The <c>[Fact]</c>/<c>[Theory]</c> tests run on ILGPU's <b>CPU accelerator</b> (<c>preferCpu: true</c>)
/// so parity is covered on every machine with no CUDA — the native kernel logic is identical on CUDA.</item>
/// <item>The nested <c>Cuda</c> class re-runs the same graphs on real <b>hardware CUDA</b> (asserts
/// <see cref="IlgpuEngine.IsHardwareGpu"/>), skipping cleanly when no CUDA device exists and skipping on a
/// device <c>out of memory</c> (shared GPU box).</item>
/// </list></para>
/// </summary>
public class GpuQuantizedLlmOpsTests
{
    private readonly ITestOutputHelper _out;
    public GpuQuantizedLlmOpsTests(ITestOutputHelper output) => _out = output;

    // ---- builders ------------------------------------------------------------------------------

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<byte> U8(int[] dims, byte[] data) => new(new TensorShape(dims), data);

    private static GraphNode N(string op, string name, string[] inputs, string[] outputs,
        Dictionary<string, object>? attrs = null) => new(op, name, inputs, outputs, attrs);

    private static Dictionary<string, NamedTensor> Feeds(params (string name, Tensor t)[] feeds) =>
        feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch { return false; }
    }

    /// <summary>Packs two 4-bit values into one byte (low nibble = lo, high nibble = hi).</summary>
    private static byte Pack4(int lo, int hi) => (byte)((lo & 0xF) | ((hi & 0xF) << 4));

    // ---- parity driver -------------------------------------------------------------------------

    /// <summary>
    /// Runs <paramref name="graph"/> on the ILGPU engine (CPU accelerator unless <paramref name="cuda"/>) and on
    /// the managed CPU engine and asserts every float output matches within a <paramref name="relTol"/> relative
    /// tolerance (absolute floor 1e-4 for near-zero values), so GPU vs CPU float-accumulation differences pass.
    /// </summary>
    private void AssertParity(string what, ModelGraph graph, Dictionary<string, NamedTensor> feeds,
        bool cuda, float relTol = 1e-3f)
    {
        using var gpu = new IlgpuEngine(graph, preferCpu: !cuda);
        if (cuda)
            Assert.True(gpu.IsHardwareGpu, $"{what}: expected a hardware GPU but got '{gpu.AcceleratorName}'.");
        _out.WriteLine($"{what}: accelerator='{gpu.AcceleratorName}' IsHardwareGpu={gpu.IsHardwareGpu}.");

        using var cpu = new ManagedCpuEngine(graph);
        IReadOnlyDictionary<string, NamedTensor> gpuOut = gpu.Run(feeds);
        IReadOnlyDictionary<string, NamedTensor> cpuOut = cpu.Run(feeds);

        foreach (string name in graph.Outputs)
        {
            Tensor g = gpuOut[name].Tensor;
            Tensor c = cpuOut[name].Tensor;
            Assert.Equal(c.Shape.Dimensions.ToArray(), g.Shape.Dimensions.ToArray());
            float[] ca = c.AsFloat().Span.ToArray(), ga = g.AsFloat().Span.ToArray();
            Assert.Equal(ca.Length, ga.Length);
            for (int i = 0; i < ca.Length; i++)
            {
                float tol = relTol * MathF.Max(1f, MathF.Abs(ca[i])) + 1e-4f;
                Assert.True(MathF.Abs(ca[i] - ga[i]) <= tol,
                    $"{what}:{name}[{i}] cpu={ca[i]} gpu={ga[i]} (tol={tol})");
            }
        }
    }

    // ============================================================================================
    //  MatMulNBits graph builders
    // ============================================================================================

    /// <summary>
    /// A bare <c>MatMulNBits</c> over random A and hand-packed quantized weights, parameterized over bits (4/8),
    /// block size, and zero-point form ("default" symmetric, "float" per-(row,block), or "packed" n-bit). The
    /// quantized codes are random in [0, 2^bits); scales are small positive floats. Output Y = A · dequant(B)ᵀ.
    /// </summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildMatMulNBits(
        int M, int K, int Ncols, int bits, int blockSize, string zpKind, int seed)
    {
        var rnd = new Random(seed);
        int nBlocksPerRow = (K + blockSize - 1) / blockSize;
        int blobSize = (blockSize * bits + 7) / 8;
        int zpRowBytes = (nBlocksPerRow * bits + 7) / 8;
        int maxCode = 1 << bits;

        // Random per-(row,k) quantized codes, then pack into [N, nBlocksPerRow, blobSize] uint8.
        int[,] codes = new int[Ncols, K];
        for (int n = 0; n < Ncols; n++)
            for (int k = 0; k < K; k++)
                codes[n, k] = rnd.Next(0, maxCode);

        var bPacked = new byte[Ncols * nBlocksPerRow * blobSize];
        for (int n = 0; n < Ncols; n++)
        {
            int rowBase = n * nBlocksPerRow * blobSize;
            for (int bk = 0; bk < nBlocksPerRow; bk++)
            {
                int blobBase = rowBase + bk * blobSize;
                int kStart = bk * blockSize;
                int kEnd = Math.Min(kStart + blockSize, K);
                for (int k = kStart; k < kEnd; k++)
                {
                    int inBlock = k - kStart;
                    int q = codes[n, k];
                    if (bits == 8) bPacked[blobBase + inBlock] = (byte)q;
                    else
                    {
                        int byteOff = blobBase + (inBlock >> 1);
                        int shift = (inBlock & 1) * 4;
                        bPacked[byteOff] |= (byte)((q & 0xF) << shift);
                    }
                }
            }
        }

        // Scales (positive). One per (row, block).
        var scales = new float[Ncols * nBlocksPerRow];
        for (int i = 0; i < scales.Length; i++) scales[i] = 0.01f + 0.05f * (float)rnd.NextDouble();

        string[] inputs;
        var inits = new Dictionary<string, Tensor>
        {
            ["B"] = U8(new[] { Ncols, nBlocksPerRow, blobSize }, bPacked),
            ["scales"] = new Tensor<float>(new TensorShape(scales.Length), scales),
        };
        switch (zpKind)
        {
            case "default":
                inputs = new[] { "A", "B", "scales" };
                break;
            case "float":
            {
                var zp = new float[Ncols * nBlocksPerRow];
                for (int i = 0; i < zp.Length; i++) zp[i] = (1 << (bits - 1)) + rnd.Next(-2, 3);
                inits["zp"] = new Tensor<float>(new TensorShape(zp.Length), zp);
                inputs = new[] { "A", "B", "scales", "zp" };
                break;
            }
            case "packed":
            {
                // Packed n-bit zero points, row stride zpRowBytes, same nibble packing as B.
                var zpPacked = new byte[Ncols * zpRowBytes];
                for (int n = 0; n < Ncols; n++)
                {
                    int zpRowBase = n * zpRowBytes;
                    for (int bk = 0; bk < nBlocksPerRow; bk++)
                    {
                        int z = (1 << (bits - 1)) + rnd.Next(-2, 3);
                        if (bits == 8) zpPacked[zpRowBase + bk] = (byte)z;
                        else
                        {
                            int byteOff = zpRowBase + (bk >> 1);
                            int shift = (bk & 1) * 4;
                            zpPacked[byteOff] |= (byte)((z & 0xF) << shift);
                        }
                    }
                }
                inits["zp"] = U8(new[] { Ncols, zpRowBytes }, zpPacked);
                inputs = new[] { "A", "B", "scales", "zp" };
                break;
            }
            default:
                throw new ArgumentException(zpKind);
        }

        var graph = new ModelGraph
        {
            Inputs = new[] { "A" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                N("MatMulNBits", "mmnb", inputs, new[] { "Y" },
                    new Dictionary<string, object>
                    {
                        ["N"] = (long)Ncols, ["K"] = (long)K, ["bits"] = (long)bits, ["block_size"] = (long)blockSize,
                    }),
            },
            Initializers = inits,
        };

        var aData = new float[M * K];
        for (int i = 0; i < aData.Length; i++) aData[i] = -1f + 2f * (float)rnd.NextDouble();
        var feeds = Feeds(("A", F(new[] { M, K }, aData)));
        return (graph, feeds);
    }

    // ============================================================================================
    //  GroupQueryAttention graph builders
    // ============================================================================================

    private static float[] RandF(int n, Random rnd, float lo = -1f, float hi = 1f)
    {
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = lo + (hi - lo) * (float)rnd.NextDouble();
        return d;
    }

    /// <summary>Packed-QKV GQA with optional in-op rotary; output + present_key/present_value asserted vs CPU.</summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildGqaPacked(
        int batch, int s, int numHeads, int kvNumHeads, int headDim, bool rotary, int seed)
    {
        var rnd = new Random(seed);
        int unit = numHeads + 2 * kvNumHeads;
        int hidden = unit * headDim;
        var qkv = RandF(batch * s * hidden, rnd);

        var inits = new Dictionary<string, Tensor>();
        string[] inputs;
        var attrs = new Dictionary<string, object>
        {
            ["num_heads"] = (long)numHeads, ["kv_num_heads"] = (long)kvNumHeads,
        };
        if (rotary)
        {
            int half = headDim / 2;
            int maxPos = s + 2;
            var cos = new float[maxPos * half];
            var sin = new float[maxPos * half];
            for (int p = 0; p < maxPos; p++)
                for (int j = 0; j < half; j++)
                {
                    float theta = p / MathF.Pow(10000f, 2f * j / headDim);
                    cos[p * half + j] = MathF.Cos(theta);
                    sin[p * half + j] = MathF.Sin(theta);
                }
            inits["cos"] = new Tensor<float>(new TensorShape(maxPos, half), cos);
            inits["sin"] = new Tensor<float>(new TensorShape(maxPos, half), sin);
            attrs["do_rotary"] = 1L;
            inputs = new[] { "qkv", "", "", "", "", "", "", "cos", "sin" };
        }
        else
        {
            inputs = new[] { "qkv" };
        }

        var graph = new ModelGraph
        {
            Inputs = new[] { "qkv" },
            Outputs = new[] { "out", "pk", "pv" },
            Nodes = new[] { N("GroupQueryAttention", "gqa", inputs, new[] { "out", "pk", "pv" }, attrs) },
            Initializers = inits,
        };
        var feeds = Feeds(("qkv", F(new[] { batch, s, hidden }, qkv)));
        return (graph, feeds);
    }

    /// <summary>Unpacked GQA decode-with-past: single new token attends over past + current, with seqlens_k.</summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildGqaDecodeWithPast(
        int batch, int numHeads, int kvNumHeads, int headDim, int pastSeq, int seed)
    {
        var rnd = new Random(seed);
        int qHid = numHeads * headDim;
        int kvHid = kvNumHeads * headDim;
        int s = 1;
        int totalSeq = pastSeq + s;

        var q = RandF(batch * s * qHid, rnd);
        var k = RandF(batch * s * kvHid, rnd);
        var v = RandF(batch * s * kvHid, rnd);
        var pk = RandF(batch * kvNumHeads * pastSeq * headDim, rnd);
        var pv = RandF(batch * kvNumHeads * pastSeq * headDim, rnd);
        var seqlens = new int[batch];
        for (int b = 0; b < batch; b++) seqlens[b] = totalSeq - 1;   // all keys valid.

        var graph = new ModelGraph
        {
            Inputs = new[] { "q", "k", "v", "past_k", "past_v", "seqlens" },
            Outputs = new[] { "out", "present_k", "present_v" },
            Nodes = new[]
            {
                N("GroupQueryAttention", "gqa",
                    new[] { "q", "k", "v", "past_k", "past_v", "seqlens" },
                    new[] { "out", "present_k", "present_v" },
                    new Dictionary<string, object>
                    {
                        ["num_heads"] = (long)numHeads, ["kv_num_heads"] = (long)kvNumHeads,
                    }),
            },
            Initializers = new Dictionary<string, Tensor>(),
        };
        var feeds = Feeds(
            ("q", F(new[] { batch, s, qHid }, q)),
            ("k", F(new[] { batch, s, kvHid }, k)),
            ("v", F(new[] { batch, s, kvHid }, v)),
            ("past_k", F(new[] { batch, kvNumHeads, pastSeq, headDim }, pk)),
            ("past_v", F(new[] { batch, kvNumHeads, pastSeq, headDim }, pv)),
            ("seqlens", new Tensor<int>(new TensorShape(batch), seqlens)));
        return (graph, feeds);
    }

    // ============================================================================================
    //  [Fact]/[Theory]s — ILGPU CPU accelerator (no CUDA required)
    // ============================================================================================

    public static IEnumerable<object[]> MatMulNBitsCases()
    {
        // M, K, N, bits, blockSize, zpKind, seed
        yield return new object[] { 1, 8, 4, 4, 8, "default", 1 };       // INT4, single block, symmetric zp
        yield return new object[] { 3, 16, 5, 4, 8, "default", 2 };      // INT4, two blocks
        yield return new object[] { 4, 32, 6, 4, 16, "float", 3 };       // INT4, float zero points
        yield return new object[] { 2, 24, 4, 4, 8, "packed", 4 };       // INT4, packed n-bit zero points
        yield return new object[] { 3, 16, 4, 8, 16, "default", 5 };     // INT8, single block
        yield return new object[] { 4, 40, 5, 8, 16, "float", 6 };       // INT8, float zero points
        yield return new object[] { 2, 32, 6, 8, 8, "packed", 7 };       // INT8, packed zero points
        yield return new object[] { 8, 64, 8, 4, 32, "default", 8 };     // larger INT4, block 32
        yield return new object[] { 1, 12, 3, 4, 4, "default", 9 };      // block not dividing K evenly (12/4=3 ok); odd N
        yield return new object[] { 5, 20, 7, 4, 16, "float", 10 };      // partial last block (20 % 16 = 4)
    }

    [Theory]
    [MemberData(nameof(MatMulNBitsCases))]
    public void MatMulNBits_Native_Parity_CpuAccel(int m, int k, int n, int bits, int blockSize, string zpKind, int seed)
    {
        var (g, f) = BuildMatMulNBits(m, k, n, bits, blockSize, zpKind, seed);
        AssertParity($"MatMulNBits[{m}x{k}x{n},bits{bits},bs{blockSize},{zpKind}]", g, f, cuda: false);
    }

    public static IEnumerable<object[]> GqaPackedCases()
    {
        // batch, s, numHeads, kvNumHeads, headDim, rotary, seed
        yield return new object[] { 1, 4, 2, 1, 4, false, 1 };   // packed-QKV, no rotary, repeat-KV (group 2)
        yield return new object[] { 1, 3, 4, 2, 4, false, 2 };   // group size 2, 4 q heads / 2 kv heads
        yield return new object[] { 1, 4, 2, 1, 4, true, 3 };    // packed-QKV + rotary (half-split)
        yield return new object[] { 2, 3, 2, 1, 6, true, 4 };    // batch 2, rotary
        yield return new object[] { 1, 5, 4, 4, 8, true, 5 };    // MHA-style (kv==q heads), rotary
    }

    [Theory]
    [MemberData(nameof(GqaPackedCases))]
    public void Gqa_Packed_Parity_CpuAccel(int batch, int s, int nh, int kvh, int hd, bool rotary, int seed)
    {
        var (g, f) = BuildGqaPacked(batch, s, nh, kvh, hd, rotary, seed);
        AssertParity($"GqaPacked[b{batch},s{s},nh{nh},kvh{kvh},hd{hd},rot{rotary}]", g, f, cuda: false);
    }

    public static IEnumerable<object[]> GqaDecodeCases()
    {
        // batch, numHeads, kvNumHeads, headDim, pastSeq, seed
        yield return new object[] { 1, 2, 1, 2, 2, 1 };
        yield return new object[] { 1, 4, 2, 4, 3, 2 };
        yield return new object[] { 2, 2, 1, 4, 4, 3 };
    }

    [Theory]
    [MemberData(nameof(GqaDecodeCases))]
    public void Gqa_DecodeWithPast_Parity_CpuAccel(int batch, int nh, int kvh, int hd, int pastSeq, int seed)
    {
        var (g, f) = BuildGqaDecodeWithPast(batch, nh, kvh, hd, pastSeq, seed);
        AssertParity($"GqaDecode[b{batch},nh{nh},kvh{kvh},hd{hd},past{pastSeq}]", g, f, cuda: false);
    }

    // ============================================================================================
    //  Cuda_* — same graphs on real hardware CUDA (skips cleanly when absent / OOM)
    // ============================================================================================

    [Collection("CudaGpu")]
    public class Cuda
    {
        private readonly ITestOutputHelper _out;
        public Cuda(ITestOutputHelper output) => _out = output;

        private void Run(string what, (ModelGraph g, Dictionary<string, NamedTensor> f) gf)
        {
            if (!HardwareGpuAvailable())
            {
                _out.WriteLine($"{what}: no CUDA device; skipping.");
                return;
            }
            try
            {
                new GpuQuantizedLlmOpsTests(_out).AssertParity(what, gf.g, gf.f, cuda: true);
            }
            catch (Exception ex) when (ex.Message.Contains("out of memory"))
            {
                _out.WriteLine($"{what}: device out of memory (shared GPU box); skipping.");
            }
        }

        [Theory]
        [MemberData(nameof(MatMulNBitsCases), MemberType = typeof(GpuQuantizedLlmOpsTests))]
        public void Cuda_MatMulNBits_Native(int m, int k, int n, int bits, int blockSize, string zpKind, int seed)
            => Run($"MatMulNBits[{m}x{k}x{n},bits{bits},bs{blockSize},{zpKind}]",
                BuildMatMulNBits(m, k, n, bits, blockSize, zpKind, seed));

        [Theory]
        [MemberData(nameof(GqaPackedCases), MemberType = typeof(GpuQuantizedLlmOpsTests))]
        public void Cuda_Gqa_Packed(int batch, int s, int nh, int kvh, int hd, bool rotary, int seed)
            => Run($"GqaPacked[b{batch},s{s},nh{nh},kvh{kvh},hd{hd},rot{rotary}]",
                BuildGqaPacked(batch, s, nh, kvh, hd, rotary, seed));

        [Theory]
        [MemberData(nameof(GqaDecodeCases), MemberType = typeof(GpuQuantizedLlmOpsTests))]
        public void Cuda_Gqa_DecodeWithPast(int batch, int nh, int kvh, int hd, int pastSeq, int seed)
            => Run($"GqaDecode[b{batch},nh{nh},kvh{kvh},hd{hd},past{pastSeq}]",
                BuildGqaDecodeWithPast(batch, nh, kvh, hd, pastSeq, seed));
    }
}
