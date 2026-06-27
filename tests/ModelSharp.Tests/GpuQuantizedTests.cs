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
/// Quantized-graph GPU/CPU parity: builds small in-memory <see cref="ModelGraph"/>s that exercise the
/// quantized op family — <c>DequantizeLinear</c>, <c>QuantizeLinear</c>, <c>DynamicQuantizeLinear</c>,
/// <c>MatMulInteger</c>, the <c>QLinear*</c> family, and a tiny quantized transformer-ish block — and
/// asserts the <see cref="IlgpuEngine"/> result matches the <see cref="ManagedCpuEngine"/> exactly.
///
/// <para>The quantized kernels have no native GPU implementation, so on the GPU engine they run through the
/// per-op CPU fallback (inputs downloaded to host, the CPU kernel run, outputs re-homed by dtype). These tests
/// prove that quantized inference is correct end-to-end on the GPU engine, including where quantized fallback
/// ops are interleaved with native GPU float ops (MatMul/Softmax/Add).</para>
///
/// <para>Two execution surfaces:
/// <list type="bullet">
/// <item>The <c>[Fact]</c> tests run on ILGPU's <b>CPU accelerator</b> (<c>preferCpu: true</c>) so the parity is
/// covered on every machine, with no CUDA required — the device-routing/fallback logic is identical to CUDA.</item>
/// <item>The <c>Cuda_*</c> tests in the <c>CudaGpu</c> collection re-run the same graphs on real <b>hardware
/// CUDA</b> (asserting <see cref="IlgpuEngine.IsHardwareGpu"/>), skipping cleanly when no CUDA device exists.</item>
/// </list></para>
/// </summary>
public class GpuQuantizedTests
{
    private readonly ITestOutputHelper _out;
    public GpuQuantizedTests(ITestOutputHelper output) => _out = output;

    // ---- tensor builders ----------------------------------------------------------------------

    /// <summary>ONNX TensorProto data_type for FLOAT (the <c>Cast</c> 'to' attribute value).</summary>
    private const long OnnxFloat = 1L;

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<float> Scalar(float v) => new(new TensorShape(), new[] { v });
    private static Tensor<byte> U8(int[] dims, params byte[] data) => new(new TensorShape(dims), data);
    private static Tensor<byte> ZpU8(byte v) => new(new TensorShape(), new[] { v });
    private static Tensor<sbyte> S8(int[] dims, params sbyte[] data) => new(new TensorShape(dims), data);

    private static Tensor<float> Rand(int[] dims, int seed, float lo = -1f, float hi = 1f)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (a, d) => a * d);
        float[] data = Enumerable.Range(0, n).Select(_ => lo + (float)rnd.NextDouble() * (hi - lo)).ToArray();
        return F(dims, data);
    }

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

    // ---- parity drivers -----------------------------------------------------------------------

    /// <summary>
    /// Runs <paramref name="graph"/> on the ILGPU engine (CPU accelerator unless <paramref name="cuda"/>) and on
    /// the managed CPU engine and asserts every declared output matches — element-exact for integer/byte outputs,
    /// within <paramref name="tol"/> for float outputs.
    /// </summary>
    private void AssertParity(string what, ModelGraph graph, Dictionary<string, NamedTensor> feeds,
        bool cuda, float tol = 1e-4f)
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
            Assert.Equal(c.Dtype, g.Dtype);
            Assert.Equal(c.Shape.Dimensions.ToArray(), g.Shape.Dimensions.ToArray());
            AssertTensorEqual(what, name, c, g, tol);
        }
    }

    private static void AssertTensorEqual(string what, string name, Tensor c, Tensor g, float tol)
    {
        switch (c.Dtype)
        {
            case ElementType.UInt8:
                Assert.Equal(((Tensor<byte>)c).Span.ToArray(), ((Tensor<byte>)g).Span.ToArray());
                break;
            case ElementType.Int8:
                Assert.Equal(((Tensor<sbyte>)c).Span.ToArray(), ((Tensor<sbyte>)g).Span.ToArray());
                break;
            case ElementType.Int32:
                Assert.Equal(c.AsInt32().Span.ToArray(), g.AsInt32().Span.ToArray());
                break;
            case ElementType.Int64:
                Assert.Equal(c.AsInt64().Span.ToArray(), g.AsInt64().Span.ToArray());
                break;
            default:
                float[] ca = c.AsFloat().Span.ToArray(), ga = g.AsFloat().Span.ToArray();
                Assert.Equal(ca.Length, ga.Length);
                for (int i = 0; i < ca.Length; i++)
                    Assert.True(MathF.Abs(ca[i] - ga[i]) < tol, $"{what}:{name}[{i}] cpu={ca[i]} gpu={ga[i]}");
                break;
        }
    }

    // ============================================================================================
    //  Graph builders (shared by the CPU-accelerator [Fact]s and the CUDA re-runs)
    // ============================================================================================

    /// <summary>MatMulInteger (uint8 A/B, scalar zero-points) → Cast(int32→float) → Mul(by a float scale).</summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildMatMulIntegerDequant()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "A", "B" },
            Outputs = new[] { "Y", "Yf" },
            Nodes = new[]
            {
                N("MatMulInteger", "mmi", new[] { "A", "B", "az", "bz" }, new[] { "Y" }),
                // Cast 'to' is the ONNX TensorProto dtype int (FLOAT=1), not the internal ElementType enum.
                N("Cast", "cast", new[] { "Y" }, new[] { "Yc" },
                    new Dictionary<string, object> { ["to"] = OnnxFloat }),
                N("Mul", "scale", new[] { "Yc", "yscale" }, new[] { "Yf" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["az"] = ZpU8(128),
                ["bz"] = ZpU8(64),
                ["yscale"] = Scalar(0.002f),
            },
        };
        // A [3,4] uint8, B [4,2] uint8.
        byte[] aq = { 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210 };
        byte[] bq = { 10, 20, 30, 40, 50, 60, 70, 80 };
        var feeds = Feeds(("A", U8(new[] { 3, 4 }, aq)), ("B", U8(new[] { 4, 2 }, bq)));
        return (graph, feeds);
    }

    /// <summary>QuantizeLinear(float→uint8) → QLinearMatMul → DequantizeLinear(uint8→float), per-tensor.</summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildQuantizeQLinearMatMulDequant()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "Xf" },
            Outputs = new[] { "Yf" },
            Nodes = new[]
            {
                // Quantize the dynamic activation A.
                N("QuantizeLinear", "qx", new[] { "Xf", "a_s", "a_z" }, new[] { "Aq" }),
                // Quantized matmul against a pre-quantized weight Bq.
                N("QLinearMatMul", "qmm",
                    new[] { "Aq", "a_s", "a_z", "Bq", "b_s", "b_z", "y_s", "y_z" }, new[] { "Yq" }),
                // Dequantize the result back to float.
                N("DequantizeLinear", "dq", new[] { "Yq", "y_s", "y_z" }, new[] { "Yf" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["a_s"] = Scalar(0.04f),
                ["a_z"] = ZpU8(128),
                ["Bq"] = U8(new[] { 3, 2 }, 10, 20, 30, 40, 50, 60),
                ["b_s"] = Scalar(0.02f),
                ["b_z"] = ZpU8(0),
                ["y_s"] = Scalar(0.1f),
                ["y_z"] = ZpU8(64),
            },
        };
        // A [2,3] float.
        var feeds = Feeds(("Xf", F(new[] { 2, 3 }, 1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f)));
        return (graph, feeds);
    }

    /// <summary>
    /// A quantized linear (Gemm-style) with PER-CHANNEL weight scales/zero-points: the weight is dequantized
    /// per output-channel (axis=0), matmul'd against the activation, and a bias added — the structure an INT8
    /// per-channel Linear layer lowers to. Exercises per-axis DequantizeLinear on the GPU engine.
    /// </summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildPerChannelQuantLinear()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "X" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                // Per-channel dequant of W [out=3, in=4] along axis 0.
                N("DequantizeLinear", "dqw", new[] { "Wq", "w_s", "w_z" }, new[] { "Wf" },
                    new Dictionary<string, object> { ["axis"] = 0L }),
                N("Transpose", "wt", new[] { "Wf" }, new[] { "Wt" },
                    new Dictionary<string, object> { ["perm"] = new[] { 1, 0 } }),
                N("MatMul", "mm", new[] { "X", "Wt" }, new[] { "XW" }),
                N("Add", "bias", new[] { "XW", "b" }, new[] { "Y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["Wq"] = S8(new[] { 3, 4 },
                    10, -20, 30, -40,
                    -5, 15, -25, 35,
                    50, -60, 5, -15),
                ["w_s"] = F(new[] { 3 }, 0.01f, 0.02f, 0.005f),   // per-channel scales
                ["w_z"] = S8(new[] { 3 }, 0, -2, 1),               // per-channel zero-points
                ["b"] = F(new[] { 3 }, 0.1f, -0.2f, 0.3f),
            },
        };
        var feeds = Feeds(("X", Rand(new[] { 2, 4 }, 7)));
        return (graph, feeds);
    }

    /// <summary>
    /// A tiny quantized transformer-ish block: a quantized linear projection (DynamicQuantizeLinear →
    /// MatMulInteger → dequant via Cast+Mul) feeds a self-attention (Softmax over QK scores → context),
    /// then a second DynamicQuantizeLinear → MatMulInteger → dequant output projection. Interleaves quantized
    /// fallback ops with native GPU float ops (MatMul/Transpose/Softmax/Mul/Add) end-to-end.
    /// </summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildQuantTransformerBlock()
    {
        // Shapes: X [S=3, D=4]. Wq_in/Wq_out are pre-quantized weights [4,4].
        var graph = new ModelGraph
        {
            Inputs = new[] { "X" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                // ---- quantized input projection:  H = (X @ Win) ----
                N("DynamicQuantizeLinear", "dq1", new[] { "X" }, new[] { "Xq", "Xs", "Xz" }),
                N("MatMulInteger", "mmi1", new[] { "Xq", "Win_q", "Xz", "Win_z" }, new[] { "Hi" }),
                N("Cast", "c1", new[] { "Hi" }, new[] { "Hc" },
                    new Dictionary<string, object> { ["to"] = OnnxFloat }),
                // dequant scale = Xs * Win_s (per-tensor)
                N("Mul", "hs", new[] { "Hc", "Xs" }, new[] { "Hsx" }),
                N("Mul", "hsw", new[] { "Hsx", "Win_s" }, new[] { "H" }),   // H [3,4]

                // ---- self-attention over H (single head): scores = H @ H^T, softmax, ctx = A @ H ----
                N("Transpose", "ht", new[] { "H" }, new[] { "Ht" },
                    new Dictionary<string, object> { ["perm"] = new[] { 1, 0 } }),
                N("MatMul", "qk", new[] { "H", "Ht" }, new[] { "S0" }),     // [3,3]
                N("Softmax", "sm", new[] { "S0" }, new[] { "A" },
                    new Dictionary<string, object> { ["axis"] = -1L }),
                N("MatMul", "av", new[] { "A", "H" }, new[] { "Ctx" }),     // [3,4]

                // ---- quantized output projection:  Y = (Ctx @ Wout) ----
                N("DynamicQuantizeLinear", "dq2", new[] { "Ctx" }, new[] { "Cq", "Cs", "Cz" }),
                N("MatMulInteger", "mmi2", new[] { "Cq", "Wout_q", "Cz", "Wout_z" }, new[] { "Oi" }),
                N("Cast", "c2", new[] { "Oi" }, new[] { "Oc" },
                    new Dictionary<string, object> { ["to"] = OnnxFloat }),
                N("Mul", "os", new[] { "Oc", "Cs" }, new[] { "Osx" }),
                N("Mul", "osw", new[] { "Osx", "Wout_s" }, new[] { "Y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["Win_q"] = U8(new[] { 4, 4 },
                    120, 130, 140, 150, 110, 125, 135, 145,
                    100, 160, 120, 140, 130, 150, 110, 120),
                ["Win_z"] = ZpU8(128),
                ["Win_s"] = Scalar(0.01f),
                ["Wout_q"] = U8(new[] { 4, 4 },
                    140, 120, 130, 110, 150, 125, 115, 135,
                    120, 140, 160, 100, 130, 110, 120, 150),
                ["Wout_z"] = ZpU8(128),
                ["Wout_s"] = Scalar(0.008f),
            },
        };
        var feeds = Feeds(("X", Rand(new[] { 3, 4 }, 99, -2f, 2f)));
        return (graph, feeds);
    }

    /// <summary>QLinearAdd then QLinearMul — residual-style quantized elementwise with per-tensor scales.</summary>
    private static (ModelGraph g, Dictionary<string, NamedTensor> f) BuildQLinearElementwise()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "A", "B", "C" },
            Outputs = new[] { "Y" },
            Nodes = new[]
            {
                N("QLinearAdd", "qadd",
                    new[] { "A", "a_s", "a_z", "B", "b_s", "b_z", "t_s", "t_z" }, new[] { "T" }),
                N("QLinearMul", "qmul",
                    new[] { "T", "t_s", "t_z", "C", "c_s", "c_z", "y_s", "y_z" }, new[] { "Y" }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["a_s"] = Scalar(0.1f),  ["a_z"] = ZpU8(8),
                ["b_s"] = Scalar(0.2f),  ["b_z"] = ZpU8(4),
                ["t_s"] = Scalar(0.15f), ["t_z"] = ZpU8(16),
                ["c_s"] = Scalar(0.05f), ["c_z"] = ZpU8(2),
                ["y_s"] = Scalar(0.05f), ["y_z"] = ZpU8(0),
            },
        };
        var feeds = Feeds(
            ("A", U8(new[] { 2, 3 }, 10, 20, 30, 40, 50, 60)),
            ("B", U8(new[] { 2, 3 }, 5, 15, 25, 35, 45, 55)),
            ("C", U8(new[] { 2, 3 }, 3, 6, 9, 12, 15, 18)));
        return (graph, feeds);
    }

    // ============================================================================================
    //  [Fact]s — run on the ILGPU CPU accelerator (no CUDA required, runs on every machine)
    // ============================================================================================

    [Fact]
    public void MatMulInteger_Dequant_Parity_CpuAccel()
    {
        var (g, f) = BuildMatMulIntegerDequant();
        AssertParity("MatMulIntegerDequant", g, f, cuda: false);
    }

    [Fact]
    public void Quantize_QLinearMatMul_Dequant_Parity_CpuAccel()
    {
        var (g, f) = BuildQuantizeQLinearMatMulDequant();
        AssertParity("QuantizeQLinearMatMulDequant", g, f, cuda: false);
    }

    [Fact]
    public void PerChannel_QuantLinear_Parity_CpuAccel()
    {
        var (g, f) = BuildPerChannelQuantLinear();
        AssertParity("PerChannelQuantLinear", g, f, cuda: false);
    }

    [Fact]
    public void QuantTransformerBlock_Parity_CpuAccel()
    {
        var (g, f) = BuildQuantTransformerBlock();
        AssertParity("QuantTransformerBlock", g, f, cuda: false);
    }

    [Fact]
    public void QLinearElementwise_Parity_CpuAccel()
    {
        var (g, f) = BuildQLinearElementwise();
        AssertParity("QLinearElementwise", g, f, cuda: false);
    }

    // ============================================================================================
    //  Cuda_* — same graphs on real hardware CUDA (skips cleanly when absent)
    // ============================================================================================

    [Collection("CudaGpu")]
    public class Cuda
    {
        private readonly ITestOutputHelper _out;
        public Cuda(ITestOutputHelper output) => _out = output;

        private void Run(string what, (ModelGraph g, Dictionary<string, NamedTensor> f) gf)
        {
            var owner = new GpuQuantizedTests(_out);
            if (!HardwareGpuAvailable())
            {
                _out.WriteLine($"{what}: no CUDA device; skipping.");
                return;
            }
            owner.AssertParity(what, gf.g, gf.f, cuda: true);
        }

        [Fact] public void Cuda_MatMulInteger_Dequant() => Run("MatMulIntegerDequant", BuildMatMulIntegerDequant());
        [Fact] public void Cuda_Quantize_QLinearMatMul_Dequant() => Run("QuantizeQLinearMatMulDequant", BuildQuantizeQLinearMatMulDequant());
        [Fact] public void Cuda_PerChannel_QuantLinear() => Run("PerChannelQuantLinear", BuildPerChannelQuantLinear());
        [Fact] public void Cuda_QuantTransformerBlock() => Run("QuantTransformerBlock", BuildQuantTransformerBlock());
        [Fact] public void Cuda_QLinearElementwise() => Run("QLinearElementwise", BuildQLinearElementwise());
    }
}
