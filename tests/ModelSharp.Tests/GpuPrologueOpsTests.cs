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
/// B5 completion — hardware-CUDA parity for the six integer/boolean mask &amp; position-id "prologue" ops the
/// GPU engine now dispatches (Range, ConstantOfShape, Equal, Greater, Trilu, ScatterND). These are control-flow
/// ops that build distilgpt2's causal mask and position ids; the GPU engine runs them host-side (on the
/// int/bool tensors that already flow host-side) so the whole distilgpt2 graph becomes GPU-dispatchable. Each
/// test asserts GPU-vs-CPU output parity at the native dtype and confirms <c>IsHardwareGpu==true</c>, skipping
/// cleanly when no CUDA device is present. Shares the serialized <c>CudaGpu</c> collection (one live CUDA
/// context at a time), mirroring <see cref="GpuLlmTests"/>.
/// </summary>
[Collection("CudaGpu")]
public class GpuPrologueOpsTests
{
    private readonly ITestOutputHelper _out;
    public GpuPrologueOpsTests(ITestOutputHelper output) => _out = output;

    private static GraphNode N(string op, string name, string[] inputs, string[] outputs,
        Dictionary<string, object>? attrs = null) => new GraphNode(op, name, inputs, outputs, attrs);

    private static bool HardwareGpuAvailable()
    {
        try
        {
            using Context ctx = Context.CreateDefault();
            return ctx.Devices.Any(d => d.AcceleratorType != AcceleratorType.CPU);
        }
        catch { return false; }
    }

    /// <summary>
    /// Runs a graph (with no feeds — these ops are driven entirely by initializers/attributes) on CUDA and on
    /// the CPU engine and asserts every output matches exactly at its native dtype. Skips green without CUDA.
    /// </summary>
    private void AssertCudaMatchesCpu(string what, ModelGraph graph,
        Dictionary<string, NamedTensor>? feeds = null)
    {
        if (!HardwareGpuAvailable())
        {
            _out.WriteLine($"{what}: no CUDA device; skipping.");
            return;
        }

        feeds ??= new Dictionary<string, NamedTensor>();
        using var gpu = new IlgpuEngine(graph, preferCpu: false);
        Assert.True(gpu.IsHardwareGpu, $"{what}: expected a hardware GPU but got '{gpu.AcceleratorName}'.");
        _out.WriteLine($"{what}: hardware GPU '{gpu.AcceleratorName}' (IsHardwareGpu=true).");

        using var cpu = new ManagedCpuEngine(graph);
        IReadOnlyDictionary<string, NamedTensor> gpuOut = gpu.Run(feeds);
        IReadOnlyDictionary<string, NamedTensor> cpuOut = cpu.Run(feeds);

        foreach (string name in graph.Outputs)
        {
            Tensor g = gpuOut[name].Tensor;
            Tensor c = cpuOut[name].Tensor;
            Assert.Equal(c.Dtype, g.Dtype);
            Assert.Equal(c.Shape.Dimensions.ToArray(), g.Shape.Dimensions.ToArray());
            AssertTensorsEqual($"{what}:{name}", c, g);
        }
    }

    /// <summary>Exact element-wise equality at the tensor's native dtype (int/bool exact; float bit-for-bit).</summary>
    private static void AssertTensorsEqual(string what, Tensor c, Tensor g)
    {
        switch (c.Dtype)
        {
            case ElementType.Int64:
            {
                long[] cc = c.AsInt64().Span.ToArray(), gg = g.AsInt64().Span.ToArray();
                Assert.Equal(cc.Length, gg.Length);
                for (int i = 0; i < cc.Length; i++) Assert.True(cc[i] == gg[i], $"{what}[{i}] cpu={cc[i]} gpu={gg[i]}");
                break;
            }
            case ElementType.Int32:
            {
                int[] cc = c.AsInt32().Span.ToArray(), gg = g.AsInt32().Span.ToArray();
                Assert.Equal(cc.Length, gg.Length);
                for (int i = 0; i < cc.Length; i++) Assert.True(cc[i] == gg[i], $"{what}[{i}] cpu={cc[i]} gpu={gg[i]}");
                break;
            }
            case ElementType.Boolean:
            {
                bool[] cc = c.AsBool().Span.ToArray(), gg = g.AsBool().Span.ToArray();
                Assert.Equal(cc.Length, gg.Length);
                for (int i = 0; i < cc.Length; i++) Assert.True(cc[i] == gg[i], $"{what}[{i}] cpu={cc[i]} gpu={gg[i]}");
                break;
            }
            default:
            {
                float[] cc = c.AsFloat().Span.ToArray(), gg = g.AsFloat().Span.ToArray();
                Assert.Equal(cc.Length, gg.Length);
                for (int i = 0; i < cc.Length; i++) Assert.True(cc[i] == gg[i], $"{what}[{i}] cpu={cc[i]} gpu={gg[i]}");
                break;
            }
        }
    }

    // ---- Range (int64 and float) ----

    [Fact]
    public void Cuda_Range_Int64()
        => AssertCudaMatchesCpu("RangeInt64",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[] { N("Range", "r", new[] { "start", "limit", "delta" }, new[] { "y" }) },
                Initializers = new Dictionary<string, Tensor>
                {
                    ["start"] = new Tensor<long>(new TensorShape(Array.Empty<int>()), new long[] { 0 }),
                    ["limit"] = new Tensor<long>(new TensorShape(Array.Empty<int>()), new long[] { 10 }),
                    ["delta"] = new Tensor<long>(new TensorShape(Array.Empty<int>()), new long[] { 2 }),
                },
            });

    [Fact]
    public void Cuda_Range_Float()
        => AssertCudaMatchesCpu("RangeFloat",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[] { N("Range", "r", new[] { "start", "limit", "delta" }, new[] { "y" }) },
                Initializers = new Dictionary<string, Tensor>
                {
                    ["start"] = new Tensor<float>(new TensorShape(1), new[] { 1f }),
                    ["limit"] = new Tensor<float>(new TensorShape(1), new[] { 3f }),
                    ["delta"] = new Tensor<float>(new TensorShape(1), new[] { 0.5f }),
                },
            });

    // ---- ConstantOfShape (float fill, and int64 fill) ----

    [Fact]
    public void Cuda_ConstantOfShape_FloatFill()
        => AssertCudaMatchesCpu("ConstantOfShapeFloat",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[]
                {
                    N("ConstantOfShape", "cos", new[] { "shape" }, new[] { "y" },
                        new Dictionary<string, object>
                        {
                            ["value"] = new Tensor<float>(new TensorShape(1), new[] { -3.5f }),
                        }),
                },
                Initializers = new Dictionary<string, Tensor>
                {
                    ["shape"] = new Tensor<long>(new TensorShape(2), new long[] { 2, 3 }),
                },
            });

    [Fact]
    public void Cuda_ConstantOfShape_Int64Fill()
        => AssertCudaMatchesCpu("ConstantOfShapeInt64",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[]
                {
                    N("ConstantOfShape", "cos", new[] { "shape" }, new[] { "y" },
                        new Dictionary<string, object>
                        {
                            ["value"] = new Tensor<long>(new TensorShape(1), new long[] { 7 }),
                        }),
                },
                Initializers = new Dictionary<string, Tensor>
                {
                    ["shape"] = new Tensor<long>(new TensorShape(3), new long[] { 1, 2, 2 }),
                },
            });

    // ---- Equal (bool output, with broadcasting) ----

    [Fact]
    public void Cuda_Equal_Int64_Broadcast()
        => AssertCudaMatchesCpu("Equal",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[] { N("Equal", "eq", new[] { "a", "b" }, new[] { "y" }) },
                Initializers = new Dictionary<string, Tensor>
                {
                    // a:[2,3], b:[3] broadcast -> [2,3] bool
                    ["a"] = new Tensor<long>(new TensorShape(2, 3), new long[] { 0, 1, 2, 2, 1, 5 }),
                    ["b"] = new Tensor<long>(new TensorShape(3), new long[] { 2, 1, 2 }),
                },
            });

    // ---- Greater (bool output) ----

    [Fact]
    public void Cuda_Greater_Int64()
        => AssertCudaMatchesCpu("Greater",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[] { N("Greater", "gt", new[] { "a", "b" }, new[] { "y" }) },
                Initializers = new Dictionary<string, Tensor>
                {
                    ["a"] = new Tensor<long>(new TensorShape(2, 2), new long[] { 5, 1, 3, 4 }),
                    ["b"] = new Tensor<long>(new TensorShape(2, 2), new long[] { 2, 1, 9, 0 }),
                },
            });

    // ---- Trilu (upper and lower; causal-mask shape) ----

    [Fact]
    public void Cuda_Trilu_Upper_Default()
        => AssertCudaMatchesCpu("TriluUpper",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[] { N("Trilu", "tu", new[] { "x" }, new[] { "y" }) }, // upper defaults to 1
                Initializers = new Dictionary<string, Tensor>
                {
                    ["x"] = new Tensor<long>(new TensorShape(4, 4),
                        Enumerable.Range(1, 16).Select(v => (long)v).ToArray()),
                },
            });

    [Fact]
    public void Cuda_Trilu_Lower_WithK()
        => AssertCudaMatchesCpu("TriluLower",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[]
                {
                    N("Trilu", "tl", new[] { "x", "k" }, new[] { "y" },
                        new Dictionary<string, object> { ["upper"] = 0L }),
                },
                Initializers = new Dictionary<string, Tensor>
                {
                    ["x"] = new Tensor<float>(new TensorShape(4, 4),
                        Enumerable.Range(1, 16).Select(v => (float)v).ToArray()),
                    ["k"] = new Tensor<long>(new TensorShape(Array.Empty<int>()), new long[] { 1 }),
                },
            });

    // ---- ScatterND (index update into a copied tensor) ----

    [Fact]
    public void Cuda_ScatterND_Update()
        => AssertCudaMatchesCpu("ScatterND",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[] { N("ScatterND", "sc", new[] { "data", "indices", "updates" }, new[] { "y" }) },
                Initializers = new Dictionary<string, Tensor>
                {
                    // data:[4,3] float; indices:[2,1] (k=1) address whole rows; updates:[2,3] rows.
                    ["data"] = new Tensor<float>(new TensorShape(4, 3),
                        Enumerable.Range(0, 12).Select(v => (float)v).ToArray()),
                    ["indices"] = new Tensor<long>(new TensorShape(2, 1), new long[] { 0, 2 }),
                    ["updates"] = new Tensor<float>(new TensorShape(2, 3),
                        new[] { 100f, 101f, 102f, 200f, 201f, 202f }),
                },
            });

    [Fact]
    public void Cuda_ScatterND_Int64()
        => AssertCudaMatchesCpu("ScatterNDInt64",
            new ModelGraph
            {
                Outputs = new[] { "y" },
                Nodes = new[] { N("ScatterND", "sc", new[] { "data", "indices", "updates" }, new[] { "y" }) },
                Initializers = new Dictionary<string, Tensor>
                {
                    // data:[3,2] int64; indices:[2,2] (k=2) address individual elements; updates:[2] scalars.
                    ["data"] = new Tensor<long>(new TensorShape(3, 2), new long[] { 0, 1, 2, 3, 4, 5 }),
                    ["indices"] = new Tensor<long>(new TensorShape(2, 2), new long[] { 0, 0, 2, 1 }),
                    ["updates"] = new Tensor<long>(new TensorShape(2), new long[] { 99, 88 }),
                },
            });
}
