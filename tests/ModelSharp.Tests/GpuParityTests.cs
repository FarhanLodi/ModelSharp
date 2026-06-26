using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Gpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Parity coverage for the GPU ops added in roadmap B2 (multi-dtype: int32/int64 tensors flowing
/// through the engine) and B3 (LayerNormalization, Gather, Concat, Slice, Cast). Each op is run on
/// BOTH the ILGPU engine (on its CPU accelerator, so it works on any machine) and the managed CPU
/// engine, and the outputs are asserted to match (float within 1e-4, integers exactly). Engine
/// construction mirrors <see cref="GpuEngineTests"/> exactly.
/// </summary>
public class GpuParityTests
{
    private static Tensor<float> T(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> I64(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    private static Tensor<int> I32(int[] dims, params int[] data) =>
        Tensor<int>.FromArray(new TensorShape(dims), data);

    private static Dictionary<string, NamedTensor> Feeds(params (string name, Tensor t)[] feeds) =>
        feeds.ToDictionary(f => f.name, f => new NamedTensor(f.name, f.t));

    private static Tensor<float> Rand(int[] dims, int seed, float lo = -1f, float hi = 1f)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (a, d) => a * d);
        float[] data = Enumerable.Range(0, n).Select(_ => lo + (float)rnd.NextDouble() * (hi - lo)).ToArray();
        return T(dims, data);
    }

    /// <summary>
    /// Runs the same graph + feeds through the GPU engine and the managed CPU engine and asserts every
    /// output matches. Float outputs are compared to <paramref name="tol"/>; integer/bool outputs exactly.
    /// </summary>
    private static void AssertGpuMatchesCpu(ModelGraph graph, Dictionary<string, NamedTensor> feeds, float tol = 1e-4f)
    {
        using var gpu = new IlgpuEngine(graph, preferCpu: true);
        using var cpu = new ManagedCpuEngine(graph);

        IReadOnlyDictionary<string, NamedTensor> gpuOut = gpu.Run(feeds);
        IReadOnlyDictionary<string, NamedTensor> cpuOut = cpu.Run(feeds);

        foreach (string name in graph.Outputs)
        {
            Tensor g = gpuOut[name].Tensor;
            Tensor c = cpuOut[name].Tensor;
            Assert.Equal(c.Shape.Dimensions.ToArray(), g.Shape.Dimensions.ToArray());
            Assert.Equal(c.Dtype, g.Dtype);

            switch (c.Dtype)
            {
                case ElementType.Float32:
                {
                    float[] ga = g.AsFloat().Span.ToArray();
                    float[] ca = c.AsFloat().Span.ToArray();
                    for (int i = 0; i < ca.Length; i++)
                        Assert.True(MathF.Abs(ca[i] - ga[i]) < tol, $"{name}[{i}] cpu={ca[i]} gpu={ga[i]}");
                    break;
                }
                case ElementType.Int64:
                    Assert.Equal(c.AsInt64().Span.ToArray(), g.AsInt64().Span.ToArray());
                    break;
                case ElementType.Int32:
                    Assert.Equal(c.AsInt32().Span.ToArray(), g.AsInt32().Span.ToArray());
                    break;
                case ElementType.Boolean:
                    Assert.Equal(c.AsBool().Span.ToArray(), g.AsBool().Span.ToArray());
                    break;
                default:
                    Assert.Fail($"Unexpected output dtype {c.Dtype}");
                    break;
            }
        }
    }

    // --- B3: LayerNormalization (real ILGPU device kernel) ---

    [Fact]
    public void Ilgpu_LayerNorm_NoBias_Matches_Cpu()
    {
        // x:(2,4), scale:(4) — normalize over the last axis (default axis=-1), no bias.
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("LayerNormalization", "ln", new[] { "x", "scale" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["scale"] = T(new[] { 4 }, 1f, 0.5f, 2f, -1f) },
        };
        AssertGpuMatchesCpu(graph, Feeds(("x", Rand(new[] { 2, 4 }, 1, -3f, 3f))), 1e-3f);
    }

    [Fact]
    public void Ilgpu_LayerNorm_Bias_Axis_Matches_Cpu()
    {
        // x:(2,3,4), axis=2, with scale and bias over the trailing 4 elements.
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("LayerNormalization", "ln", new[] { "x", "scale", "bias" }, new[] { "y" },
                    new Dictionary<string, object> { ["axis"] = 2L, ["epsilon"] = 1e-5f }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["scale"] = T(new[] { 4 }, 1.0f, 0.8f, 1.2f, 0.5f),
                ["bias"] = T(new[] { 4 }, 0.1f, -0.2f, 0.3f, 0f),
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(("x", Rand(new[] { 2, 3, 4 }, 2, -4f, 4f))), 1e-3f);
    }

    // --- B3: Gather ---

    [Fact]
    public void Ilgpu_Gather_Axis0_Int64Indices_Matches_Cpu()
    {
        // data:(4,3) float, gather rows [2,0,2] with int64 indices -> (3,3).
        var graph = new ModelGraph
        {
            Inputs = new[] { "data" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Gather", "g", new[] { "data", "idx" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor> { ["idx"] = I64(new[] { 3 }, 2, 0, 2) },
        };
        AssertGpuMatchesCpu(graph, Feeds(("data", Rand(new[] { 4, 3 }, 3))));
    }

    [Fact]
    public void Ilgpu_Gather_Axis1_2DIndices_NegativeIndex_Matches_Cpu()
    {
        // data:(2,5) gather along axis=1 with a (2,2) int32 index grid, including a negative index.
        var graph = new ModelGraph
        {
            Inputs = new[] { "data" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Gather", "g", new[] { "data", "idx" }, new[] { "y" },
                    new Dictionary<string, object> { ["axis"] = 1L }),
            },
            Initializers = new Dictionary<string, Tensor> { ["idx"] = I32(new[] { 2, 2 }, 0, -1, 3, 1) },
        };
        AssertGpuMatchesCpu(graph, Feeds(("data", Rand(new[] { 2, 5 }, 4))));
    }

    // --- B3: Concat ---

    [Fact]
    public void Ilgpu_Concat_Axis0_Float_Matches_Cpu()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Concat", "c", new[] { "a", "b" }, new[] { "y" },
                    new Dictionary<string, object> { ["axis"] = 0L }),
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(
            ("a", Rand(new[] { 2, 3 }, 5)),
            ("b", Rand(new[] { 1, 3 }, 6))));
    }

    [Fact]
    public void Ilgpu_Concat_Axis1_Float_Matches_Cpu()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "a", "b", "c" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Concat", "cc", new[] { "a", "b", "c" }, new[] { "y" },
                    new Dictionary<string, object> { ["axis"] = 1L }),
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(
            ("a", Rand(new[] { 2, 2 }, 7)),
            ("b", Rand(new[] { 2, 3 }, 8)),
            ("c", Rand(new[] { 2, 1 }, 9))));
    }

    // --- B3: Slice ---

    [Fact]
    public void Ilgpu_Slice_Basic_Matches_Cpu()
    {
        // Slice axis 0 [1:3] and axis 1 [0:4:2] via the opset-10 integer inputs.
        var graph = new ModelGraph
        {
            Inputs = new[] { "data" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Slice", "s", new[] { "data", "starts", "ends", "axes", "steps" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["starts"] = I64(new[] { 2 }, 1, 0),
                ["ends"] = I64(new[] { 2 }, 3, 4),
                ["axes"] = I64(new[] { 2 }, 0, 1),
                ["steps"] = I64(new[] { 2 }, 1, 2),
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(("data", Rand(new[] { 4, 5 }, 10))));
    }

    [Fact]
    public void Ilgpu_Slice_NegativeStep_Matches_Cpu()
    {
        // Reverse axis 1: starts=-1, ends=-100 (clamps to before index 0), step=-1.
        var graph = new ModelGraph
        {
            Inputs = new[] { "data" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Slice", "s", new[] { "data", "starts", "ends", "axes", "steps" }, new[] { "y" }) },
            Initializers = new Dictionary<string, Tensor>
            {
                ["starts"] = I64(new[] { 1 }, -1),
                ["ends"] = I64(new[] { 1 }, -100),
                ["axes"] = I64(new[] { 1 }, 1),
                ["steps"] = I64(new[] { 1 }, -1),
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(("data", Rand(new[] { 2, 4 }, 11))));
    }

    // --- B3: Cast ---

    [Fact]
    public void Ilgpu_Cast_Float_To_Int64_Matches_Cpu()
    {
        // Truncation toward zero; output is an int64 tensor (compared exactly).
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Cast", "cast", new[] { "x" }, new[] { "y" },
                    new Dictionary<string, object> { ["to"] = 7L }), // INT64
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(("x", T(new[] { 2, 3 }, 1.9f, -1.9f, 0.4f, -0.4f, 3.5f, -3.5f))));
    }

    [Fact]
    public void Ilgpu_Cast_Float_To_Int32_Matches_Cpu()
    {
        var graph = new ModelGraph
        {
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Cast", "cast", new[] { "x" }, new[] { "y" },
                    new Dictionary<string, object> { ["to"] = 6L }), // INT32
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(("x", T(new[] { 4 }, 2.7f, -2.7f, 100.9f, -100.9f))));
    }

    // --- B2: integer-dtype tensors flow through without throwing ---

    [Fact]
    public void Ilgpu_Cast_Int64_To_Float_PassesThrough_Without_Throwing()
    {
        // An int64 input flows in and is cast to float on the device — proves the engine no longer
        // forces float inputs and does not crash on an int64 feed.
        var graph = new ModelGraph
        {
            Inputs = new[] { "ids" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("Cast", "cast", new[] { "ids" }, new[] { "y" },
                    new Dictionary<string, object> { ["to"] = 1L }), // FLOAT
            },
        };
        AssertGpuMatchesCpu(graph, Feeds(("ids", I64(new[] { 1, 5 }, 101, 7, 0, 42, 9))));
    }

    [Fact]
    public void Ilgpu_Gather_With_Int64_Input_Indices_PassesThrough()
    {
        // Token-embedding style: int64 indices fed as a graph input (not an initializer) gather rows
        // of a float embedding table. Exercises an int64 input flowing through the engine end-to-end.
        var graph = new ModelGraph
        {
            Inputs = new[] { "table", "ids" },
            Outputs = new[] { "y" },
            Nodes = new[] { new GraphNode("Gather", "g", new[] { "table", "ids" }, new[] { "y" }) },
        };
        AssertGpuMatchesCpu(graph, Feeds(
            ("table", Rand(new[] { 6, 4 }, 12)),
            ("ids", I64(new[] { 3 }, 5, 0, 3))));
    }
}
