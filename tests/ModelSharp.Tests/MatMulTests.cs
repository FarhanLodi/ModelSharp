using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

public class MatMulTests
{
    private static string Case(string name, string file) =>
        Path.Combine(AppContext.BaseDirectory, "assets", name, file);

    private static void RunOnnxMatMulCase(string caseName)
    {
        ModelGraph graph = OnnxModelLoader.LoadModel(Case(caseName, "model.onnx"));
        using var engine = new ManagedCpuEngine(graph);

        var feeds = new Dictionary<string, NamedTensor>();
        for (int i = 0; i < engine.Inputs.Count; i++)
        {
            string name = engine.Inputs[i].Name;
            Tensor<float> t = OnnxModelLoader.LoadTensor(Case(caseName, $"test_data_set_0/input_{i}.pb"));
            feeds[name] = new NamedTensor(name, t);
        }

        Tensor<float> actual = engine.Run(feeds).Values.Single().Data;
        Tensor<float> expected = OnnxModelLoader.LoadTensor(Case(caseName, "test_data_set_0/output_0.pb"));

        Assert.Equal(expected.Length, actual.Length);
        float[] e = expected.Span.ToArray();
        float[] a = actual.Span.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) < 1e-4f, $"[{i}] expected {e[i]}, got {a[i]}");
    }

    [Fact]
    public void MatMul_2d_Matches_Onnx_Reference() => RunOnnxMatMulCase("test_matmul_2d");

    [Fact]
    public void MatMul_3d_Batched_Matches_Onnx_Reference() => RunOnnxMatMulCase("test_matmul_3d");

    [Fact]
    public void MatMul_4d_Batched_Matches_Onnx_Reference() => RunOnnxMatMulCase("test_matmul_4d");
}
