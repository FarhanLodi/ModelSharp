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

public class OnnxMnistTests
{
    private static string Asset(string n) => Path.Combine(AppContext.BaseDirectory, "assets", n);

    [Fact]
    public void Loads_Mnist_Graph()
    {
        ModelGraph g = OnnxModelLoader.LoadModel(Asset("mnist-12.onnx"));
        Assert.Single(g.Inputs);
        Assert.Single(g.Outputs);
        Assert.NotEmpty(g.Nodes);
        Assert.Contains(g.Nodes, n => n.OpType == "Conv");
        Assert.NotEmpty(g.Initializers);
    }

    [Fact]
    public void Mnist_Matches_Onnx_Reference_Output()
    {
        ModelGraph g = OnnxModelLoader.LoadModel(Asset("mnist-12.onnx"));
        Tensor<float> input = OnnxModelLoader.LoadTensor(Asset("input_0.pb"));
        Tensor<float> expected = OnnxModelLoader.LoadTensor(Asset("output_0.pb"));

        using var engine = new ManagedCpuEngine(g);
        string inName = engine.Inputs.Single().Name;
        IReadOnlyDictionary<string, NamedTensor> outputs = engine.Run(new Dictionary<string, NamedTensor>
        {
            [inName] = new NamedTensor(inName, input),
        });
        Tensor<float> actual = outputs.Values.Single().Data;

        Assert.Equal(expected.Length, actual.Length);
        float[] e = expected.Span.ToArray();
        float[] a = actual.Span.ToArray();

        // Same predicted digit as ONNX Runtime, and every logit within tolerance.
        Assert.Equal(ArgMax(e), ArgMax(a));
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) < 1e-2f, $"logit {i}: expected {e[i]}, got {a[i]}");
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}
