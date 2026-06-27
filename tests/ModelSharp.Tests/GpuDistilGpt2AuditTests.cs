using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// B5 audit: enumerate the distinct op types in the real distilgpt2 decoder and report which the
/// GPU engine (<see cref="ModelSharp.Gpu.IlgpuEngine"/>) dispatches versus which still fall back.
/// This is a reporting test — it always passes; the data is written to the test output.
/// </summary>
public class GpuDistilGpt2AuditTests
{
    private readonly ITestOutputHelper _out;
    public GpuDistilGpt2AuditTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Local-only discovery of <c>distilgpt2.onnx</c>: <c>MODELSHARP_MODELS_DIR</c> → a repo-relative
    /// <c>models/</c> dir found by walking up from the test output directory. Skips cleanly when absent.
    /// </summary>
    private static bool TryFindModel(out string path)
    {
        string? env = Environment.GetEnvironmentVariable("MODELSHARP_MODELS_DIR");
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(env))
            candidates.Add(Path.Combine(env, "distilgpt2.onnx"));
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            candidates.Add(Path.Combine(dir.FullName, "models", "distilgpt2.onnx"));
            dir = dir.Parent;
        }
        foreach (string c in candidates)
            if (File.Exists(c)) { path = c; return true; }
        path = candidates.Count > 0 ? candidates[0] : "distilgpt2.onnx";
        return false;
    }

    // The op types the GPU engine's Run switch dispatches (mirror IlgpuEngine.Run exactly).
    private static readonly HashSet<string> GpuSupported = new(StringComparer.Ordinal)
    {
        "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh", "Gelu", "Exp", "Sqrt",
        "LeakyRelu", "Transpose", "Softmax", "ReduceSum", "ReduceMean", "LayerNormalization",
        "Gather", "Concat", "Slice", "Cast", "MatMul", "Gemm", "Conv",
        // B5 additions:
        "Erf", "Pow", "Where", "Reshape", "Unsqueeze", "Squeeze", "Shape", "Constant",
        "Expand", "Split",
        // B5 completion — integer/boolean mask & position-id prologue ops (host-side):
        "Range", "ConstantOfShape", "Equal", "Greater", "Trilu", "ScatterND",
    };

    [Fact]
    public void Audit_DistilGpt2_Gpu_Op_Coverage()
    {
        if (!TryFindModel(out string modelPath))
        {
            _out.WriteLine("distilgpt2 not present (looked under MODELSHARP_MODELS_DIR / repo models/); skipping audit.");
            return;
        }

        ModelGraph graph = OnnxModelLoader.LoadModel(modelPath);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (GraphNode n in graph.Nodes)
            counts[n.OpType] = counts.TryGetValue(n.OpType, out int c) ? c + 1 : 1;

        _out.WriteLine($"distilgpt2: {graph.Nodes.Count} nodes, {counts.Count} distinct op types.");
        _out.WriteLine($"Inputs: {string.Join(", ", graph.Inputs)}");
        _out.WriteLine($"Outputs: {string.Join(", ", graph.Outputs)}");
        _out.WriteLine("--- op | count | GPU? ---");
        foreach (KeyValuePair<string, int> kv in counts.OrderByDescending(k => k.Value))
        {
            bool sup = GpuSupported.Contains(kv.Key);
            _out.WriteLine($"{kv.Key,-24} {kv.Value,5}  {(sup ? "GPU" : "FALLBACK")}");
        }

        var missing = counts.Keys.Where(k => !GpuSupported.Contains(k)).OrderBy(k => k).ToList();
        _out.WriteLine("--- MISSING (not on GPU): ---");
        _out.WriteLine(missing.Count == 0 ? "(none)" : string.Join(", ", missing));

        // How many nodes (in topo order) the GPU engine could dispatch before hitting the first fallback op.
        int reachable = 0;
        foreach (GraphNode n in graph.Nodes)
        {
            if (!GpuSupported.Contains(n.OpType)) break;
            reachable++;
        }
        _out.WriteLine($"GPU-dispatchable nodes before first fallback op (topo order): {reachable}/{graph.Nodes.Count}");
        int firstFallbackIdx = graph.Nodes.ToList().FindIndex(n => !GpuSupported.Contains(n.OpType));
        if (firstFallbackIdx >= 0)
            _out.WriteLine($"First fallback op: '{graph.Nodes[firstFallbackIdx].OpType}' (node #{firstFallbackIdx} '{graph.Nodes[firstFallbackIdx].Name}')");
    }
}
