using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Graph;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Engine;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu;

/// <summary>
/// Pure-managed CPU execution engine (Phase 1). Runs a <see cref="ModelGraph"/>
/// node-by-node in topological order via the kernel registry. No native dependencies.
/// </summary>
public sealed class ManagedCpuEngine : IExecutionEngine
{
    private readonly ModelGraph _graph;
    private readonly KernelRegistry _kernels;

    /// <inheritdoc />
    public IReadOnlyList<TensorInfo> Inputs { get; }

    /// <inheritdoc />
    public IReadOnlyList<TensorInfo> Outputs { get; }

    public ManagedCpuEngine(ModelGraph graph, KernelRegistry? kernels = null)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
        _kernels = kernels ?? KernelRegistry.CreateDefault();
        Inputs = _graph.Inputs.Select(n => new TensorInfo(n, ElementType.Float32, Array.Empty<int>())).ToList();
        Outputs = _graph.Outputs.Select(n => new TensorInfo(n, ElementType.Float32, Array.Empty<int>())).ToList();
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
    {
        // Seed the environment with baked-in initializers, then the caller's feeds.
        var env = new Dictionary<string, Tensor>();
        foreach (KeyValuePair<string, Tensor> init in _graph.Initializers)
            env[init.Key] = init.Value;

        foreach (string input in _graph.Inputs)
        {
            if (!feeds.TryGetValue(input, out NamedTensor? fed))
                throw new ModelSharpException($"Missing feed for required input '{input}'.");
            env[input] = fed.Tensor;
        }

        var ctx = new GraphContext(env);

        // Execute nodes in topological order.
        foreach (GraphNode node in _graph.Nodes)
        {
            if (!_kernels.TryGet(node.OpType, out IKernel? kernel) || kernel is null)
                throw new UnsupportedOperatorException(node.OpType, $"node '{node.Name}'");
            kernel.Execute(node, ctx);
        }

        // Collect the requested outputs.
        var result = new Dictionary<string, NamedTensor>();
        foreach (string outName in _graph.Outputs)
            result[outName] = new NamedTensor(outName, ctx.GetTensor(outName));
        return result;
    }

    /// <inheritdoc />
    public void Dispose() { /* No unmanaged resources in Phase 1. */ }
}
