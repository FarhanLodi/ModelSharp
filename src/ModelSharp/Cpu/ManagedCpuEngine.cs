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

        Dictionary<string, Tensor> outputs = ExecuteNodes(_graph, env);

        // Collect the requested outputs.
        var result = new Dictionary<string, NamedTensor>();
        foreach (string outName in _graph.Outputs)
            result[outName] = new NamedTensor(outName, outputs[outName]);
        return result;
    }

    /// <summary>
    /// Executes a graph's nodes against an already-seeded environment and returns that
    /// environment (mutated in place). Shared by <see cref="Run"/> and the subgraph runner
    /// so control-flow ops (If/Loop/Scan) re-enter the same dispatch loop. The
    /// <see cref="GraphContext"/> is wired with a subgraph runner that, for each nested
    /// subgraph, layers the subgraph's feeds over a snapshot of the outer environment so
    /// captured outer-scope names resolve, then runs the subgraph's nodes in isolation.
    /// </summary>
    private Dictionary<string, Tensor> ExecuteNodes(ModelGraph graph, Dictionary<string, Tensor> env)
    {
        var ctx = new GraphContext(env, RunSubgraph);

        foreach (GraphNode node in graph.Nodes)
        {
            if (!_kernels.TryGet(node.OpType, out IKernel? kernel) || kernel is null)
                throw new UnsupportedOperatorException(node.OpType, $"node '{node.Name}'");
            kernel.Execute(node, ctx);
        }

        return env;
    }

    /// <summary>
    /// Subgraph runner installed on every <see cref="GraphContext"/>. Builds a child
    /// environment seeded with (1) the captured outer-scope values, (2) the subgraph's own
    /// initializers, then (3) the per-iteration feeds (formal subgraph inputs) — feeds win
    /// on name collisions. Executes the subgraph and returns its declared outputs by name.
    /// </summary>
    private IReadOnlyDictionary<string, Tensor> RunSubgraph(
        ModelGraph subgraph,
        IReadOnlyDictionary<string, Tensor> feeds,
        IReadOnlyDictionary<string, Tensor> outerValues)
    {
        var childEnv = new Dictionary<string, Tensor>(outerValues);
        foreach (KeyValuePair<string, Tensor> init in subgraph.Initializers)
            childEnv[init.Key] = init.Value;
        foreach (KeyValuePair<string, Tensor> feed in feeds)
            childEnv[feed.Key] = feed.Value;

        Dictionary<string, Tensor> result = ExecuteNodes(subgraph, childEnv);

        var outputs = new Dictionary<string, Tensor>(subgraph.Outputs.Count);
        foreach (string outName in subgraph.Outputs)
        {
            if (!result.TryGetValue(outName, out Tensor? t))
                throw new ModelSharpException(
                    $"Subgraph did not produce declared output '{outName}'.");
            outputs[outName] = t;
        }
        return outputs;
    }

    /// <inheritdoc />
    public void Dispose() { /* No unmanaged resources in Phase 1. */ }
}
