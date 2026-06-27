using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Graph;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Sequence;
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

        // Top-level graph: tensor inputs/outputs only (the public contract). Sequence/optional
        // values live solely on the intra-graph wire, so we read the tensor env for outputs.
        ExecuteNodes(_graph, env);

        // Collect the requested outputs.
        var result = new Dictionary<string, NamedTensor>();
        foreach (string outName in _graph.Outputs)
            result[outName] = new NamedTensor(outName, env[outName]);
        return result;
    }

    /// <summary>
    /// Executes a graph's nodes against an already-seeded environment and returns the
    /// <see cref="GraphContext"/> (whose tensor env is <paramref name="env"/>, mutated in place,
    /// and whose sequence/optional map holds any non-tensor values produced). Shared by
    /// <see cref="Run"/> and the subgraph runner so control-flow ops (If/Loop/Scan) re-enter the
    /// same dispatch loop. <paramref name="seedSeqValues"/> seeds the captured outer-scope
    /// sequence/optional values when entering a subgraph body (null for the top-level graph).
    /// </summary>
    private GraphContext ExecuteNodes(
        ModelGraph graph,
        Dictionary<string, Tensor> env,
        IReadOnlyDictionary<string, SeqValue>? seedSeqValues = null)
    {
        var ctx = new GraphContext(env, RunSubgraph, seedSeqValues);

        foreach (GraphNode node in graph.Nodes)
        {
            if (!_kernels.TryGet(node.OpType, out IKernel? kernel) || kernel is null)
                throw new UnsupportedOperatorException(node.OpType, $"node '{node.Name}'");
            kernel.Execute(node, ctx);
        }

        return ctx;
    }

    /// <summary>
    /// Subgraph runner installed on every <see cref="GraphContext"/>. Builds a child
    /// environment seeded with (1) the captured outer-scope tensor values, (2) the subgraph's own
    /// initializers, then (3) the per-iteration feeds (formal subgraph inputs) — feeds win on name
    /// collisions. The captured outer-scope sequence/optional values (<paramref name="outerSeq"/>)
    /// are seeded into the child context as well so a subgraph body can read an outer
    /// <c>Sequence*</c>/<c>Optional*</c> value. Executes the subgraph and returns its declared
    /// outputs split into tensor outputs and sequence/optional outputs.
    /// </summary>
    private GraphContext.SubgraphResult RunSubgraph(
        ModelGraph subgraph,
        IReadOnlyDictionary<string, Tensor> feeds,
        IReadOnlyDictionary<string, Tensor> outerValues,
        IReadOnlyDictionary<string, SeqValue>? outerSeq)
    {
        var childEnv = new Dictionary<string, Tensor>(outerValues);
        foreach (KeyValuePair<string, Tensor> init in subgraph.Initializers)
            childEnv[init.Key] = init.Value;
        foreach (KeyValuePair<string, Tensor> feed in feeds)
            childEnv[feed.Key] = feed.Value;

        GraphContext childCtx = ExecuteNodes(subgraph, childEnv, outerSeq);
        IReadOnlyDictionary<string, SeqValue>? childSeq = childCtx.SeqValues;

        var tensorOuts = new Dictionary<string, Tensor>(subgraph.Outputs.Count);
        Dictionary<string, SeqValue>? seqOuts = null;
        foreach (string outName in subgraph.Outputs)
        {
            // A declared output is either a tensor or a sequence/optional value. Resolve it from
            // whichever map produced it; prefer the tensor binding (the common case).
            if (childEnv.TryGetValue(outName, out Tensor? t))
            {
                tensorOuts[outName] = t;
            }
            else if (childSeq is not null && childSeq.TryGetValue(outName, out SeqValue? sv))
            {
                (seqOuts ??= new Dictionary<string, SeqValue>())[outName] = sv;
            }
            else
            {
                throw new ModelSharpException(
                    $"Subgraph did not produce declared output '{outName}'.");
            }
        }

        return new GraphContext.SubgraphResult(
            tensorOuts,
            (IReadOnlyDictionary<string, SeqValue>?)seqOuts ?? EmptySeq);
    }

    /// <summary>Shared empty sequence-output map for the (common) tensor-only subgraph case.</summary>
    private static readonly Dictionary<string, SeqValue> EmptySeq = new();

    /// <inheritdoc />
    public void Dispose() { /* No unmanaged resources in Phase 1. */ }
}
