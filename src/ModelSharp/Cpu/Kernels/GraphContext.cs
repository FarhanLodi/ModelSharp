using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Sequence;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels;

/// <summary>The mutable name → tensor environment threaded through kernel execution.</summary>
public sealed class GraphContext
{
    private readonly Dictionary<string, Tensor> _values;

    /// <summary>
    /// Parallel name → non-tensor value map carrying the ONNX <c>Sequence*</c> / <c>Optional*</c>
    /// runtime values (<see cref="SeqValue"/>). This is kept <b>separate</b> from the tensor map
    /// so the established tensor plumbing (<see cref="Get"/>/<see cref="Set"/>/<see cref="Values"/>,
    /// the subgraph runner, the public <c>Run</c> contract) is completely unaffected: tensor-only
    /// graphs never touch this map. Sequence/optional values live only on the wire between nodes;
    /// graph inputs/outputs remain tensors. Lazily allocated — null until a sequence/optional op
    /// actually produces one.
    /// </summary>
    private Dictionary<string, SeqValue>? _seqValues;

    /// <summary>
    /// Result of running a nested subgraph: its declared outputs, partitioned into the plain
    /// tensor outputs and the (typically empty) sequence/optional outputs. Keeping the two
    /// kinds separate lets the calling kernel propagate each back into the correct parent slot
    /// (tensor map vs. <see cref="_seqValues"/>) without inspecting types here.
    /// </summary>
    public readonly struct SubgraphResult
    {
        public SubgraphResult(
            IReadOnlyDictionary<string, Tensor> tensors,
            IReadOnlyDictionary<string, SeqValue> sequences)
        {
            Tensors = tensors;
            Sequences = sequences;
        }

        /// <summary>Subgraph outputs that are plain tensors, keyed by output name.</summary>
        public IReadOnlyDictionary<string, Tensor> Tensors { get; }

        /// <summary>Subgraph outputs that are sequence/optional values, keyed by output name.</summary>
        public IReadOnlyDictionary<string, SeqValue> Sequences { get; }
    }

    /// <summary>
    /// Hook that executes a nested ONNX subgraph (the value of a GRAPH attribute on a
    /// control-flow node) and returns its outputs by graph-output name. Control-flow
    /// kernels (<c>If</c>/<c>Loop</c>/<c>Scan</c>) invoke this via
    /// <see cref="RunSubgraph(ModelGraph, IReadOnlyDictionary{string, Tensor})"/>.
    /// The runner seeds a child environment with the current outer-scope values (so
    /// subgraph nodes can capture outer names) plus the supplied per-iteration feeds. The
    /// captured outer scope includes both tensor values and the sequence/optional snapshot
    /// (4th argument), so a subgraph body can read an outer <c>Sequence*</c>/<c>Optional*</c>
    /// value (e.g. a <c>SequenceAt</c> over a sequence built in the parent graph).
    /// Null when no engine installed a runner (e.g. a bare context built in a unit test);
    /// in that case control-flow ops throw a clear error.
    /// </summary>
    private readonly Func<
        ModelGraph,
        IReadOnlyDictionary<string, Tensor>,
        IReadOnlyDictionary<string, Tensor>,
        IReadOnlyDictionary<string, SeqValue>?,
        SubgraphResult>? _subgraphRunner;

    /// <summary>Snapshot of every name currently visible (initializers, feeds, and produced
    /// intermediates) — the outer scope a subgraph captures from.</summary>
    internal IReadOnlyDictionary<string, Tensor> Values => _values;

    public GraphContext(Dictionary<string, Tensor> values) : this(values, null) { }

    internal GraphContext(
        Dictionary<string, Tensor> values,
        Func<
            ModelGraph,
            IReadOnlyDictionary<string, Tensor>,
            IReadOnlyDictionary<string, Tensor>,
            IReadOnlyDictionary<string, SeqValue>?,
            SubgraphResult>? subgraphRunner,
        IReadOnlyDictionary<string, SeqValue>? seedSeqValues = null)
    {
        _values = values;
        _subgraphRunner = subgraphRunner;
        // Seed captured outer-scope sequence/optional values (used when entering a subgraph body),
        // copied into a fresh owned map so the child can shadow/extend names without mutating the
        // parent snapshot. Null/empty when none — tensor-only graphs stay on the cheap path.
        if (seedSeqValues is { Count: > 0 })
            _seqValues = new Dictionary<string, SeqValue>(seedSeqValues);
    }

    /// <summary>
    /// Snapshot of the sequence/optional values currently visible (null when none have been
    /// produced yet) — the outer-scope <see cref="SeqValue"/> bindings a subgraph captures from,
    /// alongside <see cref="Values"/>. Internal so the engine's subgraph runner can seed the
    /// child environment; tensor-only graphs leave this null and pay nothing.
    /// </summary>
    internal IReadOnlyDictionary<string, SeqValue>? SeqValues => _seqValues;

    /// <summary>
    /// Reads a tensor by name as float32, throwing if it hasn't been produced yet
    /// or if its dtype is not Float32. Existing float kernels use this overload.
    /// </summary>
    public Tensor<float> Get(string name) => GetTensor(name).AsFloat();

    /// <summary>
    /// Reads a tensor by name preserving its dtype, throwing if it hasn't been
    /// produced yet. Dtype-aware kernels use this and then call
    /// <see cref="Tensor.AsInt64"/>/<see cref="Tensor.AsBool"/>/etc.
    /// </summary>
    public Tensor GetTensor(string name) =>
        _values.TryGetValue(name, out Tensor? t)
            ? t
            : throw new ModelSharpException($"Tensor '{name}' is not available in the execution context.");

    /// <summary>True if a tensor with the given (non-empty) name is bound in this scope.</summary>
    public bool Has(string name) => name.Length != 0 && _values.ContainsKey(name);

    /// <summary>Writes a tensor by name (any dtype; <c>Tensor&lt;float&gt;</c> upcasts implicitly).</summary>
    public void Set(string name, Tensor value) => _values[name] = value;

    // ---- Sequence / Optional non-tensor value plumbing (additive) -----------------------------

    /// <summary>
    /// Reads a non-tensor value (sequence or optional) by name. Throws if the name is not bound
    /// as a <see cref="SeqValue"/> (a tensor bound under the same name is a separate slot).
    /// </summary>
    public SeqValue GetSeq(string name) =>
        _seqValues is not null && _seqValues.TryGetValue(name, out SeqValue? v)
            ? v
            : throw new ModelSharpException(
                $"Sequence/optional value '{name}' is not available in the execution context.");

    /// <summary>True if a non-tensor (sequence/optional) value with the given name is bound.</summary>
    public bool HasSeq(string name) =>
        name.Length != 0 && _seqValues is not null && _seqValues.ContainsKey(name);

    /// <summary>Writes a non-tensor (sequence/optional) value by name.</summary>
    public void SetSeq(string name, SeqValue value)
    {
        _seqValues ??= new Dictionary<string, SeqValue>();
        _seqValues[name] = value;
    }

    /// <summary>
    /// Executes a nested subgraph with outer-scope capture and returns its <b>tensor</b> outputs
    /// keyed by the subgraph's declared output names. <paramref name="feeds"/> supplies the
    /// subgraph's own formal inputs (e.g. a Loop body's iteration count and carried values);
    /// the current outer scope — both tensors and the sequence/optional snapshot — is captured
    /// automatically so subgraph nodes can reference outer names (including reading an outer
    /// <c>Sequence*</c>/<c>Optional*</c> value).
    ///
    /// <para>
    /// Any sequence/optional values declared as subgraph outputs are propagated <i>side-band</i>
    /// into this (parent) context's <see cref="_seqValues"/> map under their subgraph-output name,
    /// so a control-flow body can produce a sequence/optional that flows back out. For each such
    /// non-tensor output a benign empty placeholder tensor is also returned under the same name,
    /// so the control-flow kernels' positional output mapping (which copies tensors by output
    /// name) stays well-defined; the real value is the <see cref="SeqValue"/> just bound.
    /// </para>
    ///
    /// Throws if no engine installed a subgraph runner.
    /// </summary>
    public IReadOnlyDictionary<string, Tensor> RunSubgraph(
        ModelGraph subgraph, IReadOnlyDictionary<string, Tensor> feeds)
    {
        if (_subgraphRunner is null)
            throw new ModelSharpException(
                "This GraphContext has no subgraph runner; control-flow ops (If/Loop/Scan) "
                + "require execution through ManagedCpuEngine.");

        SubgraphResult result = _subgraphRunner(subgraph, feeds, _values, _seqValues);

        // Fast path: tensor-only subgraph (the overwhelmingly common case). No seq/optional
        // outputs were produced, so return the tensor map untouched — zero behavioral change.
        if (result.Sequences.Count == 0)
            return result.Tensors;

        // The subgraph produced one or more sequence/optional outputs. Bind each into THIS
        // context's seq map (so the value crosses the subgraph boundary outward) and surface a
        // placeholder tensor of the same name so a frozen control-flow kernel's
        // "ctx.Set(nodeOut, outs[subOut])" remains well-defined.
        var tensors = new Dictionary<string, Tensor>(result.Tensors);
        foreach (KeyValuePair<string, SeqValue> seqOut in result.Sequences)
        {
            SetSeq(seqOut.Key, seqOut.Value);
            tensors.TryAdd(seqOut.Key, SeqPlaceholder);
        }
        return tensors;
    }

    /// <summary>A shared, empty UInt8 placeholder stand-in for a sequence/optional subgraph output
    /// in the tensor-keyed return of <see cref="RunSubgraph"/> (the real value lives in the seq map).</summary>
    private static readonly Tensor SeqPlaceholder =
        new Tensor<byte>(new TensorShape(0), System.Array.Empty<byte>());
}
