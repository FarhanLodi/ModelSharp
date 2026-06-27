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
    /// Hook that executes a nested ONNX subgraph (the value of a GRAPH attribute on a
    /// control-flow node) and returns its outputs by graph-output name. Control-flow
    /// kernels (<c>If</c>/<c>Loop</c>/<c>Scan</c>) invoke this via
    /// <see cref="RunSubgraph(ModelGraph, IReadOnlyDictionary{string, Tensor})"/>.
    /// The runner seeds a child environment with the current outer-scope values (so
    /// subgraph nodes can capture outer names) plus the supplied per-iteration feeds.
    /// Null when no engine installed a runner (e.g. a bare context built in a unit test);
    /// in that case control-flow ops throw a clear error.
    /// </summary>
    private readonly Func<ModelGraph, IReadOnlyDictionary<string, Tensor>, IReadOnlyDictionary<string, Tensor>, IReadOnlyDictionary<string, Tensor>>? _subgraphRunner;

    /// <summary>Snapshot of every name currently visible (initializers, feeds, and produced
    /// intermediates) — the outer scope a subgraph captures from.</summary>
    internal IReadOnlyDictionary<string, Tensor> Values => _values;

    public GraphContext(Dictionary<string, Tensor> values) : this(values, null) { }

    internal GraphContext(
        Dictionary<string, Tensor> values,
        Func<ModelGraph, IReadOnlyDictionary<string, Tensor>, IReadOnlyDictionary<string, Tensor>, IReadOnlyDictionary<string, Tensor>>? subgraphRunner)
    {
        _values = values;
        _subgraphRunner = subgraphRunner;
    }

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
            : throw new KeyNotFoundException($"Tensor '{name}' is not available in the execution context.");

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
            : throw new KeyNotFoundException(
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
    /// Executes a nested subgraph with outer-scope capture and returns its outputs keyed
    /// by the subgraph's declared output names. <paramref name="feeds"/> supplies the
    /// subgraph's own formal inputs (e.g. a Loop body's iteration count and carried values);
    /// the current outer scope is captured automatically so subgraph nodes can reference
    /// outer tensor names. Throws if no engine installed a subgraph runner.
    /// </summary>
    public IReadOnlyDictionary<string, Tensor> RunSubgraph(
        ModelGraph subgraph, IReadOnlyDictionary<string, Tensor> feeds)
    {
        if (_subgraphRunner is null)
            throw new ModelSharpException(
                "This GraphContext has no subgraph runner; control-flow ops (If/Loop/Scan) "
                + "require execution through ManagedCpuEngine.");
        return _subgraphRunner(subgraph, feeds, _values);
    }
}
