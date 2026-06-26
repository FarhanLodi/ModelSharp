using System.Collections.Generic;
using ModelSharp.Tensors;

namespace ModelSharp.Graph;

/// <summary>
/// An in-memory model graph — the engine-agnostic intermediate representation.
/// The ONNX loader populates this from a .onnx file; tests build it directly;
/// any <c>IExecutionEngine</c> consumes it.
/// </summary>
public sealed class ModelGraph
{
    /// <summary>Names of the graph's required inputs (feeds), excluding initializers.</summary>
    public IReadOnlyList<string> Inputs { get; init; } = new List<string>();

    /// <summary>Names of the graph's outputs to return.</summary>
    public IReadOnlyList<string> Outputs { get; init; } = new List<string>();

    /// <summary>Nodes in topological execution order.</summary>
    public IReadOnlyList<GraphNode> Nodes { get; init; } = new List<GraphNode>();

    /// <summary>Constant tensors baked into the model (weights, biases, shape constants).
    /// Values carry their own dtype (Float32, Int64, Int32, Boolean, ...).</summary>
    public IReadOnlyDictionary<string, Tensor> Initializers { get; init; }
        = new Dictionary<string, Tensor>();

    /// <summary>
    /// Free-form key/value metadata embedded in the ONNX model (ModelProto.metadata_props).
    /// Surfaced verbatim so the manifest resolver can recognize keys like
    /// <c>task</c>, <c>layout</c>, <c>labels</c>, <c>mean</c>, <c>std</c>, <c>vocab</c>.
    /// Empty when the model carries no metadata or was built in-memory.
    /// </summary>
    public IReadOnlyDictionary<string, string> MetadataProps { get; init; }
        = new Dictionary<string, string>();
}
