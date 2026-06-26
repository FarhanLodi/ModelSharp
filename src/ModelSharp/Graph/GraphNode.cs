using System.Collections.Generic;

namespace ModelSharp.Graph;

/// <summary>A single operation in a model graph (one ONNX node).</summary>
public sealed class GraphNode
{
    /// <summary>The ONNX op type (e.g. "Conv", "Add").</summary>
    public string OpType { get; }

    /// <summary>The node's name (for diagnostics).</summary>
    public string Name { get; }

    /// <summary>Names of the input tensors, in order. Empty strings denote omitted optional inputs.</summary>
    public IReadOnlyList<string> Inputs { get; }

    /// <summary>Names of the output tensors, in order.</summary>
    public IReadOnlyList<string> Outputs { get; }

    /// <summary>
    /// Operator attributes keyed by name. Values are boxed as: <c>long</c> (INT),
    /// <c>float</c> (FLOAT), <c>string</c> (STRING), <c>long[]</c> (INTS),
    /// <c>float[]</c> (FLOATS), or <c>Tensor</c> (TENSOR; the dtype-carrying base).
    /// </summary>
    public IReadOnlyDictionary<string, object> Attributes { get; }

    public GraphNode(
        string opType,
        string name,
        IReadOnlyList<string> inputs,
        IReadOnlyList<string> outputs,
        IReadOnlyDictionary<string, object>? attributes = null)
    {
        OpType = opType;
        Name = name;
        Inputs = inputs;
        Outputs = outputs;
        Attributes = attributes ?? new Dictionary<string, object>();
    }
}
