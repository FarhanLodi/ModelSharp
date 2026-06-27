using System.Collections.Generic;
using ModelSharp.Graph;

namespace ModelSharp.Cpu.Kernels.ControlFlow;

/// <summary>
/// Control-flow-specific attribute readers. Subgraph (GRAPH) attributes are boxed as
/// <see cref="ModelGraph"/> values by the ONNX loader / in-memory test builders.
/// </summary>
internal static class CfAttr
{
    /// <summary>Reads a required GRAPH attribute (a nested subgraph) by name.</summary>
    public static ModelGraph Graph(GraphNode n, string name)
    {
        if (n.Attributes.TryGetValue(name, out object? v) && v is ModelGraph g)
            return g;
        throw new ModelSharpException(
            $"{n.OpType} node '{n.Name}' is missing required subgraph attribute '{name}'.");
    }

    /// <summary>Reads an optional GRAPH attribute, or null when absent.</summary>
    public static ModelGraph? GraphOrNull(GraphNode n, string name)
        => n.Attributes.TryGetValue(name, out object? v) && v is ModelGraph g ? g : null;
}
