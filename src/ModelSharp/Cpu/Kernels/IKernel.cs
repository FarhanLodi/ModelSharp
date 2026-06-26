using ModelSharp.Graph;

namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Executes one operator. Implementations read named inputs from the
/// <see cref="GraphContext"/> and write named outputs back. Keep them
/// allocation-light and SIMD-friendly.
/// </summary>
public interface IKernel
{
    /// <summary>The ONNX op type this kernel handles (e.g. "Relu", "Add", "Conv").</summary>
    string OpType { get; }

    /// <summary>Runs the operator against the execution context.</summary>
    void Execute(GraphNode node, GraphContext ctx);
}
