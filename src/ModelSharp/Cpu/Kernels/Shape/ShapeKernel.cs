using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// Returns the shape of the input as a 1-D Int64 tensor. Honors the optional
/// <c>start</c>/<c>end</c> attributes (opset 15+): the result is the slice
/// <c>dims[start:end]</c>, with negative indices counted from the end and both
/// bounds clamped to <c>[0, rank]</c>. Defaults to the full shape.
/// </summary>
public sealed class ShapeKernel : IKernel
{
    public string OpType => "Shape";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        // Read dtype-agnostically: only the shape matters, not the element data.
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        int rank = input.Shape.Rank;

        int start = ClampAxis(Attr.Int(node, "start", 0), rank);
        int end = ClampAxis(Attr.Int(node, "end", rank), rank);
        int count = end - start;
        if (count < 0) count = 0;

        System.ReadOnlySpan<int> dims = input.Shape.Dimensions;
        var buf = new long[count];
        for (int i = 0; i < count; i++) buf[i] = dims[start + i];

        ctx.Set(node.Outputs[0], new Tensor<long>(new TensorShape(count), buf));
    }

    /// <summary>Resolves a (possibly negative) start/end attribute to an absolute,
    /// clamped axis index in <c>[0, rank]</c>, matching ONNX Shape-15 semantics.</summary>
    private static int ClampAxis(long value, int rank)
    {
        if (value < 0) value += rank;
        if (value < 0) value = 0;
        if (value > rank) value = rank;
        return (int)value;
    }
}
