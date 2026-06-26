using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// Produces a constant tensor from a node attribute. Supports <c>value</c> (TENSOR,
/// dtype preserved), <c>value_int</c>/<c>value_float</c> (rank-0 scalars), and
/// <c>value_ints</c>/<c>value_floats</c> (1-D tensors). The output dtype matches the
/// attribute. <c>sparse_value</c>/<c>value_string(s)</c> are not supported.
/// </summary>
public sealed class ConstantKernel : IKernel
{
    public string OpType => "Constant";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        string outName = node.Outputs[0];

        // value: a full TENSOR attribute — pass it through with its own dtype.
        if (node.Attributes.TryGetValue("value", out object? tv) && tv is Tensor t)
        {
            ctx.Set(outName, t);
            return;
        }

        // value_ints: 1-D int64 tensor.
        if (node.Attributes.TryGetValue("value_ints", out object? viv) && viv is long[] ints)
        {
            ctx.Set(outName, new Tensor<long>(new TensorShape(ints.Length), ints));
            return;
        }

        // value_floats: 1-D float32 tensor.
        if (node.Attributes.TryGetValue("value_floats", out object? vfv) && vfv is float[] floats)
        {
            ctx.Set(outName, new Tensor<float>(new TensorShape(floats.Length), floats));
            return;
        }

        // value_int: rank-0 int64 scalar.
        if (node.Attributes.TryGetValue("value_int", out object? iv))
        {
            ctx.Set(outName, new Tensor<long>(Scalar(), new[] { System.Convert.ToInt64(iv) }));
            return;
        }

        // value_float: rank-0 float32 scalar.
        if (node.Attributes.TryGetValue("value_float", out object? fv))
        {
            ctx.Set(outName, new Tensor<float>(Scalar(), new[] { System.Convert.ToSingle(fv) }));
            return;
        }

        throw new ModelSharpException(
            $"Constant node '{node.Name}' has no supported value attribute " +
            "(expected one of value, value_int, value_float, value_ints, value_floats).");
    }

    /// <summary>A rank-0 (scalar) shape; <see cref="TensorShape.Length"/> is 1.</summary>
    private static TensorShape Scalar() => new TensorShape(System.Array.Empty<int>());
}
