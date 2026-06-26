using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Trilu</c>: keeps the upper (<c>upper=1</c>, default) or lower (<c>upper=0</c>) triangle of
/// the trailing 2-D matrices, zeroing the rest. The optional scalar <c>k</c> input shifts the diagonal
/// (positive moves it up-and-right). Operates batched over all leading dims. Used for causal masks.
/// Dtype-preserving (Float32 / Int64 / Int32 / Boolean).
/// </summary>
public sealed class TriluKernel : IKernel
{
    public string OpType => "Trilu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        bool upper = Attr.Int(node, "upper", 1) != 0;
        long k = 0;
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
            k = TensorInts.Read(ctx.GetTensor(node.Inputs[1]))[0];

        ctx.Set(node.Outputs[0], data.Dtype switch
        {
            ElementType.Int64 => Trilu<long>(data.AsInt64(), k, upper),
            ElementType.Int32 => Trilu<int>(data.AsInt32(), k, upper),
            ElementType.Boolean => Trilu<bool>(data.AsBool(), k, upper),
            _ => Trilu<float>(data.AsFloat(), k, upper),
        });
    }

    private static Tensor<T> Trilu<T>(Tensor<T> x, long k, bool upper) where T : unmanaged
    {
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        if (rank < 2) throw new ModelSharpException("Trilu requires a tensor of rank >= 2.");
        int rows = dims[rank - 2], cols = dims[rank - 1];
        int batch = 1; for (int i = 0; i < rank - 2; i++) batch *= dims[i];

        var y = new Tensor<T>(x.Shape);
        Span<T> xs = x.Span, ys = y.Span;
        int mat = rows * cols;
        T zero = default;

        for (int b = 0; b < batch; b++)
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            int idx = b * mat + i * cols + j;
            bool keep = upper ? (long)j >= (long)i + k : (long)j <= (long)i + k;
            ys[idx] = keep ? xs[idx] : zero;
        }
        return y;
    }
}
