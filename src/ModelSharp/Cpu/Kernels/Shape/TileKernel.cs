using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Tile</c>: repeats <c>data</c> along each axis by the counts in the 1-D <c>repeats</c>
/// input (length == rank). Output axis <c>i</c> has size <c>dims[i] * repeats[i]</c>.
/// Dtype-preserving (Float32 / Int64 / Int32 / Boolean).
/// </summary>
public sealed class TileKernel : IKernel
{
    public string OpType => "Tile";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        long[] reps = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));

        ctx.Set(node.Outputs[0], data.Dtype switch
        {
            ElementType.Int64 => Tile<long>(data.AsInt64(), reps),
            ElementType.Int32 => Tile<int>(data.AsInt32(), reps),
            ElementType.Boolean => Tile<bool>(data.AsBool(), reps),
            _ => Tile<float>(data.AsFloat(), reps),
        });
    }

    private static Tensor<T> Tile<T>(Tensor<T> x, long[] reps) where T : unmanaged
    {
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        int[] inStrides = Nd.Strides(dims);

        var outDims = new int[rank];
        for (int i = 0; i < rank; i++) outDims[i] = dims[i] * (int)reps[i];

        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> xs = x.Span, ys = y.Span;
        int n = (int)y.Shape.Length;
        var coord = new int[rank];

        for (int idx = 0; idx < n; idx++)
        {
            int src = 0;
            for (int i = 0; i < rank; i++) src += (coord[i] % dims[i]) * inStrides[i];
            ys[idx] = xs[src];
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }
        return y;
    }
}
