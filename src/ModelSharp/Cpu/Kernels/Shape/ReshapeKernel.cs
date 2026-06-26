using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>Reshape data to the shape given by the second input (int64/int32/float); supports 0 and -1.
/// Dtype-preserving view over the data buffer.</summary>
public sealed class ReshapeKernel : IKernel
{
    public string OpType => "Reshape";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        long[] sp = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        System.ReadOnlySpan<int> inDims = data.Shape.Dimensions;
        long total = data.Length;

        int n = sp.Length;
        var dims = new int[n];
        int inferIdx = -1;
        long known = 1;
        for (int i = 0; i < n; i++)
        {
            long d = sp[i];
            if (d == -1) { inferIdx = i; }
            else if (d == 0) { dims[i] = inDims[i]; known *= dims[i]; }
            else { dims[i] = (int)d; known *= d; }
        }
        if (inferIdx >= 0) dims[inferIdx] = (int)(total / known);

        ctx.Set(node.Outputs[0], data.WithShape(new TensorShape(dims)));
    }
}
