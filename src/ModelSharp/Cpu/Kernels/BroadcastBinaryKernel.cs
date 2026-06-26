using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Shared elementwise-with-broadcasting machinery for binary ops (Add, Mul, ...).
/// Fast path for equal shapes; NumPy-style broadcasting otherwise.
/// </summary>
public abstract class BroadcastBinaryKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The scalar operation applied elementwise.</summary>
    protected abstract float Apply(float a, float b);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> a = ctx.Get(node.Inputs[0]);
        Tensor<float> b = ctx.Get(node.Inputs[1]);

        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new Tensor<float>(a.Shape);
            System.Span<float> sa = a.Span, sb = b.Span, sy = yEqual.Span;
            for (int i = 0; i < sy.Length; i++) sy[i] = Apply(sa[i], sb[i]);
            ctx.Set(node.Outputs[0], yEqual);
            return;
        }

        int[] outd = Nd.BroadcastShape(a.Shape.Dimensions, b.Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = Nd.BroadcastStrides(a.Shape.Dimensions, rank);
        int[] strideB = Nd.BroadcastStrides(b.Shape.Dimensions, rank);

        var y = new Tensor<float>(outShape);
        System.Span<float> da = a.Span, db = b.Span, dy = y.Span;
        int n = (int)outShape.Length;
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            dy[idx] = Apply(da[aOff], db[bOff]);
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                aOff += strideA[ax];
                bOff += strideB[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                aOff -= strideA[ax] * outd[ax];
                bOff -= strideB[ax] * outd[ax];
            }
        }
        ctx.Set(node.Outputs[0], y);
    }
}
