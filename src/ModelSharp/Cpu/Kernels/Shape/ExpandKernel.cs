using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Expand</c>: broadcasts the first input to the shape given by the second
/// input (a 1-D shape tensor), following bidirectional (NumPy-style) broadcasting.
/// The output shape is the broadcast of the input shape and the requested shape, so
/// the result can be larger than the requested shape where the input dimension is.
/// The input dtype (Float32 / Int64 / Int32 / Boolean) is preserved.
/// </summary>
public sealed class ExpandKernel : IKernel
{
    public string OpType => "Expand";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        long[] requested = ReadShape(ctx.GetTensor(node.Inputs[1]));

        var requestedDims = new int[requested.Length];
        for (int i = 0; i < requested.Length; i++) requestedDims[i] = (int)requested[i];

        int[] outd = Nd.BroadcastShape(input.Shape.Dimensions, requestedDims);

        Tensor result = input.Dtype switch
        {
            ElementType.Int64 => Broadcast(input.AsInt64(), outd),
            ElementType.Int32 => Broadcast(input.AsInt32(), outd),
            ElementType.Boolean => Broadcast(input.AsBool(), outd),
            _ => Broadcast(input.AsFloat(), outd),
        };

        ctx.Set(node.Outputs[0], result);
    }

    /// <summary>Copies <paramref name="input"/> into the broadcast output shape <paramref name="outd"/>.</summary>
    private static Tensor<T> Broadcast<T>(Tensor<T> input, int[] outd) where T : unmanaged
    {
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] stride = Nd.BroadcastStrides(input.Shape.Dimensions, rank);

        var y = new Tensor<T>(outShape);
        Span<T> src = input.Span, dst = y.Span;
        int n = (int)outShape.Length;
        var coord = new int[rank];
        int off = 0;
        for (int idx = 0; idx < n; idx++)
        {
            dst[idx] = src[off];
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                off += stride[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                off -= stride[ax] * outd[ax];
            }
        }
        return y;
    }

    /// <summary>Reads a 1-D shape tensor's values as int64 regardless of its dtype
    /// (Int64/Int32/Float32 are all accepted, mirroring how ONNX exporters emit it).</summary>
    private static long[] ReadShape(Tensor t)
    {
        switch (t.Dtype)
        {
            case ElementType.Int64:
            {
                Span<long> s = t.AsInt64().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i];
                return r;
            }
            case ElementType.Int32:
            {
                Span<int> s = t.AsInt32().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i];
                return r;
            }
            default:
            {
                Span<float> s = t.AsFloat().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = (long)MathF.Round(s[i]);
                return r;
            }
        }
    }
}
