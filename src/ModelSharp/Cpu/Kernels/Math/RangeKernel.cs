using System;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Math;

/// <summary>
/// ONNX <c>Range</c>: generates the 1-D sequence <c>start, start+delta, ...</c> stopping before
/// <c>limit</c>. The output dtype follows the scalar inputs (Float32 / Int64 / Int32 supported).
/// </summary>
public sealed class RangeKernel : IKernel
{
    public string OpType => "Range";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor start = ctx.GetTensor(node.Inputs[0]);
        Tensor limit = ctx.GetTensor(node.Inputs[1]);
        Tensor delta = ctx.GetTensor(node.Inputs[2]);

        switch (start.Dtype)
        {
            case ElementType.Int64:
            {
                long s = start.AsInt64().Span[0], l = limit.AsInt64().Span[0], d = delta.AsInt64().Span[0];
                int n = Count(s, l, d);
                var buf = new long[n];
                for (int i = 0; i < n; i++) buf[i] = s + (long)i * d;
                ctx.Set(node.Outputs[0], new Tensor<long>(new TensorShape(n), buf));
                break;
            }
            case ElementType.Int32:
            {
                int s = start.AsInt32().Span[0], l = limit.AsInt32().Span[0], d = delta.AsInt32().Span[0];
                int n = Count(s, l, d);
                var buf = new int[n];
                for (int i = 0; i < n; i++) buf[i] = s + i * d;
                ctx.Set(node.Outputs[0], new Tensor<int>(new TensorShape(n), buf));
                break;
            }
            default:
            {
                float s = start.AsFloat().Span[0], l = limit.AsFloat().Span[0], d = delta.AsFloat().Span[0];
                int n = Count(s, l, d);
                var buf = new float[n];
                for (int i = 0; i < n; i++) buf[i] = s + i * d;
                ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(n), buf));
                break;
            }
        }
    }

    /// <summary>Element count = max(ceil((limit − start) / delta), 0), per the ONNX spec.</summary>
    private static int Count(double start, double limit, double delta)
    {
        if (delta == 0d) throw new ModelSharpException("Range 'delta' cannot be 0.");
        double c = System.Math.Ceiling((limit - start) / delta);
        return c <= 0d ? 0 : (int)c;
    }
}
