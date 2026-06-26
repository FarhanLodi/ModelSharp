using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// Cast: convert the input tensor element-wise to the dtype named by the required
/// <c>to</c> attribute (an ONNX TensorProto data_type int: FLOAT=1, INT32=6,
/// INT64=7, BOOL=9, DOUBLE=11). Shape is preserved. Float-to-integer conversion
/// truncates toward zero (C-style), and any nonzero value casts to <c>true</c>.
/// </summary>
public sealed class CastKernel : IKernel
{
    public string OpType => "Cast";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        ElementType target = MapOnnxType(Attr.Int(node, "to", 0));

        // Same dtype: pass the tensor through unchanged (Cast is a no-op view here).
        if (target == input.Dtype)
        {
            ctx.Set(node.Outputs[0], input);
            return;
        }

        TensorShape shape = input.Shape;
        double[] src = ReadAsDoubles(input);
        int n = src.Length;

        Tensor result;
        switch (target)
        {
            case ElementType.Float32:
            {
                var buf = new float[n];
                for (int i = 0; i < n; i++) buf[i] = (float)src[i];
                result = new Tensor<float>(shape, buf);
                break;
            }
            case ElementType.Float64:
            {
                // src already holds the values as double; reuse it directly.
                result = new Tensor<double>(shape, src);
                break;
            }
            case ElementType.Int32:
            {
                var buf = new int[n];
                for (int i = 0; i < n; i++) buf[i] = (int)src[i]; // truncates toward zero
                result = new Tensor<int>(shape, buf);
                break;
            }
            case ElementType.Int64:
            {
                var buf = new long[n];
                for (int i = 0; i < n; i++) buf[i] = (long)src[i]; // truncates toward zero
                result = new Tensor<long>(shape, buf);
                break;
            }
            case ElementType.Boolean:
            {
                var buf = new bool[n];
                for (int i = 0; i < n; i++) buf[i] = src[i] != 0d;
                result = new Tensor<bool>(shape, buf);
                break;
            }
            default:
                throw new ModelSharpException($"Cast target dtype {target} is not supported.");
        }

        ctx.Set(node.Outputs[0], result);
    }

    /// <summary>Reads any supported source tensor's values as <c>double</c>
    /// (bool maps to 1/0). Used as a common intermediate for conversion.</summary>
    private static double[] ReadAsDoubles(Tensor src)
    {
        int n = checked((int)src.Length);
        var r = new double[n];
        switch (src.Dtype)
        {
            case ElementType.Float32:
            {
                System.Span<float> s = src.AsFloat().Span;
                for (int i = 0; i < n; i++) r[i] = s[i];
                break;
            }
            case ElementType.Int64:
            {
                System.Span<long> s = src.AsInt64().Span;
                for (int i = 0; i < n; i++) r[i] = s[i];
                break;
            }
            case ElementType.Int32:
            {
                System.Span<int> s = src.AsInt32().Span;
                for (int i = 0; i < n; i++) r[i] = s[i];
                break;
            }
            case ElementType.Boolean:
            {
                System.Span<bool> s = src.AsBool().Span;
                for (int i = 0; i < n; i++) r[i] = s[i] ? 1d : 0d;
                break;
            }
            default:
                throw new ModelSharpException($"Cast source dtype {src.Dtype} is not supported.");
        }
        return r;
    }

    /// <summary>Maps an ONNX TensorProto data_type int (the <c>to</c> attribute)
    /// to a ModelSharp <see cref="ElementType"/>.</summary>
    private static ElementType MapOnnxType(long to) => to switch
    {
        1 => ElementType.Float32,
        6 => ElementType.Int32,
        7 => ElementType.Int64,
        9 => ElementType.Boolean,
        11 => ElementType.Float64,
        _ => throw new ModelSharpException($"Cast 'to' data_type {to} is not supported."),
    };
}
