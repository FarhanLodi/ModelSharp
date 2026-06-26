using System;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>ConstantOfShape</c>: produces a tensor whose shape is given by the 1-D
/// integer input, with every element set to the scalar <c>value</c> attribute.
/// The output dtype follows the <c>value</c> tensor's dtype; when <c>value</c> is
/// omitted the output is a float32 tensor filled with 0.
/// </summary>
public sealed class ConstantOfShapeKernel : IKernel
{
    public string OpType => "ConstantOfShape";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        // Input is a 1-D tensor of element counts per axis (int64 per spec; we also
        // accept int32/float32 the way ONNX exporters sometimes emit shape tensors).
        long[] shapeVals = ReadShape(ctx.GetTensor(node.Inputs[0]));
        var dims = new int[shapeVals.Length];
        for (int i = 0; i < shapeVals.Length; i++) dims[i] = checked((int)shapeVals[i]);
        var shape = new TensorShape(dims);
        int count = checked((int)shape.Length);

        // The optional 'value' attribute is a one-element tensor whose dtype drives
        // the output dtype (default: a single float32 zero).
        Tensor? value = node.Attributes.TryGetValue("value", out object? v) ? v as Tensor : null;
        ElementType dtype = value?.Dtype ?? ElementType.Float32;

        switch (dtype)
        {
            case ElementType.Int64:
            {
                Span<long> vs = value!.AsInt64().Span;
                var buf = new long[count];
                Array.Fill(buf, vs.Length > 0 ? vs[0] : 0L);
                ctx.Set(node.Outputs[0], new Tensor<long>(shape, buf));
                break;
            }
            case ElementType.Int32:
            {
                Span<int> vs = value!.AsInt32().Span;
                var buf = new int[count];
                Array.Fill(buf, vs.Length > 0 ? vs[0] : 0);
                ctx.Set(node.Outputs[0], new Tensor<int>(shape, buf));
                break;
            }
            case ElementType.Boolean:
            {
                Span<bool> vs = value!.AsBool().Span;
                var buf = new bool[count];
                Array.Fill(buf, vs.Length > 0 && vs[0]);
                ctx.Set(node.Outputs[0], new Tensor<bool>(shape, buf));
                break;
            }
            case ElementType.Float32:
            {
                float fill = 0f;
                if (value is not null)
                {
                    Span<float> vs = value.AsFloat().Span;
                    if (vs.Length > 0) fill = vs[0];
                }
                var buf = new float[count];
                Array.Fill(buf, fill);
                ctx.Set(node.Outputs[0], new Tensor<float>(shape, buf));
                break;
            }
            default:
                throw new ModelSharpException(
                    $"ConstantOfShape: unsupported 'value' dtype {dtype}.");
        }
    }

    /// <summary>Reads a 1-D shape tensor's values as int64 regardless of its dtype
    /// (Int64/Int32/Float32 are all accepted), mirroring <c>ReshapeKernel.ReadShape</c>.</summary>
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
