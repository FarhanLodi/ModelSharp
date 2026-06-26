using System;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Internal;

/// <summary>
/// Reads an integer-ish tensor's values as int64 regardless of its actual dtype
/// (Int64 / Int32 / Bool / Float all accepted). Used for shape tensors, axes, and indices,
/// which ONNX exporters emit with varying element types.
/// </summary>
internal static class TensorInts
{
    public static long[] Read(Tensor t)
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
            case ElementType.Boolean:
            {
                Span<bool> s = t.AsBool().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i] ? 1 : 0;
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
