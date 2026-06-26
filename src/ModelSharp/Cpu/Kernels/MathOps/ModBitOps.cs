using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.MathOps;

/// <summary>
/// Shared NumPy-style broadcasting machinery for dtype-generic binary integer/float ops.
/// Mirrors <see cref="BroadcastBinaryKernel"/> but stays generic over the element type so the
/// concrete op (Mod, BitShift) can run over Int64 / Int32 / UInt8 / Float32 without boxing.
/// </summary>
internal static class TypedBroadcast
{
    /// <summary>Applies <paramref name="op"/> elementwise with NumPy broadcasting, allocating the output.</summary>
    public static Tensor<T> Apply<T>(Tensor<T> a, Tensor<T> b, Func<T, T, T> op) where T : unmanaged
    {
        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new Tensor<T>(a.Shape);
            Span<T> sa = a.Span, sb = b.Span, sy = yEqual.Span;
            for (int i = 0; i < sy.Length; i++) sy[i] = op(sa[i], sb[i]);
            return yEqual;
        }

        int[] outd = Nd.BroadcastShape(a.Shape.Dimensions, b.Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = Nd.BroadcastStrides(a.Shape.Dimensions, rank);
        int[] strideB = Nd.BroadcastStrides(b.Shape.Dimensions, rank);

        var y = new Tensor<T>(outShape);
        Span<T> da = a.Span, db = b.Span, dy = y.Span;
        int n = (int)outShape.Length;
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            dy[idx] = op(da[aOff], db[bOff]);
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
        return y;
    }
}

/// <summary>
/// ONNX <c>Mod</c>: elementwise remainder with NumPy broadcasting. The <c>fmod</c> attribute
/// (default 0) selects the semantics: <c>fmod=0</c> is integer modulo taking the sign of the
/// divisor (Python <c>%</c>) and is only valid for integer dtypes; <c>fmod=1</c> is C
/// <c>fmod</c> (sign of the dividend) and works for floats. Supports Int64 / Int32 / Float32.
/// </summary>
public sealed class ModKernel : IKernel
{
    public string OpType => "Mod";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor a = ctx.GetTensor(node.Inputs[0]);
        Tensor b = ctx.GetTensor(node.Inputs[1]);
        bool fmod = Attr.Int(node, "fmod", 0) != 0;

        switch (a.Dtype)
        {
            case ElementType.Int64:
            {
                Func<long, long, long> op = fmod ? CFmod : PyMod;
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(a.AsInt64(), b.AsInt64(), op));
                break;
            }
            case ElementType.Int32:
            {
                Func<int, int, int> op = fmod ? CFmod : PyMod;
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(a.AsInt32(), b.AsInt32(), op));
                break;
            }
            case ElementType.Float32:
            {
                if (!fmod)
                    throw new ModelSharpException(
                        "Mod with fmod=0 is only defined for integer tensors; set fmod=1 for float inputs.");
                var y = TypedBroadcast.Apply(a.AsFloat(), b.AsFloat(), Fmod);
                ctx.Set(node.Outputs[0], y);
                break;
            }
            default:
                throw new ModelSharpException($"Mod: unsupported dtype {a.Dtype}.");
        }
    }

    // Python-style modulo: result has the sign of the divisor.
    private static long PyMod(long x, long m) => m == 0 ? throw new ModelSharpException("Mod by zero.") : ((x % m) + m) % m;
    private static int PyMod(int x, int m) => m == 0 ? throw new ModelSharpException("Mod by zero.") : ((x % m) + m) % m;

    // C fmod-style: result has the sign of the dividend (truncated division remainder).
    private static long CFmod(long x, long m) => m == 0 ? throw new ModelSharpException("Mod by zero.") : x % m;
    private static int CFmod(int x, int m) => m == 0 ? throw new ModelSharpException("Mod by zero.") : x % m;

    // C fmod for floats: sign of the dividend.
    private static float Fmod(float x, float m) => x % m;
}

/// <summary>
/// ONNX <c>BitShift</c>: elementwise bit shift with NumPy broadcasting. The required
/// <c>direction</c> attribute is "LEFT" or "RIGHT". Supports Int64 / Int32 / UInt8.
/// </summary>
public sealed class BitShiftKernel : IKernel
{
    public string OpType => "BitShift";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor a = ctx.GetTensor(node.Inputs[0]);
        Tensor b = ctx.GetTensor(node.Inputs[1]);
        string dir = Attr.Str(node, "direction", "LEFT");
        bool left = dir == "LEFT";
        if (!left && dir != "RIGHT")
            throw new ModelSharpException($"BitShift: 'direction' must be LEFT or RIGHT, got '{dir}'.");

        switch (a.Dtype)
        {
            case ElementType.Int64:
            {
                Func<long, long, long> op = left
                    ? (x, s) => x << (int)s
                    : (x, s) => (long)((ulong)x >> (int)s);
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(a.AsInt64(), b.AsInt64(), op));
                break;
            }
            case ElementType.Int32:
            {
                Func<int, int, int> op = left
                    ? (x, s) => x << s
                    : (x, s) => (int)((uint)x >> s);
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(a.AsInt32(), b.AsInt32(), op));
                break;
            }
            case ElementType.UInt8:
            {
                Tensor<byte> ba = (a as Tensor<byte>)!, bb = (b as Tensor<byte>)!;
                Func<byte, byte, byte> op = left
                    ? (x, s) => (byte)(x << s)
                    : (x, s) => (byte)(x >> s);
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(ba, bb, op));
                break;
            }
            default:
                throw new ModelSharpException($"BitShift: unsupported dtype {a.Dtype}.");
        }
    }
}
