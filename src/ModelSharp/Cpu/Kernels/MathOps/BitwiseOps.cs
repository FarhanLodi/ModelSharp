using System;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.MathOps;

/// <summary>
/// Shared machinery for the binary bitwise ops <c>BitwiseAnd</c>, <c>BitwiseOr</c>,
/// <c>BitwiseXor</c> (NumPy-style broadcasting, integer dtypes Int64 / Int32 / Int8 / UInt8).
/// Unlike the logical <c>And</c>/<c>Or</c>/<c>Xor</c> these operate bit-by-bit on integers.
/// </summary>
public abstract class BinaryBitwiseKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The scalar bitwise op, expressed over int64 (narrower dtypes reuse it then truncate).</summary>
    protected abstract long Apply(long a, long b);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor a = ctx.GetTensor(node.Inputs[0]);
        Tensor b = ctx.GetTensor(node.Inputs[1]);
        switch (a.Dtype)
        {
            case ElementType.Int64:
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(a.AsInt64(), b.AsInt64(), (x, y) => Apply(x, y)));
                break;
            case ElementType.Int32:
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(a.AsInt32(), b.AsInt32(), (x, y) => (int)Apply(x, y)));
                break;
            case ElementType.Int8:
            {
                Tensor<sbyte> sa = (a as Tensor<sbyte>)!, sb = (b as Tensor<sbyte>)!;
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(sa, sb, (x, y) => (sbyte)Apply(x, y)));
                break;
            }
            case ElementType.UInt8:
            {
                Tensor<byte> ba = (a as Tensor<byte>)!, bb = (b as Tensor<byte>)!;
                ctx.Set(node.Outputs[0], TypedBroadcast.Apply(ba, bb, (x, y) => (byte)Apply(x, y)));
                break;
            }
            default:
                throw new ModelSharpException($"{OpType}: unsupported dtype {a.Dtype}.");
        }
    }
}

/// <summary>ONNX <c>BitwiseAnd</c>: elementwise bitwise AND of two integer tensors (broadcasting).</summary>
public sealed class BitwiseAndKernel : BinaryBitwiseKernel
{
    public override string OpType => "BitwiseAnd";
    protected override long Apply(long a, long b) => a & b;
}

/// <summary>ONNX <c>BitwiseOr</c>: elementwise bitwise OR of two integer tensors (broadcasting).</summary>
public sealed class BitwiseOrKernel : BinaryBitwiseKernel
{
    public override string OpType => "BitwiseOr";
    protected override long Apply(long a, long b) => a | b;
}

/// <summary>ONNX <c>BitwiseXor</c>: elementwise bitwise XOR of two integer tensors (broadcasting).</summary>
public sealed class BitwiseXorKernel : BinaryBitwiseKernel
{
    public override string OpType => "BitwiseXor";
    protected override long Apply(long a, long b) => a ^ b;
}

/// <summary>
/// ONNX <c>BitwiseNot</c>: elementwise bitwise complement of an integer tensor.
/// Supports Int64 / Int32 / Int8 / UInt8; shape and dtype are preserved.
/// </summary>
public sealed class BitwiseNotKernel : IKernel
{
    public string OpType => "BitwiseNot";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor x = ctx.GetTensor(node.Inputs[0]);
        switch (x.Dtype)
        {
            case ElementType.Int64:
            {
                var t = x.AsInt64(); var y = new Tensor<long>(t.Shape);
                Span<long> s = t.Span, o = y.Span;
                for (int i = 0; i < s.Length; i++) o[i] = ~s[i];
                ctx.Set(node.Outputs[0], y);
                break;
            }
            case ElementType.Int32:
            {
                var t = x.AsInt32(); var y = new Tensor<int>(t.Shape);
                Span<int> s = t.Span, o = y.Span;
                for (int i = 0; i < s.Length; i++) o[i] = ~s[i];
                ctx.Set(node.Outputs[0], y);
                break;
            }
            case ElementType.Int8:
            {
                var t = (x as Tensor<sbyte>)!; var y = new Tensor<sbyte>(t.Shape);
                Span<sbyte> s = t.Span, o = y.Span;
                for (int i = 0; i < s.Length; i++) o[i] = (sbyte)~s[i];
                ctx.Set(node.Outputs[0], y);
                break;
            }
            case ElementType.UInt8:
            {
                var t = (x as Tensor<byte>)!; var y = new Tensor<byte>(t.Shape);
                Span<byte> s = t.Span, o = y.Span;
                for (int i = 0; i < s.Length; i++) o[i] = (byte)~s[i];
                ctx.Set(node.Outputs[0], y);
                break;
            }
            default:
                throw new ModelSharpException($"BitwiseNot: unsupported dtype {x.Dtype}.");
        }
    }
}
