using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Logical;

/// <summary>
/// Shared machinery for binary boolean-input/boolean-output ops (<c>And</c>, <c>Or</c>,
/// <c>Xor</c>) with NumPy-style broadcasting. Both inputs must be Boolean tensors.
/// </summary>
public abstract class BinaryBoolKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The scalar boolean operation applied elementwise.</summary>
    protected abstract bool Apply(bool a, bool b);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<bool> a = ctx.GetTensor(node.Inputs[0]).AsBool();
        Tensor<bool> b = ctx.GetTensor(node.Inputs[1]).AsBool();

        if (a.Shape.Equals(b.Shape))
        {
            var yEqual = new Tensor<bool>(a.Shape);
            Span<bool> sa = a.Span, sb = b.Span, sy = yEqual.Span;
            for (int i = 0; i < sy.Length; i++) sy[i] = Apply(sa[i], sb[i]);
            ctx.Set(node.Outputs[0], yEqual);
            return;
        }

        int[] outd = Nd.BroadcastShape(a.Shape.Dimensions, b.Shape.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = Nd.BroadcastStrides(a.Shape.Dimensions, rank);
        int[] strideB = Nd.BroadcastStrides(b.Shape.Dimensions, rank);

        var y = new Tensor<bool>(outShape);
        Span<bool> da = a.Span, db = b.Span, dy = y.Span;
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

/// <summary>ONNX <c>And</c>: elementwise logical AND of two Boolean tensors (broadcasting).</summary>
public sealed class AndKernel : BinaryBoolKernel
{
    public override string OpType => "And";
    protected override bool Apply(bool a, bool b) => a && b;
}

/// <summary>ONNX <c>Or</c>: elementwise logical OR of two Boolean tensors (broadcasting).</summary>
public sealed class OrKernel : BinaryBoolKernel
{
    public override string OpType => "Or";
    protected override bool Apply(bool a, bool b) => a || b;
}

/// <summary>ONNX <c>Xor</c>: elementwise logical XOR of two Boolean tensors (broadcasting).</summary>
public sealed class XorKernel : BinaryBoolKernel
{
    public override string OpType => "Xor";
    protected override bool Apply(bool a, bool b) => a ^ b;
}

/// <summary>ONNX <c>Not</c>: elementwise logical negation of a Boolean tensor.</summary>
public sealed class NotKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "Not";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<bool> x = ctx.GetTensor(node.Inputs[0]).AsBool();
        var y = new Tensor<bool>(x.Shape);
        Span<bool> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = !xs[i];
        ctx.Set(node.Outputs[0], y);
    }
}
