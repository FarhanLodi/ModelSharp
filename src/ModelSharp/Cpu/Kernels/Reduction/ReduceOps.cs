using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Reduction;

/// <summary>
/// Shared machinery for axis reductions (ReduceSum / ReduceMax / ReduceMin / ReduceProd / ReduceL2).
/// Axes come from the <c>axes</c> attribute (opset &lt; 18) or the optional second input (opset 18+);
/// an empty/absent axes list reduces all axes unless <c>noop_with_empty_axes</c> is set (then it is a
/// no-op identity copy). Negative axes are normalized. <c>keepdims</c> defaults to 1.
/// </summary>
public abstract class ReduceKernelBase : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>The accumulator's initial value before any element is folded in.</summary>
    protected abstract float Init { get; }

    /// <summary>Folds one element into the running accumulator.</summary>
    protected abstract float Combine(float acc, float x);

    /// <summary>Post-processes the accumulator given the number of reduced elements (default: identity).</summary>
    protected virtual float Postprocess(float acc, int count) => acc;

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        // Dtype-aware: floats fold directly; integer inputs (e.g. ReduceSum over an int64 attention_mask)
        // are folded in float and emitted back in their original integer dtype.
        Tensor xt = ctx.GetTensor(node.Inputs[0]);
        ElementType inType = xt.Dtype;
        Tensor<float> x = xt as Tensor<float> ?? new Tensor<float>(xt.Shape, ReadAsFloat(xt));
        ReadOnlySpan<int> inDims = x.Shape.Dimensions;
        int rank = inDims.Length;

        int[]? axes = Attr.Ints(node, "axes");
        if (axes is null && node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
        {
            long[] a = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
            axes = new int[a.Length];
            for (int i = 0; i < a.Length; i++) axes[i] = (int)a[i];
        }
        bool keepdims = Attr.Int(node, "keepdims", 1) != 0;
        bool noopEmpty = Attr.Int(node, "noop_with_empty_axes", 0) != 0;

        if ((axes is null || axes.Length == 0) && noopEmpty)
        {
            // Identity: copy through unchanged, preserving the input dtype.
            ctx.Set(node.Outputs[0], xt);
            return;
        }

        var reduced = new bool[rank];
        if (axes is null || axes.Length == 0)
            for (int i = 0; i < rank; i++) reduced[i] = true;
        else
            foreach (int ax in axes) reduced[ax < 0 ? ax + rank : ax] = true;

        int[] inStrides = Nd.Strides(inDims);
        var keepDims = new int[rank];
        for (int i = 0; i < rank; i++) keepDims[i] = reduced[i] ? 1 : inDims[i];
        int[] keepStrides = Nd.Strides(keepDims);

        int outLen = 1;
        foreach (int d in keepDims) outLen *= d;
        int count = 1;
        for (int i = 0; i < rank; i++) if (reduced[i]) count *= inDims[i];

        var acc = new float[outLen];
        for (int i = 0; i < outLen; i++) acc[i] = Init;

        Span<float> xs = x.Span;
        int n = (int)x.Shape.Length;
        for (int idx = 0; idx < n; idx++)
        {
            int outIdx = 0;
            for (int ax = 0; ax < rank; ax++)
            {
                if (reduced[ax]) continue;
                int coord = (idx / inStrides[ax]) % inDims[ax];
                outIdx += coord * keepStrides[ax];
            }
            acc[outIdx] = Combine(acc[outIdx], xs[idx]);
        }
        for (int i = 0; i < outLen; i++) acc[i] = Postprocess(acc[i], count);

        int[] finalDims;
        if (keepdims)
        {
            finalDims = keepDims;
        }
        else
        {
            var list = new List<int>();
            for (int i = 0; i < rank; i++) if (!reduced[i]) list.Add(inDims[i]);
            finalDims = list.ToArray();
        }
        var outShape = new TensorShape(finalDims);
        ctx.Set(node.Outputs[0], inType switch
        {
            ElementType.Int64 => new Tensor<long>(outShape, Array.ConvertAll(acc, v => (long)MathF.Round(v))),
            ElementType.Int32 => new Tensor<int>(outShape, Array.ConvertAll(acc, v => (int)MathF.Round(v))),
            _ => new Tensor<float>(outShape, acc),
        });
    }

    /// <summary>Reads any numeric tensor (float/int64/int32) as a float array for folding.</summary>
    private static float[] ReadAsFloat(Tensor t) => t.Dtype switch
    {
        ElementType.Float32 => t.AsFloat().Buffer.ToArray(),
        ElementType.Int64 => Array.ConvertAll(t.AsInt64().Buffer.ToArray(), v => (float)v),
        ElementType.Int32 => Array.ConvertAll(t.AsInt32().Buffer.ToArray(), v => (float)v),
        _ => throw new ModelSharpException($"Reduce does not support input dtype {t.Dtype}."),
    };
}

/// <summary>Sum over the given axes (ONNX <c>ReduceSum</c>).</summary>
public sealed class ReduceSumKernel : ReduceKernelBase
{
    public override string OpType => "ReduceSum";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x;
}

/// <summary>Maximum over the given axes (ONNX <c>ReduceMax</c>).</summary>
public sealed class ReduceMaxKernel : ReduceKernelBase
{
    public override string OpType => "ReduceMax";
    protected override float Init => float.NegativeInfinity;
    protected override float Combine(float acc, float x) => x > acc ? x : acc;
}

/// <summary>Minimum over the given axes (ONNX <c>ReduceMin</c>).</summary>
public sealed class ReduceMinKernel : ReduceKernelBase
{
    public override string OpType => "ReduceMin";
    protected override float Init => float.PositiveInfinity;
    protected override float Combine(float acc, float x) => x < acc ? x : acc;
}

/// <summary>Product over the given axes (ONNX <c>ReduceProd</c>).</summary>
public sealed class ReduceProdKernel : ReduceKernelBase
{
    public override string OpType => "ReduceProd";
    protected override float Init => 1f;
    protected override float Combine(float acc, float x) => acc * x;
}

/// <summary>L2 norm (sqrt of sum of squares) over the given axes (ONNX <c>ReduceL2</c>).</summary>
public sealed class ReduceL2Kernel : ReduceKernelBase
{
    public override string OpType => "ReduceL2";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x * x;
    protected override float Postprocess(float acc, int count) => MathF.Sqrt(acc);
}

/// <summary>Sum of absolute values over the given axes (ONNX <c>ReduceL1</c>).</summary>
public sealed class ReduceL1Kernel : ReduceKernelBase
{
    public override string OpType => "ReduceL1";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + MathF.Abs(x);
}

/// <summary>Sum of squares over the given axes (ONNX <c>ReduceSumSquare</c>).</summary>
public sealed class ReduceSumSquareKernel : ReduceKernelBase
{
    public override string OpType => "ReduceSumSquare";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x * x;
}

/// <summary>Log of the sum over the given axes (ONNX <c>ReduceLogSum</c>): <c>log(Σ x)</c>.</summary>
public sealed class ReduceLogSumKernel : ReduceKernelBase
{
    public override string OpType => "ReduceLogSum";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + x;
    protected override float Postprocess(float acc, int count) => MathF.Log(acc);
}

/// <summary>Log of the sum of exponentials over the given axes (ONNX <c>ReduceLogSumExp</c>): <c>log(Σ exp(x))</c>.</summary>
public sealed class ReduceLogSumExpKernel : ReduceKernelBase
{
    public override string OpType => "ReduceLogSumExp";
    protected override float Init => 0f;
    protected override float Combine(float acc, float x) => acc + MathF.Exp(x);
    protected override float Postprocess(float acc, int count) => MathF.Log(acc);
}
