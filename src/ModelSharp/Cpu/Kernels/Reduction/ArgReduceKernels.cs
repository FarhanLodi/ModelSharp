using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Reduction;

/// <summary>
/// Shared machinery for <c>ArgMax</c>/<c>ArgMin</c>: reduces a single <c>axis</c> (default 0,
/// negative-normalized) to the index of the extreme element, emitting an Int64 tensor.
/// Honors <c>keepdims</c> (default 1) and <c>select_last_index</c> (default 0).
/// </summary>
public abstract class ArgReduceKernelBase : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>True when <paramref name="candidate"/> is a strictly better extreme than <paramref name="best"/>.</summary>
    protected abstract bool IsBetter(float candidate, float best);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;

        long axisAttr = Attr.Int(node, "axis", 0);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        bool keepdims = Attr.Int(node, "keepdims", 1) != 0;
        bool selectLast = Attr.Int(node, "select_last_index", 0) != 0;

        int axisDim = dims[axis];
        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        var outBuf = new long[outer * inner];
        Span<float> xs = x.Span;
        for (int o = 0; o < outer; o++)
        for (int q = 0; q < inner; q++)
        {
            int baseIdx = o * axisDim * inner + q;
            float best = xs[baseIdx];
            long bestIdx = 0;
            for (int s = 1; s < axisDim; s++)
            {
                float v = xs[baseIdx + s * inner];
                if (IsBetter(v, best) || (selectLast && v == best))
                {
                    best = v;
                    bestIdx = s;
                }
            }
            outBuf[o * inner + q] = bestIdx;
        }

        int[] outDims;
        if (keepdims)
        {
            outDims = dims.ToArray();
            outDims[axis] = 1;
        }
        else
        {
            var list = new List<int>();
            for (int i = 0; i < rank; i++) if (i != axis) list.Add(dims[i]);
            outDims = list.ToArray();
        }
        ctx.Set(node.Outputs[0], new Tensor<long>(new TensorShape(outDims), outBuf));
    }
}

/// <summary>Index of the maximum element along an axis (ONNX <c>ArgMax</c>).</summary>
public sealed class ArgMaxKernel : ArgReduceKernelBase
{
    public override string OpType => "ArgMax";
    protected override bool IsBetter(float candidate, float best) => candidate > best;
}

/// <summary>Index of the minimum element along an axis (ONNX <c>ArgMin</c>).</summary>
public sealed class ArgMinKernel : ArgReduceKernelBase
{
    public override string OpType => "ArgMin";
    protected override bool IsBetter(float candidate, float best) => candidate < best;
}
