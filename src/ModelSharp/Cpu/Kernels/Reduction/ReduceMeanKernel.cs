using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Reduction;

/// <summary>
/// Mean over the given axes (ONNX <c>ReduceMean</c>). Axes come from the attribute (opset &lt; 18)
/// or the optional second input (opset 18+); empty axes reduces all. <c>keepdims</c> default 1.
/// </summary>
public sealed class ReduceMeanKernel : IKernel
{
    public string OpType => "ReduceMean";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        System.ReadOnlySpan<int> inDims = x.Shape.Dimensions;
        int rank = inDims.Length;

        int[]? axes = Attr.Ints(node, "axes");
        if (axes is null && node.Inputs.Count > 1 && node.Inputs[1].Length > 0)
        {
            System.Span<float> a = ctx.Get(node.Inputs[1]).Span;
            axes = new int[a.Length];
            for (int i = 0; i < a.Length; i++) axes[i] = (int)a[i];
        }
        bool keepdims = Attr.Int(node, "keepdims", 1) != 0;

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

        var sum = new float[outLen];
        System.Span<float> xs = x.Span;
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
            sum[outIdx] += xs[idx];
        }
        for (int i = 0; i < outLen; i++) sum[i] /= count;

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
        ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(finalDims), sum));
    }
}
