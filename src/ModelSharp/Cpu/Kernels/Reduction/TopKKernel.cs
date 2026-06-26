using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Reduction;

/// <summary>
/// ONNX <c>TopK</c>: for each slice along <c>axis</c> (default -1) returns the <c>k</c> largest
/// (or smallest, when <c>largest=0</c>) values and their Int64 indices. <c>k</c> comes from the
/// second input (opset 10+) or the <c>k</c> attribute (opset 1). Output is value-sorted; ties
/// break toward the lower index. Float32 data only.
/// </summary>
public sealed class TopKKernel : IKernel
{
    public string OpType => "TopK";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;

        long axisAttr = Attr.Int(node, "axis", -1);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        bool largest = Attr.Int(node, "largest", 1) != 0;

        int k;
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
            k = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[1]))[0];
        else
            k = (int)Attr.Int(node, "k", 0);

        int axisDim = dims[axis];
        if (k < 0 || k > axisDim)
            throw new ModelSharpException($"TopK 'k' ({k}) is out of range for axis size {axisDim}.");

        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        int[] outDims = dims.ToArray();
        outDims[axis] = k;
        var values = new Tensor<float>(new TensorShape(outDims));
        var indices = new Tensor<long>(new TensorShape(outDims));
        Span<float> xs = x.Span, vs = values.Span;
        Span<long> ids = indices.Span;

        var sliceVals = new float[axisDim];
        var sliceIdx = new int[axisDim];
        Comparison<int> cmp = largest
            ? (a, b) => { int c = sliceVals[b].CompareTo(sliceVals[a]); return c != 0 ? c : a.CompareTo(b); }
            : (a, b) => { int c = sliceVals[a].CompareTo(sliceVals[b]); return c != 0 ? c : a.CompareTo(b); };

        for (int o = 0; o < outer; o++)
        for (int q = 0; q < inner; q++)
        {
            int baseIdx = o * axisDim * inner + q;
            for (int s = 0; s < axisDim; s++) { sliceVals[s] = xs[baseIdx + s * inner]; sliceIdx[s] = s; }
            Array.Sort(sliceIdx, cmp);

            int outBase = o * k * inner + q;
            for (int t = 0; t < k; t++)
            {
                int s = sliceIdx[t];
                vs[outBase + t * inner] = sliceVals[s];
                ids[outBase + t * inner] = s;
            }
        }

        ctx.Set(node.Outputs[0], values);
        if (node.Outputs.Count > 1) ctx.Set(node.Outputs[1], indices);
    }
}
