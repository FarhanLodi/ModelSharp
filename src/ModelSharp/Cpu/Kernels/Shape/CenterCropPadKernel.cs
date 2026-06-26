using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>CenterCropPad</c>: center-crop or center-pad (with zeros) the input so the selected axes
/// match the target sizes given by the 1-D <c>shape</c> input. The optional <c>axes</c> attribute
/// lists which axes the target sizes apply to (default: all axes, in order). Per axis, if the target
/// is smaller the input is cropped symmetrically about the center; if larger it is zero-padded
/// symmetrically. Dtype-preserving (Float32 / Int64 / Int32).
/// </summary>
public sealed class CenterCropPadKernel : IKernel
{
    public string OpType => "CenterCropPad";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        long[] targets = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        int rank = data.Shape.Rank;

        int[] axes;
        if (node.Attributes.ContainsKey("axes"))
        {
            axes = Attr.Ints(node, "axes")!;
            for (int i = 0; i < axes.Length; i++) if (axes[i] < 0) axes[i] += rank;
        }
        else { axes = new int[rank]; for (int i = 0; i < rank; i++) axes[i] = i; }
        if (axes.Length != targets.Length)
            throw new ModelSharpException("CenterCropPad: 'shape' length must match the number of axes.");

        var outDims = data.Shape.Dimensions.ToArray();
        for (int i = 0; i < axes.Length; i++) outDims[axes[i]] = (int)targets[i];

        ctx.Set(node.Outputs[0], data.Dtype switch
        {
            ElementType.Int64 => Run<long>(data.AsInt64(), outDims),
            ElementType.Int32 => Run<int>(data.AsInt32(), outDims),
            _ => Run<float>(data.AsFloat(), outDims),
        });
    }

    private static Tensor<T> Run<T>(Tensor<T> x, int[] outDims) where T : unmanaged
    {
        ReadOnlySpan<int> inDims = x.Shape.Dimensions;
        int rank = inDims.Length;
        int[] inStrides = Nd.Strides(inDims);
        var y = new Tensor<T>(new TensorShape(outDims));

        // For each axis, the offset mapping output coord -> input coord (centered).
        // outStart/inStart pick the symmetric window.
        var inStart = new int[rank];
        var outStart = new int[rank];
        for (int a = 0; a < rank; a++)
        {
            int diff = outDims[a] - inDims[a];
            if (diff < 0) { inStart[a] = (-diff) / 2; outStart[a] = 0; }   // crop
            else { inStart[a] = 0; outStart[a] = diff / 2; }              // pad
        }

        Span<T> xs = x.Span, ys = y.Span;
        int n = (int)x.Shape.Length;
        var coord = new int[rank];
        int[] outStrides = Nd.Strides(outDims);
        for (int idx = 0; idx < n; idx++)
        {
            // Map input coord -> output coord; copy only if it lands in range.
            bool inside = true;
            int dst = 0;
            for (int a = 0; a < rank; a++)
            {
                int oc = coord[a] - inStart[a] + outStart[a];
                if (oc < 0 || oc >= outDims[a]) { inside = false; break; }
                dst += oc * outStrides[a];
            }
            if (inside) ys[dst] = xs[idx];
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < inDims[ax]) break; coord[ax] = 0; }
        }
        return y;
    }
}
