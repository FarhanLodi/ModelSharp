using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>Upsample</c> (deprecated, but still emitted by older models): scales an N-D input by the
/// per-axis factors in the <c>scales</c> input (1-D float, length == rank). <c>mode</c> is "nearest"
/// (default) or "linear"; linear is implemented as asymmetric N-linear interpolation matching the
/// legacy Upsample/Resize "asymmetric" coordinate transform. Output axis <c>i</c> has size
/// <c>floor(dims[i] * scales[i])</c>. Float32.
/// </summary>
public sealed class UpsampleKernel : IKernel
{
    public string OpType => "Upsample";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float[] scales = ReadScales(ctx.GetTensor(node.Inputs[1]));
        string mode = Attr.Str(node, "mode", "nearest");

        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        if (scales.Length != rank) throw new ModelSharpException("Upsample: scales length must equal input rank.");

        var outDims = new int[rank];
        for (int i = 0; i < rank; i++) outDims[i] = (int)(dims[i] * scales[i]);
        var y = new Tensor<float>(new TensorShape(outDims));

        int[] inStrides = Nd.Strides(dims);
        int[] outStrides = Nd.Strides(outDims);
        Span<float> xs = x.Span, ys = y.Span;
        int n = (int)y.Shape.Length;
        var coord = new int[rank];
        bool linear = mode == "linear" || mode == "bilinear";

        for (int idx = 0; idx < n; idx++)
        {
            ys[idx] = linear ? LinearSample(xs, dims, inStrides, coord, scales)
                             : NearestSample(xs, dims, inStrides, coord, scales);
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }
        ctx.Set(node.Outputs[0], y);
    }

    private static float NearestSample(Span<float> xs, ReadOnlySpan<int> dims, int[] inStrides, int[] coord, float[] scales)
    {
        int src = 0;
        for (int i = 0; i < dims.Length; i++)
        {
            int s = (int)(coord[i] / scales[i]);
            if (s >= dims[i]) s = dims[i] - 1;
            src += s * inStrides[i];
        }
        return xs[src];
    }

    private static float LinearSample(Span<float> xs, ReadOnlySpan<int> dims, int[] inStrides, int[] coord, float[] scales)
    {
        int rank = dims.Length;
        // Asymmetric: in = out / scale.
        var lo = new int[rank];
        var w1 = new float[rank];
        for (int i = 0; i < rank; i++)
        {
            float pos = coord[i] / scales[i];
            if (pos < 0) pos = 0;
            int l = (int)MathF.Floor(pos);
            if (l > dims[i] - 1) l = dims[i] - 1;
            lo[i] = l;
            w1[i] = (dims[i] == 1) ? 0f : MathF.Min(1f, pos - l);
        }

        // Interpolate over the 2^rank corners.
        int corners = 1 << rank;
        float acc = 0f;
        for (int mask = 0; mask < corners; mask++)
        {
            float w = 1f;
            int src = 0;
            for (int i = 0; i < rank; i++)
            {
                bool high = (mask & (1 << i)) != 0;
                int c = lo[i] + (high ? 1 : 0);
                if (c > dims[i] - 1) c = dims[i] - 1;
                src += c * inStrides[i];
                w *= high ? w1[i] : (1f - w1[i]);
            }
            if (w != 0f) acc += w * xs[src];
        }
        return acc;
    }

    private static float[] ReadScales(Tensor t)
    {
        if (t.Dtype == ElementType.Float32) return t.AsFloat().Span.ToArray();
        long[] v = TensorInts.Read(t);
        return Array.ConvertAll(v, x => (float)x);
    }
}
