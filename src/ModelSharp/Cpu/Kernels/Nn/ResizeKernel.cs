using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>Resize</c> (Float32): nearest and linear interpolation over arbitrary rank.
/// Output dims come from the <c>sizes</c> input or are derived from the <c>scales</c> input
/// (the optional <c>roi</c> input is ignored — tf_crop_and_resize is not supported).
/// Honors <c>coordinate_transformation_mode</c> (half_pixel, pytorch_half_pixel, align_corners,
/// asymmetric) and, for nearest, <c>nearest_mode</c>
/// (round_prefer_floor, round_prefer_ceil, floor, ceil). Linear is implemented as N-linear
/// interpolation, so axes with scale 1 (e.g. N, C) pass through unchanged.
/// </summary>
public sealed class ResizeKernel : IKernel
{
    public string OpType => "Resize";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> inDims = x.Shape.Dimensions;
        int rank = inDims.Length;

        string mode = Attr.Str(node, "mode", "nearest");
        string ct = Attr.Str(node, "coordinate_transformation_mode", "half_pixel");
        string nearestMode = Attr.Str(node, "nearest_mode", "round_prefer_floor");

        // Resolve output dims and the per-axis scale used by the coordinate transform.
        var outDims = new int[rank];
        var scale = new double[rank];
        float[]? scalesIn = ReadOptionalFloats(node, ctx, 2);
        long[]? sizesIn = ReadOptionalInts(node, ctx, 3);

        if (sizesIn is not null && sizesIn.Length == rank)
        {
            for (int i = 0; i < rank; i++)
            {
                outDims[i] = (int)sizesIn[i];
                scale[i] = (double)outDims[i] / inDims[i];
            }
        }
        else if (scalesIn is not null && scalesIn.Length == rank)
        {
            for (int i = 0; i < rank; i++)
            {
                scale[i] = scalesIn[i];
                outDims[i] = (int)System.Math.Floor(電(inDims[i], scalesIn[i]));
            }
        }
        else
        {
            throw new ModelSharpException("Resize requires a valid 'scales' or 'sizes' input of matching rank.");
        }

        var y = new Tensor<float>(new TensorShape(outDims));
        Span<float> xs = x.Span, ys = y.Span;
        int[] inStrides = Nd.Strides(inDims);
        int n = (int)y.Shape.Length;
        var coord = new int[rank];

        bool linear = mode == "linear" || mode == "bilinear";

        for (int outIdx = 0; outIdx < n; outIdx++)
        {
            if (linear) ys[outIdx] = SampleLinear(xs, inDims, inStrides, coord, scale, ct);
            else ys[outIdx] = SampleNearest(xs, inDims, inStrides, coord, scale, ct, nearestMode);
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }

        ctx.Set(node.Outputs[0], y);
    }

    private static double 電(int len, double s) => len * s; // length_original * scale

    private static float SampleNearest(
        Span<float> xs, ReadOnlySpan<int> inDims, int[] inStrides, int[] coord,
        double[] scale, string ct, string nearestMode)
    {
        int rank = inDims.Length;
        int src = 0;
        for (int i = 0; i < rank; i++)
        {
            double c = SourceCoord(coord[i], scale[i], inDims[i], OutLen(inDims[i], scale[i]), ct);
            int idx = RoundNearest(c, nearestMode);
            if (idx < 0) idx = 0; else if (idx >= inDims[i]) idx = inDims[i] - 1;
            src += idx * inStrides[i];
        }
        return xs[src];
    }

    private static float SampleLinear(
        Span<float> xs, ReadOnlySpan<int> inDims, int[] inStrides, int[] coord,
        double[] scale, string ct)
    {
        int rank = inDims.Length;
        var lo = new int[rank];
        var frac = new double[rank];
        for (int i = 0; i < rank; i++)
        {
            double c = SourceCoord(coord[i], scale[i], inDims[i], OutLen(inDims[i], scale[i]), ct);
            if (c < 0) c = 0; else if (c > inDims[i] - 1) c = inDims[i] - 1;
            int i0 = (int)System.Math.Floor(c);
            if (i0 > inDims[i] - 1) i0 = inDims[i] - 1;
            lo[i] = i0;
            frac[i] = c - i0;
        }

        // Blend the 2^rank corners; axes with frac 0 contribute only their lower corner.
        double acc = 0d;
        int corners = 1 << rank;
        for (int mask = 0; mask < corners; mask++)
        {
            double w = 1d;
            int off = 0;
            bool skip = false;
            for (int i = 0; i < rank; i++)
            {
                bool high = (mask & (1 << i)) != 0;
                if (high)
                {
                    if (frac[i] == 0d) { skip = true; break; }
                    int hi = lo[i] + 1;
                    if (hi > inDims[i] - 1) hi = inDims[i] - 1;
                    w *= frac[i];
                    off += hi * inStrides[i];
                }
                else
                {
                    w *= 1d - frac[i];
                    off += lo[i] * inStrides[i];
                }
            }
            if (skip || w == 0d) continue;
            acc += w * xs[off];
        }
        return (float)acc;
    }

    private static int OutLen(int inLen, double scale) => (int)System.Math.Floor(inLen * scale);

    /// <summary>Maps an output index to the (fractional) source coordinate per the transform mode.</summary>
    private static double SourceCoord(int outIdx, double scale, int inLen, int outLen, string ct) => ct switch
    {
        "asymmetric" => outIdx / scale,
        "align_corners" => outLen <= 1 ? 0d : outIdx * (double)(inLen - 1) / (outLen - 1),
        "pytorch_half_pixel" => outLen <= 1 ? 0d : (outIdx + 0.5) / scale - 0.5,
        _ => (outIdx + 0.5) / scale - 0.5, // half_pixel (default)
    };

    private static int RoundNearest(double c, string nearestMode) => nearestMode switch
    {
        "floor" => (int)System.Math.Floor(c),
        "ceil" => (int)System.Math.Ceiling(c),
        "round_prefer_ceil" => (int)System.Math.Floor(c + 0.5),
        _ => (int)System.Math.Ceiling(c - 0.5), // round_prefer_floor (default)
    };

    private static float[]? ReadOptionalFloats(GraphNode node, GraphContext ctx, int inputIndex)
    {
        if (node.Inputs.Count > inputIndex && !string.IsNullOrEmpty(node.Inputs[inputIndex]))
        {
            Tensor t = ctx.GetTensor(node.Inputs[inputIndex]);
            if (t.Length == 0) return null;
            Span<float> s = t.AsFloat().Span;
            return s.ToArray();
        }
        return null;
    }

    private static long[]? ReadOptionalInts(GraphNode node, GraphContext ctx, int inputIndex)
    {
        if (node.Inputs.Count > inputIndex && !string.IsNullOrEmpty(node.Inputs[inputIndex]))
        {
            Tensor t = ctx.GetTensor(node.Inputs[inputIndex]);
            if (t.Length == 0) return null;
            return TensorInts.Read(t);
        }
        return null;
    }
}
