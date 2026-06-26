using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// Instance normalization: y = (x − mean)/√(var + ε)·scale[c] + B[c], with the
/// mean and variance computed per (batch, channel) over the spatial dimensions.
/// Inputs: X [N, C, …spatial], scale [C], B [C]. Attribute: <c>epsilon</c> (1e-5).
/// </summary>
public sealed class InstanceNormalizationKernel : IKernel
{
    public string OpType => "InstanceNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> scale = ctx.Get(node.Inputs[1]);
        Tensor<float> bvec = ctx.Get(node.Inputs[2]);
        float eps = Attr.Float(node, "epsilon", 1e-5f);

        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int N = dims[0], C = dims[1];
        int spatial = 1;
        for (int i = 2; i < dims.Length; i++) spatial *= dims[i];

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span, sc = scale.Span, bb = bvec.Span;

        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int baseI = (n * C + c) * spatial;

            float mean = 0f;
            for (int s = 0; s < spatial; s++) mean += xs[baseI + s];
            mean /= spatial;

            float varSum = 0f;
            for (int s = 0; s < spatial; s++) { float d = xs[baseI + s] - mean; varSum += d * d; }
            float inv = 1f / MathF.Sqrt(varSum / spatial + eps);

            float scc = sc[c], bc = bb[c];
            for (int s = 0; s < spatial; s++)
                ys[baseI + s] = (xs[baseI + s] - mean) * inv * scc + bc;
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Group normalization (ONNX opset 18): split the C channels into <c>num_groups</c>
/// groups and normalize per (batch, group) over (channels-in-group × spatial), then
/// apply per-channel affine: y = (x − mean)/√(var + ε)·scale[c] + bias[c].
/// Inputs: X [N, C, …spatial], scale, bias. Attributes: <c>num_groups</c>, <c>epsilon</c>.
/// Convention: scale and bias are length C (the opset-18 layout), indexed per channel.
/// </summary>
public sealed class GroupNormalizationKernel : IKernel
{
    public string OpType => "GroupNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> scale = ctx.Get(node.Inputs[1]);
        Tensor<float> bias = ctx.Get(node.Inputs[2]);
        int numGroups = (int)Attr.Int(node, "num_groups", 1);
        float eps = Attr.Float(node, "epsilon", 1e-5f);

        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int N = dims[0], C = dims[1];
        int spatial = 1;
        for (int i = 2; i < dims.Length; i++) spatial *= dims[i];

        int channelsPerGroup = C / numGroups;
        int groupSize = channelsPerGroup * spatial;

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span, sc = scale.Span, bb = bias.Span;

        for (int n = 0; n < N; n++)
        for (int g = 0; g < numGroups; g++)
        {
            int firstChannel = g * channelsPerGroup;
            int baseI = (n * C + firstChannel) * spatial;

            float mean = 0f;
            for (int i = 0; i < groupSize; i++) mean += xs[baseI + i];
            mean /= groupSize;

            float varSum = 0f;
            for (int i = 0; i < groupSize; i++) { float d = xs[baseI + i] - mean; varSum += d * d; }
            float inv = 1f / MathF.Sqrt(varSum / groupSize + eps);

            for (int cg = 0; cg < channelsPerGroup; cg++)
            {
                int c = firstChannel + cg;
                float scc = sc[c], bc = bb[c];
                int chBase = (n * C + c) * spatial;
                for (int s = 0; s < spatial; s++)
                    ys[chBase + s] = (xs[chBase + s] - mean) * inv * scc + bc;
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Mean-variance normalization: y = (x − mean)/√(var + ε), with mean and variance
/// taken over the <c>axes</c> (default [0, 2, 3]). No scale or bias. ORT uses a tiny
/// epsilon of 1e-9. Reductions broadcast back over the kept axes.
/// </summary>
public sealed class MeanVarianceNormalizationKernel : IKernel
{
    public string OpType => "MeanVarianceNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;

        int[] axes = Attr.Ints(node, "axes", new[] { 0, 2, 3 });
        // Normalize negative axes and build a reduced/kept mask.
        var reduce = new bool[rank];
        foreach (int rawAxis in axes)
        {
            int a = rawAxis < 0 ? rawAxis + rank : rawAxis;
            reduce[a] = true;
        }
        const float eps = 1e-9f;

        // Strides for the full tensor (row-major).
        var strides = new int[rank];
        int acc = 1;
        for (int i = rank - 1; i >= 0; i--) { strides[i] = acc; acc *= dims[i]; }

        // Shape of the "group" tensor (kept axes only). Each group accumulates the
        // mean/var over the reduced axes. groupCount = product of kept dims;
        // reducedCount = product of reduced dims.
        long total = x.Shape.Length;
        int reducedCount = 1;
        for (int i = 0; i < rank; i++) if (reduce[i]) reducedCount *= dims[i];
        int groupCount = checked((int)(total / reducedCount));

        // Map each flat element to its group index by stripping the reduced axes.
        // Compute per-group sum and sum-of-squares.
        var sum = new float[groupCount];
        var sumSq = new float[groupCount];

        Span<float> xs = x.Span;
        var idx = new int[rank];
        for (long flat = 0; flat < total; flat++)
        {
            // Decode multi-index from flat.
            long rem = flat;
            int group = 0, groupStride = 1;
            for (int i = 0; i < rank; i++)
            {
                idx[i] = (int)(rem / strides[i]);
                rem %= strides[i];
            }
            // Build group index from kept axes (row-major over kept axes).
            for (int i = rank - 1; i >= 0; i--)
            {
                if (reduce[i]) continue;
                group += idx[i] * groupStride;
                groupStride *= dims[i];
            }
            float v = xs[(int)flat];
            sum[group] += v;
            sumSq[group] += v * v;
        }

        var mean = new float[groupCount];
        var inv = new float[groupCount];
        for (int gp = 0; gp < groupCount; gp++)
        {
            float m = sum[gp] / reducedCount;
            float var = sumSq[gp] / reducedCount - m * m;
            mean[gp] = m;
            inv[gp] = 1f / MathF.Sqrt(var + eps);
        }

        var y = new Tensor<float>(x.Shape);
        Span<float> ys = y.Span;
        for (long flat = 0; flat < total; flat++)
        {
            long rem = flat;
            int group = 0, groupStride = 1;
            for (int i = 0; i < rank; i++)
            {
                idx[i] = (int)(rem / strides[i]);
                rem %= strides[i];
            }
            for (int i = rank - 1; i >= 0; i--)
            {
                if (reduce[i]) continue;
                group += idx[i] * groupStride;
                groupStride *= dims[i];
            }
            ys[(int)flat] = (xs[(int)flat] - mean[group]) * inv[group];
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Lp normalization: normalize each slice along <c>axis</c> (default -1) by its
/// L1 (<c>p</c>=1) or L2 (<c>p</c>=2, default) norm. A zero norm leaves the slice
/// untouched (divides by 1) to avoid producing NaNs.
/// </summary>
public sealed class LpNormalizationKernel : IKernel
{
    public string OpType => "LpNormalization";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;

        int axis = (int)Attr.Int(node, "axis", -1);
        if (axis < 0) axis += rank;
        int p = (int)Attr.Int(node, "p", 2);

        int axisDim = dims[axis];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= dims[i];
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= dims[i];

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;

        // For each (outer, inner) pair, the slice walks axisDim steps of stride `inner`.
        for (int o = 0; o < outer; o++)
        for (int j = 0; j < inner; j++)
        {
            int baseI = o * axisDim * inner + j;

            float norm = 0f;
            if (p == 1)
            {
                for (int k = 0; k < axisDim; k++) norm += MathF.Abs(xs[baseI + k * inner]);
            }
            else
            {
                for (int k = 0; k < axisDim; k++) { float v = xs[baseI + k * inner]; norm += v * v; }
                norm = MathF.Sqrt(norm);
            }

            float div = norm == 0f ? 1f : norm;
            for (int k = 0; k < axisDim; k++)
            {
                int idx = baseI + k * inner;
                ys[idx] = xs[idx] / div;
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Local response normalization (LRN): each element is divided by
/// (bias + alpha/size · Σ x²)^beta, where the sum of squares runs over the
/// <c>size</c>-element neighborhood of channels centered on the element's channel.
/// Input: X [N, C, …spatial]. Attributes: <c>size</c>, <c>alpha</c> (1e-4),
/// <c>beta</c> (0.75), <c>bias</c> (1.0).
/// </summary>
public sealed class LRNKernel : IKernel
{
    public string OpType => "LRN";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        int size = (int)Attr.Int(node, "size", 1);
        float alpha = Attr.Float(node, "alpha", 1e-4f);
        float beta = Attr.Float(node, "beta", 0.75f);
        float bias = Attr.Float(node, "bias", 1f);

        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int N = dims[0], C = dims[1];
        int spatial = 1;
        for (int i = 2; i < dims.Length; i++) spatial *= dims[i];

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;

        // ONNX neighborhood: [c - floor((size-1)/2), c + ceil((size-1)/2)].
        int half = (size - 1) / 2;
        int hiExtra = size - 1 - half;
        float coeff = alpha / size;

        for (int n = 0; n < N; n++)
        for (int s = 0; s < spatial; s++)
        for (int c = 0; c < C; c++)
        {
            int lo = c - half;
            if (lo < 0) lo = 0;
            int hi = c + hiExtra;
            if (hi > C - 1) hi = C - 1;

            float sq = 0f;
            for (int cc = lo; cc <= hi; cc++)
            {
                float v = xs[(n * C + cc) * spatial + s];
                sq += v * v;
            }

            int idx = (n * C + c) * spatial + s;
            ys[idx] = xs[idx] / MathF.Pow(bias + coeff * sq, beta);
        }

        ctx.Set(node.Outputs[0], y);
    }
}
