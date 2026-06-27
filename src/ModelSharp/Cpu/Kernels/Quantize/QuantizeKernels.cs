using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Quantize;

/// <summary>
/// Shared helpers for the quantization kernels: integer-tensor readers (the base
/// <see cref="Tensor"/> type lacks dedicated <c>AsInt8</c>/<c>AsByte</c> accessors,
/// so these recover the typed views directly), saturating rounding, and the
/// per-axis broadcast index math used by <c>QuantizeLinear</c>/<c>DequantizeLinear</c>.
/// </summary>
internal static class QuantizeOps
{
    /// <summary>Reads any quantized integer tensor (int8/uint8/int32) elementwise as <c>double</c>.</summary>
    public static double[] ReadIntAsDoubles(Tensor t)
    {
        int n = checked((int)t.Length);
        var r = new double[n];
        switch (t.Dtype)
        {
            case ElementType.Int8:
            {
                Span<sbyte> s = AsInt8(t).Span;
                for (int i = 0; i < n; i++) r[i] = s[i];
                break;
            }
            case ElementType.UInt8:
            {
                Span<byte> s = AsUInt8(t).Span;
                for (int i = 0; i < n; i++) r[i] = s[i];
                break;
            }
            case ElementType.Int32:
            {
                Span<int> s = t.AsInt32().Span;
                for (int i = 0; i < n; i++) r[i] = s[i];
                break;
            }
            default:
                throw new ModelSharpException(
                    $"Quantization op expected an int8/uint8/int32 tensor, got {t.Dtype}.");
        }
        return r;
    }

    /// <summary>Recovers an int8 (<c>sbyte</c>) tensor view; throws on a dtype mismatch.</summary>
    public static Tensor<sbyte> AsInt8(Tensor t) =>
        t as Tensor<sbyte>
        ?? throw new ModelSharpException($"Tensor dtype is {t.Dtype}; expected Int8.");

    /// <summary>Recovers a uint8 (<c>byte</c>) tensor view; throws on a dtype mismatch.</summary>
    public static Tensor<byte> AsUInt8(Tensor t) =>
        t as Tensor<byte>
        ?? throw new ModelSharpException($"Tensor dtype is {t.Dtype}; expected UInt8.");

    /// <summary>IEEE round-half-to-even ("banker's rounding") of a double.</summary>
    public static double RoundHalfToEven(double v) => Math.Round(v, MidpointRounding.ToEven);

    /// <summary>Saturating-cast a (rounded) value into the int8 range [-128, 127].</summary>
    public static sbyte SaturateInt8(double v)
    {
        if (v <= -128d) return -128;
        if (v >= 127d) return 127;
        return (sbyte)v;
    }

    /// <summary>Saturating-cast a (rounded) value into the uint8 range [0, 255].</summary>
    public static byte SaturateUInt8(double v)
    {
        if (v <= 0d) return 0;
        if (v >= 255d) return 255;
        return (byte)v;
    }

    /// <summary>
    /// Resolves the positive axis from an attribute (supporting negative indexing) for a
    /// given rank. Returns <paramref name="rank"/>-relative axis.
    /// </summary>
    public static int ResolveAxis(int axis, int rank)
    {
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw new ModelSharpException($"Quantization axis {axis} is out of range for rank {rank}.");
        return axis;
    }

    /// <summary>
    /// For each flat element index of a tensor with the given <paramref name="dims"/>, returns the
    /// index along <paramref name="axis"/> (used to pick the matching per-channel scale/zero-point).
    /// </summary>
    public static int[] AxisIndices(ReadOnlySpan<int> dims, int axis)
    {
        int n = 1;
        foreach (int d in dims) n *= d;
        var idx = new int[n];
        int inner = 1;
        for (int i = axis + 1; i < dims.Length; i++) inner *= dims[i];
        int axisSize = dims[axis];
        // The axis index of flat position p is (p / inner) % axisSize.
        for (int p = 0; p < n; p++) idx[p] = (p / inner) % axisSize;
        return idx;
    }
}

/// <summary>
/// ONNX <c>DequantizeLinear</c>: <c>y = (x - x_zero_point) * x_scale</c>. Input <c>x</c> is
/// int8/uint8/int32; <c>x_scale</c> is a float scalar (per-tensor) or 1-D along the
/// <c>axis</c> attribute (per-channel); <c>x_zero_point</c> is optional and shares <c>x</c>'s
/// dtype. Output is always Float32.
/// </summary>
public sealed class DequantizeLinearKernel : IKernel
{
    public string OpType => "DequantizeLinear";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor x = ctx.GetTensor(node.Inputs[0]);
        Tensor<float> scaleT = ctx.Get(node.Inputs[1]);
        Tensor? zpT = node.Inputs.Count > 2 && node.Inputs[2].Length > 0
            ? ctx.GetTensor(node.Inputs[2])
            : null;

        double[] xv = QuantizeOps.ReadIntAsDoubles(x);
        Span<float> scale = scaleT.Span;
        double[]? zp = zpT is not null ? QuantizeOps.ReadIntAsDoubles(zpT) : null;

        int n = xv.Length;
        var y = new float[n];

        bool perAxis = scaleT.Shape.Rank > 0 && scaleT.Length > 1;
        if (!perAxis)
        {
            double s = scale.Length > 0 ? scale[0] : 1f;
            double z = zp is not null && zp.Length > 0 ? zp[0] : 0d;
            for (int i = 0; i < n; i++) y[i] = (float)((xv[i] - z) * s);
        }
        else
        {
            int axis = QuantizeOps.ResolveAxis((int)Attr.Int(node, "axis", 1), x.Shape.Rank);
            int[] ax = QuantizeOps.AxisIndices(x.Shape.Dimensions, axis);
            for (int i = 0; i < n; i++)
            {
                int c = ax[i];
                double z = zp is not null ? zp[c] : 0d;
                y[i] = (float)((xv[i] - z) * scale[c]);
            }
        }

        ctx.Set(node.Outputs[0], new Tensor<float>(x.Shape, y));
    }
}

/// <summary>
/// ONNX <c>QuantizeLinear</c>: <c>y = saturate(round(x / y_scale) + y_zero_point)</c> with
/// round-half-to-even. Input <c>x</c> is Float32; <c>y_scale</c> is a float scalar (per-tensor)
/// or 1-D along the <c>axis</c> attribute (per-channel). The optional <c>y_zero_point</c>
/// determines the output dtype (int8 or uint8); it defaults to uint8 with zero-point 0.
/// </summary>
public sealed class QuantizeLinearKernel : IKernel
{
    public string OpType => "QuantizeLinear";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> scaleT = ctx.Get(node.Inputs[1]);
        Tensor? zpT = node.Inputs.Count > 2 && node.Inputs[2].Length > 0
            ? ctx.GetTensor(node.Inputs[2])
            : null;

        Span<float> xv = x.Span;
        Span<float> scale = scaleT.Span;
        double[]? zp = zpT is not null ? QuantizeOps.ReadIntAsDoubles(zpT) : null;
        bool signed = zpT is not null && zpT.Dtype == ElementType.Int8;

        int n = checked((int)x.Length);
        bool perAxis = scaleT.Shape.Rank > 0 && scaleT.Length > 1;
        int[]? ax = null;
        if (perAxis)
        {
            int axis = QuantizeOps.ResolveAxis((int)Attr.Int(node, "axis", 1), x.Shape.Rank);
            ax = QuantizeOps.AxisIndices(x.Shape.Dimensions, axis);
        }

        if (signed)
        {
            var y = new sbyte[n];
            for (int i = 0; i < n; i++)
            {
                int c = ax is not null ? ax[i] : 0;
                double s = perAxis ? scale[c] : (scale.Length > 0 ? scale[0] : 1f);
                double z = zp is not null ? zp[perAxis ? c : 0] : 0d;
                double q = QuantizeOps.RoundHalfToEven(xv[i] / s) + z;
                y[i] = QuantizeOps.SaturateInt8(q);
            }
            ctx.Set(node.Outputs[0], new Tensor<sbyte>(x.Shape, y));
        }
        else
        {
            var y = new byte[n];
            for (int i = 0; i < n; i++)
            {
                int c = ax is not null ? ax[i] : 0;
                double s = perAxis ? scale[c] : (scale.Length > 0 ? scale[0] : 1f);
                double z = zp is not null ? zp[perAxis ? c : 0] : 0d;
                double q = QuantizeOps.RoundHalfToEven(xv[i] / s) + z;
                y[i] = QuantizeOps.SaturateUInt8(q);
            }
            ctx.Set(node.Outputs[0], new Tensor<byte>(x.Shape, y));
        }
    }
}

/// <summary>
/// ONNX <c>DynamicQuantizeLinear</c>: computes a per-tensor uint8 quantization of a Float32
/// input. <c>scale = (max(0, x_max) - min(0, x_min)) / 255</c>,
/// <c>zero_point = saturate(round(-x_min / scale))</c>, and
/// <c>y = saturate(round(x / scale) + zero_point)</c>. Produces three outputs:
/// <c>y</c> (uint8), <c>y_scale</c> (float scalar), and <c>y_zero_point</c> (uint8 scalar).
/// </summary>
public sealed class DynamicQuantizeLinearKernel : IKernel
{
    public string OpType => "DynamicQuantizeLinear";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Span<float> xv = x.Span;
        int n = checked((int)x.Length);

        float xmin = 0f, xmax = 0f;   // ONNX spec folds 0 into the range.
        for (int i = 0; i < n; i++)
        {
            float v = xv[i];
            if (v < xmin) xmin = v;
            if (v > xmax) xmax = v;
        }

        double scale = (xmax - xmin) / 255d;
        if (scale == 0d) scale = 1d;   // all-zero (or constant-zero) input: avoid divide-by-zero.

        byte zeroPoint = QuantizeOps.SaturateUInt8(QuantizeOps.RoundHalfToEven(-xmin / scale));

        var y = new byte[n];
        for (int i = 0; i < n; i++)
        {
            double q = QuantizeOps.RoundHalfToEven(xv[i] / scale) + zeroPoint;
            y[i] = QuantizeOps.SaturateUInt8(q);
        }

        ctx.Set(node.Outputs[0], new Tensor<byte>(x.Shape, y));
        ctx.Set(node.Outputs[1], new Tensor<float>(new TensorShape(), new[] { (float)scale }));
        ctx.Set(node.Outputs[2], new Tensor<byte>(new TensorShape(), new[] { zeroPoint }));
    }
}

/// <summary>
/// ONNX <c>MatMulInteger</c>: integer matrix multiply <c>Y = (A - a_zp) @ (B - b_zp)</c> with
/// int32 accumulation. <c>A</c> and <c>B</c> are uint8 or int8; the optional zero points share
/// each input's dtype (scalar per-tensor, or 1-D per-row of A / per-column of B). Supports 2-D
/// and batched (NumPy-broadcast) operands; the output is Int32.
/// </summary>
public sealed class MatMulIntegerKernel : IKernel
{
    public string OpType => "MatMulInteger";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor aT = ctx.GetTensor(node.Inputs[0]);
        Tensor bT = ctx.GetTensor(node.Inputs[1]);

        double[] a = QuantizeOps.ReadIntAsDoubles(aT);
        double[] b = QuantizeOps.ReadIntAsDoubles(bT);

        double[]? aZp = node.Inputs.Count > 2 && node.Inputs[2].Length > 0
            ? QuantizeOps.ReadIntAsDoubles(ctx.GetTensor(node.Inputs[2]))
            : null;
        double[]? bZp = node.Inputs.Count > 3 && node.Inputs[3].Length > 0
            ? QuantizeOps.ReadIntAsDoubles(ctx.GetTensor(node.Inputs[3]))
            : null;

        int[] adims = aT.Shape.Dimensions.ToArray();
        int[] bdims = bT.Shape.Dimensions.ToArray();
        if (adims.Length < 2 || bdims.Length < 2)
            throw new ModelSharpException("MatMulInteger requires inputs of rank >= 2.");

        int M = adims[^2], K = adims[^1];
        int Kb = bdims[^2], N = bdims[^1];
        if (K != Kb) throw new ModelSharpException($"MatMulInteger inner dimensions disagree: {K} vs {Kb}.");

        int[] aBatch = adims[..^2];
        int[] bBatch = bdims[..^2];
        int[] outBatch = Nd.BroadcastShape(aBatch, bBatch);
        int batchRank = outBatch.Length;
        int[] aBS = Nd.BroadcastStrides(aBatch, batchRank);
        int[] bBS = Nd.BroadcastStrides(bBatch, batchRank);

        int aMat = M * K, bMat = K * N, oMat = M * N;
        int totalBatch = 1;
        foreach (int d in outBatch) totalBatch *= d;

        var outDims = new int[batchRank + 2];
        Array.Copy(outBatch, outDims, batchRank);
        outDims[batchRank] = M;
        outDims[batchRank + 1] = N;
        var y = new int[totalBatch * oMat];

        // a_zp is per-row of A (length M) or per-tensor; b_zp is per-column of B (length N) or per-tensor.
        bool aZpPerRow = aZp is not null && aZp.Length == M && M != 1;
        bool bZpPerCol = bZp is not null && bZp.Length == N && N != 1;
        double aZpScalar = aZp is not null && aZp.Length > 0 ? aZp[0] : 0d;
        double bZpScalar = bZp is not null && bZp.Length > 0 ? bZp[0] : 0d;

        // Precompute, for each batch index, the A/B matrix offsets (in matrix units) so the row
        // work is embarrassingly parallel. Integer-as-double values are exact for the int8/uint8
        // ranges used here, and each output element is accumulated by a single thread, so the int32
        // result is bit-identical to the serial reduction (no cross-thread reordering of any sum).
        var aOffOf = new int[totalBatch];
        var bOffOf = new int[totalBatch];
        {
            var coord = new int[batchRank];
            int aOff = 0, bOff = 0;
            for (int bi = 0; bi < totalBatch; bi++)
            {
                aOffOf[bi] = aOff; bOffOf[bi] = bOff;
                for (int ax = batchRank - 1; ax >= 0; ax--)
                {
                    coord[ax]++; aOff += aBS[ax]; bOff += bBS[ax];
                    if (coord[ax] < outBatch[ax]) break;
                    coord[ax] = 0; aOff -= aBS[ax] * outBatch[ax]; bOff -= bBS[ax] * outBatch[ax];
                }
            }
        }

        long totalRows = (long)totalBatch * M;
        Linear.MatMulParallel.For(checked((int)totalRows), (long)K * N, row =>
        {
            int bi = row / M;
            int m = row % M;
            int aBase = aOffOf[bi] * aMat, bBase = bOffOf[bi] * bMat, oBase = bi * oMat;
            double az = aZp is null ? 0d : (aZpPerRow ? aZp[m] : aZpScalar);
            int aRow = aBase + m * K;
            int oRow = oBase + m * N;
            for (int nn = 0; nn < N; nn++)
            {
                double bz = bZp is null ? 0d : (bZpPerCol ? bZp[nn] : bZpScalar);
                long sum = 0;
                for (int kk = 0; kk < K; kk++)
                    sum += (long)(a[aRow + kk] - az) * (long)(b[bBase + kk * N + nn] - bz);
                y[oRow + nn] = (int)sum;
            }
        });

        ctx.Set(node.Outputs[0], new Tensor<int>(new TensorShape(outDims), y));
    }
}
