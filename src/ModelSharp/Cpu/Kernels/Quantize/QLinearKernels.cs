using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Quantize;

/// <summary>
/// Shared math for the <c>QLinear*</c> kernels and <c>ConvInteger</c>. Each op dequantizes its
/// integer inputs to float (<c>(q - zp) * scale</c>, per-tensor or per-channel), runs the existing
/// float kernel math, then re-quantizes the float result to the requested output dtype
/// (<c>saturate(round(y / y_scale) + y_zp)</c>, round-half-to-even) — exactly the reference
/// "dequant → float op → requant" path the tests compare against.
/// </summary>
internal static class QLinearOps
{
    /// <summary>Dequantizes a quantized tensor to a fresh float array. Per-tensor or per-channel
    /// along <paramref name="axis"/> (when the scale is a vector matching that axis).</summary>
    public static float[] Dequantize(Tensor q, Tensor<float> scaleT, Tensor? zpT, int axis)
    {
        double[] qv = QuantizeOps.ReadIntAsDoubles(q);
        Span<float> scale = scaleT.Span;
        double[]? zp = zpT is not null ? QuantizeOps.ReadIntAsDoubles(zpT) : null;
        int n = qv.Length;
        var y = new float[n];

        bool perAxis = scaleT.Shape.Rank > 0 && scaleT.Length > 1;
        if (!perAxis)
        {
            double s = scale.Length > 0 ? scale[0] : 1d;
            double z = zp is not null && zp.Length > 0 ? zp[0] : 0d;
            for (int i = 0; i < n; i++) y[i] = (float)((qv[i] - z) * s);
        }
        else
        {
            int ax = QuantizeOps.ResolveAxis(axis, q.Shape.Rank);
            int[] idx = QuantizeOps.AxisIndices(q.Shape.Dimensions, ax);
            for (int i = 0; i < n; i++)
            {
                int c = idx[i];
                double z = zp is not null ? zp[c] : 0d;
                y[i] = (float)((qv[i] - z) * scale[c]);
            }
        }
        return y;
    }

    /// <summary>Quantizes a float buffer to int8/uint8 per the y_zero_point dtype (per-tensor).</summary>
    public static Tensor Quantize(float[] data, TensorShape shape, Tensor<float> yScaleT, Tensor? yZpT)
    {
        double s = yScaleT.Span.Length > 0 ? yScaleT.Span[0] : 1d;
        bool signed = yZpT is not null && yZpT.Dtype == ElementType.Int8;
        double z = yZpT is not null ? QuantizeOps.ReadIntAsDoubles(yZpT)[0] : 0d;
        int n = data.Length;

        if (signed)
        {
            var y = new sbyte[n];
            for (int i = 0; i < n; i++)
                y[i] = QuantizeOps.SaturateInt8(QuantizeOps.RoundHalfToEven(data[i] / s) + z);
            return new Tensor<sbyte>(shape, y);
        }
        else
        {
            var y = new byte[n];
            for (int i = 0; i < n; i++)
                y[i] = QuantizeOps.SaturateUInt8(QuantizeOps.RoundHalfToEven(data[i] / s) + z);
            return new Tensor<byte>(shape, y);
        }
    }
}

/// <summary>
/// ONNX <c>QLinearMatMul</c>: quantized matrix multiply. Inputs are
/// <c>(a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp)</c>; <c>a</c>/<c>b</c> are int8 or uint8.
/// Computed as dequantize(a) @ dequantize(b) requantized to <c>y_zp</c>'s dtype. Full NumPy
/// batched-matmul broadcasting is supported.
/// </summary>
public sealed class QLinearMatMulKernel : IKernel
{
    public string OpType => "QLinearMatMul";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor aQ = ctx.GetTensor(node.Inputs[0]);
        Tensor<float> aScale = ctx.Get(node.Inputs[1]);
        Tensor aZp = ctx.GetTensor(node.Inputs[2]);
        Tensor bQ = ctx.GetTensor(node.Inputs[3]);
        Tensor<float> bScale = ctx.Get(node.Inputs[4]);
        Tensor bZp = ctx.GetTensor(node.Inputs[5]);
        Tensor<float> yScale = ctx.Get(node.Inputs[6]);
        Tensor yZp = ctx.GetTensor(node.Inputs[7]);

        // a is quantized per-row (axis = rank-2) and b per-column (axis = rank-1) when per-channel;
        // per-tensor scalars are handled inside Dequantize.
        float[] a = QLinearOps.Dequantize(aQ, aScale, aZp, aQ.Shape.Rank - 2);
        float[] b = QLinearOps.Dequantize(bQ, bScale, bZp, bQ.Shape.Rank - 1);

        var (yData, yShape) = MatMulFloat(a, aQ.Shape, b, bQ.Shape);
        ctx.Set(node.Outputs[0], QLinearOps.Quantize(yData, yShape, yScale, yZp));
    }

    /// <summary>Batched float matmul mirroring <c>MatMulKernel</c>, returning the flat buffer + shape.</summary>
    internal static (float[] data, TensorShape shape) MatMulFloat(
        float[] a, TensorShape aShape, float[] b, TensorShape bShape)
    {
        int[] adims = aShape.Dimensions.ToArray();
        int[] bdims = bShape.Dimensions.ToArray();
        if (adims.Length < 2 || bdims.Length < 2)
            throw new ModelSharpException("QLinearMatMul requires inputs of rank >= 2.");

        int M = adims[^2], K = adims[^1];
        int Kb = bdims[^2], N = bdims[^1];
        if (K != Kb) throw new ModelSharpException($"QLinearMatMul inner dimensions disagree: {K} vs {Kb}.");

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
        outDims[batchRank] = M; outDims[batchRank + 1] = N;
        var y = new float[totalBatch * oMat];

        // Precompute per-batch A/B offsets so output rows parallelise cleanly, and transpose each
        // distinct B matrix from [K,N] (column-strided) to row-major [N,K] once for a contiguous,
        // SIMD-friendly inner dot.
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

        var bTransposed = new System.Collections.Generic.Dictionary<int, float[]>();
        foreach (int bOff in bOffOf)
        {
            if (bTransposed.ContainsKey(bOff)) continue;
            var bt = new float[bMat];
            int bBase = bOff * bMat;
            for (int kk = 0; kk < K; kk++)
            {
                int src = bBase + kk * N;
                for (int nn = 0; nn < N; nn++) bt[nn * K + kk] = b[src + nn];
            }
            bTransposed[bOff] = bt;
        }

        long totalRows = (long)totalBatch * M;
        Linear.MatMulParallel.For(checked((int)totalRows), (long)K * N, row =>
        {
            int bi = row / M;
            int m = row % M;
            float[] bt = bTransposed[bOffOf[bi]];
            int aRow = aOffOf[bi] * aMat + m * K;
            int oRow = bi * oMat + m * N;
            for (int nn = 0; nn < N; nn++)
                y[oRow + nn] = Linear.MatMulParallel.Dot(a, aRow, bt.AsSpan(nn * K, K), K);
        });
        return (y, new TensorShape(outDims));
    }
}

/// <summary>
/// ONNX <c>QLinearAdd</c> (com.microsoft): quantized elementwise add with NumPy broadcasting.
/// Inputs <c>(A, A_scale, A_zp, B, B_scale, B_zp, C_scale, C_zp)</c>; computes
/// dequant(A) + dequant(B), requantized to C.
/// </summary>
public sealed class QLinearAddKernel : IKernel
{
    public string OpType => "QLinearAdd";
    public void Execute(GraphNode node, GraphContext ctx) =>
        QLinearBinary.Run(node, ctx, static (x, y) => x + y);
}

/// <summary>
/// ONNX <c>QLinearMul</c> (com.microsoft): quantized elementwise multiply with NumPy broadcasting.
/// Same operand layout as <see cref="QLinearAddKernel"/>; computes dequant(A) * dequant(B).
/// </summary>
public sealed class QLinearMulKernel : IKernel
{
    public string OpType => "QLinearMul";
    public void Execute(GraphNode node, GraphContext ctx) =>
        QLinearBinary.Run(node, ctx, static (x, y) => x * y);
}

/// <summary>Shared broadcasting driver for the quantized elementwise binary ops.</summary>
internal static class QLinearBinary
{
    public static void Run(GraphNode node, GraphContext ctx, Func<float, float, float> op)
    {
        Tensor aQ = ctx.GetTensor(node.Inputs[0]);
        Tensor<float> aScale = ctx.Get(node.Inputs[1]);
        Tensor aZp = ctx.GetTensor(node.Inputs[2]);
        Tensor bQ = ctx.GetTensor(node.Inputs[3]);
        Tensor<float> bScale = ctx.Get(node.Inputs[4]);
        Tensor bZp = ctx.GetTensor(node.Inputs[5]);
        Tensor<float> yScale = ctx.Get(node.Inputs[6]);
        Tensor yZp = ctx.GetTensor(node.Inputs[7]);

        // Elementwise: scales/zero-points are per-tensor scalars here (the common export form).
        float[] a = QLinearOps.Dequantize(aQ, aScale, aZp, 0);
        float[] b = QLinearOps.Dequantize(bQ, bScale, bZp, 0);

        int[] adims = aQ.Shape.Dimensions.ToArray();
        int[] bdims = bQ.Shape.Dimensions.ToArray();
        int[] outDims = Nd.BroadcastShape(adims, bdims);
        int rank = outDims.Length;
        int[] aS = Nd.BroadcastStrides(adims, rank);
        int[] bS = Nd.BroadcastStrides(bdims, rank);

        int total = 1; foreach (int d in outDims) total *= d;
        var y = new float[total];
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int i = 0; i < total; i++)
        {
            y[i] = op(a[aOff], b[bOff]);
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++; aOff += aS[ax]; bOff += bS[ax];
                if (coord[ax] < outDims[ax]) break;
                coord[ax] = 0; aOff -= aS[ax] * outDims[ax]; bOff -= bS[ax] * outDims[ax];
            }
        }

        ctx.Set(node.Outputs[0], QLinearOps.Quantize(y, new TensorShape(outDims), yScale, yZp));
    }
}
