using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Quantize;

/// <summary>
/// Block-wise n-bit (INT4 / INT8) quantized matrix multiply — the Microsoft contrib op
/// <c>com.microsoft.MatMulNBits</c>. This is the weight-quantized Linear used by the
/// ONNX Runtime GenAI INT4 LLM exports (Mistral-7B, Phi-3, Llama, …): the dense weight is
/// a <c>[N, K]</c> matrix stored as packed n-bit values with per-(row, block) scales and
/// optional zero points, and the op computes <c>Y = A · dequant(B)ᵀ</c>, i.e. for each
/// output column <c>n</c>: <c>Y[..., n] = Σ_k A[..., k] · W[n, k]</c>.
/// </summary>
/// <remarks>
/// <para><b>Layout / packing (matched to the ONNX Runtime kernel):</b></para>
/// <list type="bullet">
/// <item><description><c>A</c>: float <c>[..., K]</c> — the activations; leading dims are
/// batched like the standard MatMul (each row of the flattened <c>[M, K]</c> matrix is an
/// independent dot product against every weight row).</description></item>
/// <item><description><c>B</c>: uint8 <c>[N, n_blocks_per_row, blob_size]</c> where
/// <c>n_blocks_per_row = ceil(K / block_size)</c> and <c>blob_size = ceil(block_size · bits / 8)</c>.
/// Quantized values are packed least-significant-first: for <c>bits = 4</c>, two weights per
/// byte with the <b>low nibble</b> holding the even index and the <b>high nibble</b> the odd
/// index; for <c>bits = 8</c> one weight per byte.</description></item>
/// <item><description><c>scales</c>: float <c>[N · n_blocks_per_row]</c>, indexed
/// <c>scales[n · n_blocks_per_row + b]</c>.</description></item>
/// <item><description><c>zero_points</c> (optional): either float <c>[N · n_blocks_per_row]</c>
/// (one per (row, block)), or uint8 packed the same n-bit way as <c>B</c> with row stride
/// <c>ceil(n_blocks_per_row · bits / 8)</c> bytes. If <b>absent</b>, the default zero point is
/// the symmetric <c>2^(bits-1)</c> (8 for 4-bit, 128 for 8-bit).</description></item>
/// <item><description><c>g_idx</c> (optional): a per-column block-permutation index. Only the
/// trivial/identity form is supported; a non-trivial table throws.</description></item>
/// </list>
/// <para><b>Dequantization.</b> For output row <c>n</c> and input column <c>k</c>, the block is
/// <c>b = k / block_size</c>, the n-bit code is <c>q = unpack(B[n], k)</c>, and
/// <c>W[n, k] = (q − zero_point[n, b]) · scale[n, b]</c>. We dequantize each weight row to a
/// scratch float vector of length <c>K</c> on the fly and accumulate the dot product against
/// every A row, so memory stays O(K) per output column.</para>
/// </remarks>
public sealed class MatMulNBitsKernel : IKernel
{
    public string OpType => "MatMulNBits";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> a = ctx.Get(node.Inputs[0]);
        Tensor<byte> b = QuantizeOps.AsUInt8(ctx.GetTensor(node.Inputs[1]));

        int N = (int)Attr.Int(node, "N", 0);
        int K = (int)Attr.Int(node, "K", 0);
        int bits = (int)Attr.Int(node, "bits", 4);
        int blockSize = (int)Attr.Int(node, "block_size", 0);

        if (N <= 0 || K <= 0)
            throw new ModelSharpException(
                $"MatMulNBits requires positive N and K attributes (got N={N}, K={K}).");
        if (bits != 4 && bits != 8)
            throw new ModelSharpException($"MatMulNBits supports bits=4 or bits=8; got {bits}.");
        if (blockSize <= 0)
            throw new ModelSharpException($"MatMulNBits requires a positive block_size; got {blockSize}.");

        int nBlocksPerRow = (K + blockSize - 1) / blockSize;
        int blobSize = (blockSize * bits + 7) / 8;

        // ---- Validate B layout ------------------------------------------------------------------
        ReadOnlySpan<int> bDims = b.Shape.Dimensions;
        long expectedB = (long)N * nBlocksPerRow * blobSize;
        if (b.Length != expectedB)
            throw new ModelSharpException(
                $"MatMulNBits B has {b.Length} elements; expected {expectedB} "
                + $"(N={N} × n_blocks_per_row={nBlocksPerRow} × blob_size={blobSize}).");

        // ---- scales -----------------------------------------------------------------------------
        Tensor<float> scales = ctx.Get(node.Inputs[2]);
        long expectedScales = (long)N * nBlocksPerRow;
        if (scales.Length != expectedScales)
            throw new ModelSharpException(
                $"MatMulNBits scales has {scales.Length} elements; expected {expectedScales} "
                + $"(N={N} × n_blocks_per_row={nBlocksPerRow}).");

        // ---- zero points (optional): float per-block, or packed n-bit, or default symmetric -----
        float defaultZp = 1 << (bits - 1);   // 8 for 4-bit, 128 for 8-bit
        float[]? zpFloat = null;             // [N * nBlocksPerRow] when present
        byte[]? zpPacked = null;             // packed uint8 when present
        int zpRowBytes = (nBlocksPerRow * bits + 7) / 8;
        bool hasZp = node.Inputs.Count > 3 && ctx.Has(node.Inputs[3]);
        if (hasZp)
        {
            Tensor zpT = ctx.GetTensor(node.Inputs[3]);
            if (zpT.Dtype == ElementType.Float32)
            {
                zpFloat = zpT.AsFloat().Span.ToArray();
                if (zpFloat.Length != expectedScales)
                    throw new ModelSharpException(
                        $"MatMulNBits float zero_points has {zpFloat.Length} elements; "
                        + $"expected {expectedScales}.");
            }
            else if (zpT.Dtype == ElementType.UInt8)
            {
                zpPacked = QuantizeOps.AsUInt8(zpT).Span.ToArray();
                long expectedZp = (long)N * zpRowBytes;
                if (zpPacked.Length != expectedZp)
                    throw new ModelSharpException(
                        $"MatMulNBits packed zero_points has {zpPacked.Length} bytes; "
                        + $"expected {expectedZp} (N={N} × {zpRowBytes}).");
            }
            else
            {
                throw new ModelSharpException(
                    $"MatMulNBits zero_points must be Float32 or UInt8; got {zpT.Dtype}.");
            }
        }

        // ---- g_idx (optional): only the trivial identity permutation is supported ---------------
        if (node.Inputs.Count > 4 && ctx.Has(node.Inputs[4]))
        {
            long[] gIdx = TensorInts.Read(ctx.GetTensor(node.Inputs[4]));
            for (int k = 0; k < gIdx.Length; k++)
                if (gIdx[k] != k / blockSize)
                    throw new ModelSharpException(
                        "MatMulNBits g_idx with a non-trivial block permutation is not supported.");
        }

        // ---- Flatten A's leading dims to [M, K]; output is leadingDims + [N] ---------------------
        ReadOnlySpan<int> aDims = a.Shape.Dimensions;
        if (aDims.Length == 0 || aDims[aDims.Length - 1] != K)
            throw new ModelSharpException(
                $"MatMulNBits A inner dimension must equal K={K}; got shape [{string.Join(",", aDims.ToArray())}].");

        int M = 1;
        for (int i = 0; i < aDims.Length - 1; i++) M *= aDims[i];

        var outDims = new int[aDims.Length];
        for (int i = 0; i < aDims.Length - 1; i++) outDims[i] = aDims[i];
        outDims[aDims.Length - 1] = N;
        var y = new Tensor<float>(new TensorShape(outDims));

        Span<float> sa = a.Span, sy = y.Span, sScales = scales.Span;
        Span<byte> sb = b.Span;

        int bRowBytes = nBlocksPerRow * blobSize;
        byte mask = (byte)((1 << bits) - 1);
        var w = new float[K];   // scratch dequantized weight row, reused per output column

        for (int n = 0; n < N; n++)
        {
            // Dequantize weight row n into w[0..K).
            int bRowBase = n * bRowBytes;
            int scaleRowBase = n * nBlocksPerRow;
            int zpRowBase = n * zpRowBytes;

            for (int bk = 0; bk < nBlocksPerRow; bk++)
            {
                float scale = sScales[scaleRowBase + bk];

                float zp;
                if (zpFloat is not null) zp = zpFloat[scaleRowBase + bk];
                else if (zpPacked is not null) zp = UnpackNBit(zpPacked, zpRowBase, bk, bits, mask);
                else zp = defaultZp;

                int blobBase = bRowBase + bk * blobSize;
                int kStart = bk * blockSize;
                int kEnd = Math.Min(kStart + blockSize, K);
                for (int k = kStart; k < kEnd; k++)
                {
                    int inBlock = k - kStart;               // index within the block
                    int q = UnpackNBit(sb, blobBase, inBlock, bits, mask);
                    w[k] = (q - zp) * scale;
                }
            }

            // Accumulate Y[m, n] = Σ_k A[m, k] * w[k] for every A row.
            for (int m = 0; m < M; m++)
            {
                int aRow = m * K;
                float sum = 0f;
                for (int k = 0; k < K; k++) sum += sa[aRow + k] * w[k];
                sy[m * N + n] = sum;
            }
        }

        ctx.Set(node.Outputs[0], y);
    }

    /// <summary>
    /// Unpacks the <paramref name="index"/>-th <paramref name="bits"/>-bit value from the packed
    /// region starting at <paramref name="baseByte"/>. Values are laid out least-significant-first:
    /// for <c>bits = 4</c> the even index is the low nibble and the odd index the high nibble; for
    /// <c>bits = 8</c> one value per byte.
    /// </summary>
    private static int UnpackNBit(Span<byte> data, int baseByte, int index, int bits, byte mask)
    {
        if (bits == 8) return data[baseByte + index];
        // bits == 4
        int byteOff = baseByte + (index >> 1);
        int shift = (index & 1) * 4;        // even → low nibble (shift 0), odd → high nibble (shift 4)
        return (data[byteOff] >> shift) & mask;
    }

    private static int UnpackNBit(byte[] data, int baseByte, int index, int bits, byte mask)
        => UnpackNBit(data.AsSpan(), baseByte, index, bits, mask);
}
