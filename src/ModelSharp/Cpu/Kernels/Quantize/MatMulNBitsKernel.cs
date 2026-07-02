using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Cpu.Kernels.Linear;
using ModelSharp.Cpu.Kernels.Llm;
using ModelSharp.Graph;
using ModelSharp.Native;
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
/// <c>W[n, k] = (q − zero_point[n, b]) · scale[n, b]</c>.</para>
/// <para><b>Fused block kernel (default).</b> Per output column we read the packed weight row ONCE
/// and process it a quant block at a time: the block's nibbles are unpacked with SIMD mask/shift
/// (<see cref="MatMulNBitsSimd.DequantBlock4"/>), turned into <c>(q − zp)·scale</c> floats in a
/// small block-sized scratch, and FMA-accumulated (<see cref="MatMulNBitsSimd.FmaAccumulate"/>)
/// against the matching activation slice of every A row. We never materialize a full K-length fp32
/// weight vector, so the packed 0.5-byte codes — not 4-byte fp32 — dominate the memory traffic. The
/// fp32 dequant math is identical to a scalar dequant-then-dot; only the summation order changes (a
/// few ULP), which stays inside the parity tolerance.</para>
/// <para><b>Opt-in W4A8 (<c>MODELSHARP_W4A8</c>, OFF by default).</b> When enabled for 4-bit
/// weights, the activation block is dynamically quantized to int8 and dotted against the packed int4
/// codes with an int32 SIMD dot (AVX-VNNI <c>vpdpbusd</c> or AVX2 <c>vpmaddubsw</c>), rescaling and
/// applying zero-point compensation after. Lower precision, far less traffic; disabled by default so
/// the tight parity tests are unaffected.</para>
/// </remarks>
public sealed class MatMulNBitsKernel : IKernel
{
    public string OpType => "MatMulNBits";

    /// <summary>
    /// Opt-in flag for the fast, lower-precision W4A8 path (int8 activations × int4 weights via
    /// VNNI/maddubs). OFF by default, mirroring the <c>MODELSHARP_</c> env-flag convention used for
    /// the native quant paths; set <c>MODELSHARP_W4A8=1</c> (or <c>true</c>/<c>on</c>) to enable.
    /// When off, the kernel takes the exact fp32-preserving default path (byte-for-byte the prior
    /// behaviour up to SIMD accumulation order), so the tight parity tests stay green.
    /// </summary>
    private static readonly bool W4A8Enabled = ReadW4A8Flag();

    private static bool ReadW4A8Flag()
    {
        string? v = Environment.GetEnvironmentVariable("MODELSHARP_W4A8");
        if (string.IsNullOrEmpty(v)) return false;
        return v is "1" or "true" or "TRUE" or "True" or "on" or "ON" or "yes" or "YES";
    }

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

        // ---- Optional native fast path (opt-in; default off) ------------------------------------
        // The native ABI takes packed-uint8 (or null/symmetric) zero points only; if float zero
        // points are present we must run managed. Requires contiguous array-backed buffers.
        if ((NativeQuant.W4A8Enabled || NativeQuant.NativeNBitsEnabled) && zpFloat is null
            && System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(a.Buffer, out var aSeg)
            && aSeg.Offset == 0 && aSeg.Array is not null
            && System.Runtime.InteropServices.MemoryMarshal.TryGetArray<byte>(b.Buffer, out var bSeg)
            && bSeg.Offset == 0 && bSeg.Array is not null
            && System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(scales.Buffer, out var sSeg)
            && sSeg.Offset == 0 && sSeg.Array is not null
            && System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(y.Buffer, out var ySeg)
            && ySeg.Offset == 0 && ySeg.Array is not null)
        {
            if (NativeQuant.TryMatMulNBits(
                    aSeg.Array, bSeg.Array, sSeg.Array, zpPacked, ySeg.Array,
                    M, N, K, bits, blockSize))
            {
                ctx.Set(node.Outputs[0], y);
                return;
            }
        }

        int bRowBytes = nBlocksPerRow * blobSize;
        byte mask = (byte)((1 << bits) - 1);

        // Dense array backing so the per-column parallel lambdas can index from inside Parallel.For
        // (Span<T> is a ref struct and cannot be hoisted into a closure). KernelSimd.Array returns
        // the tensor's own zero-offset array or, for a sub-view, a dense copy — so element i always
        // equals the logical value at i.
        float[] aArr = KernelSimd.Array(a);
        float[] yArr = KernelSimd.Array(y);
        float[] scaleArr = KernelSimd.Array(scales);
        byte[] bArr = DenseByteArray(b);
        float[]? zpFloatLocal = zpFloat;
        byte[]? zpPackedLocal = zpPacked;
        int bitsLocal = bits, blockSizeLocal = blockSize, nBlocksLocal = nBlocksPerRow;
        int blobSizeLocal = blobSize, zpRowBytesLocal = zpRowBytes, KLocal = K, NLocal = N, MLocal = M;
        float defaultZpLocal = defaultZp;
        bool useW4A8 = W4A8Enabled && bits == 4 && MatMulNBitsSimd.W4A8Available;

        // Parallelise over the N output columns. Distinct columns write distinct output positions
        // (stride N), so there is no contention. Each column reads its packed weight row ONCE,
        // dequantizes a block at a time into a small (block-sized) scratch, and FMA-accumulates that
        // block against every activation row — so we never materialize a full K-length fp32 weight
        // vector and the packed weights dominate the traffic (not 8× fp32).
        MatMulParallel.For(N, (long)KLocal * (MLocal + 1), n =>
        {
            int bRowBase = n * bRowBytes;
            int scaleRowBase = n * nBlocksLocal;
            int zpRowBase = n * zpRowBytesLocal;

            if (useW4A8)
            {
                ExecuteColumnW4A8(
                    n, aArr, yArr, scaleArr, bArr, zpFloatLocal, zpPackedLocal,
                    bRowBase, scaleRowBase, zpRowBase, blockSizeLocal, nBlocksLocal,
                    blobSizeLocal, KLocal, NLocal, MLocal, defaultZpLocal, mask, bitsLocal);
                return;
            }

            // ---- DEFAULT exact-preserving path: fused block dequant + FMA accumulate --------------
            // Thread-local block scratch (only block-sized, not K) and per-row dot accumulators.
            var wBlock = new float[blockSizeLocal];
            var acc = new float[MLocal];

            for (int bk = 0; bk < nBlocksLocal; bk++)
            {
                float scale = scaleArr[scaleRowBase + bk];
                float zp;
                if (zpFloatLocal is not null) zp = zpFloatLocal[scaleRowBase + bk];
                else if (zpPackedLocal is not null) zp = UnpackNBit(zpPackedLocal, zpRowBase, bk, bitsLocal, mask);
                else zp = defaultZpLocal;

                int blobBase = bRowBase + bk * blobSizeLocal;
                int kStart = bk * blockSizeLocal;
                int len = Math.Min(blockSizeLocal, KLocal - kStart);
                if (len <= 0) break;

                // Vectorized nibble unpack + (q - zp) * scale into the block scratch.
                if (bitsLocal == 4)
                    MatMulNBitsSimd.DequantBlock4(bArr, blobBase, len, zp, scale, wBlock);
                else
                    MatMulNBitsSimd.DequantBlock8(bArr, blobBase, len, zp, scale, wBlock);

                // FMA-accumulate this weight block against the matching activation slice of each row.
                for (int m = 0; m < MLocal; m++)
                    acc[m] = MatMulNBitsSimd.FmaAccumulate(acc[m], aArr, m * KLocal + kStart, wBlock, 0, len);
            }

            for (int m = 0; m < MLocal; m++)
                yArr[m * NLocal + n] = acc[m];
        });

        ctx.Set(node.Outputs[0], y);
    }

    /// <summary>
    /// Opt-in W4A8 column kernel: dynamically quantizes each activation row's block to int8, dots it
    /// against the packed int4 weight codes with an int32 SIMD dot (VNNI / maddubs), then rescales
    /// by <c>scaleA · scaleW</c> and subtracts the zero-point compensation. Lower precision than the
    /// default path; only reachable when <c>MODELSHARP_W4A8</c> is set. Correctness identity:
    /// <c>Σ a·(q−zp)·s = s·(Σ a_q·q)·sA − s·zp·(Σ a_q)·sA</c>, where a_q are the int8 codes and
    /// <c>sA</c> the activation scale.
    /// </summary>
    private static void ExecuteColumnW4A8(
        int n, float[] aArr, float[] yArr, float[] scaleArr, byte[] bArr,
        float[]? zpFloatLocal, byte[]? zpPackedLocal,
        int bRowBase, int scaleRowBase, int zpRowBase, int blockSize, int nBlocks,
        int blobSize, int K, int N, int M, float defaultZp, byte mask, int bits)
    {
        // Weight codes for one block (0..15), as bytes, for the int8 dot. Small blocks stack; large
        // (pathological) blocks fall back to the heap so we never blow the stack.
        bool onStack = blockSize <= 8192;
        Span<byte> wCodes = onStack ? stackalloc byte[blockSize] : new byte[blockSize];
        Span<sbyte> aq = onStack ? stackalloc sbyte[blockSize] : new sbyte[blockSize];
        Span<byte> aqU = onStack ? stackalloc byte[blockSize] : new byte[blockSize]; // +128 (unsigned)

        for (int m = 0; m < M; m++)
        {
            float total = 0f;
            for (int bk = 0; bk < nBlocks; bk++)
            {
                float scaleW = scaleArr[scaleRowBase + bk];
                float zp;
                if (zpFloatLocal is not null) zp = zpFloatLocal[scaleRowBase + bk];
                else if (zpPackedLocal is not null) zp = UnpackNBit(zpPackedLocal, zpRowBase, bk, bits, mask);
                else zp = defaultZp;

                int blobBase = bRowBase + bk * blobSize;
                int kStart = bk * blockSize;
                int len = Math.Min(blockSize, K - kStart);
                if (len <= 0) break;

                // Unpack the len weight codes into bytes.
                for (int i = 0; i < len; i++)
                {
                    int b = bArr[blobBase + (i >> 1)];
                    wCodes[i] = (byte)(((i & 1) == 0) ? (b & 0x0F) : (b >> 4) & 0x0F);
                }

                // Dynamically int8-quantize this activation block.
                float scaleA = MatMulNBitsSimd.QuantizeActivationInt8(aArr, m * K + kStart, len, aq, out int sumAq);
                if (scaleA == 0f) continue;

                // Shift activation codes to unsigned [0,255] for the u8×u8 VNNI/maddubs dot, then
                // remove the +128 bias analytically: Σ (a_q+128)·q − 128·Σ q.
                int sumW = 0;
                for (int i = 0; i < len; i++)
                {
                    aqU[i] = (byte)(aq[i] + 128);
                    sumW += wCodes[i];
                }
                int rawU = MatMulNBitsSimd.DotU8xU8(aqU.Slice(0, len), wCodes.Slice(0, len), len);
                int dotAqQ = rawU - 128 * sumW;   // Σ a_q · q

                // Σ a·(q − zp)·s = s·sA·(Σ a_q·q − zp·Σ a_q)
                total += scaleW * scaleA * (dotAqQ - zp * sumAq);
            }
            yArr[m * N + n] = total;
        }
    }

    /// <summary>Dense zero-offset byte array for <paramref name="t"/> so it can be indexed inside a
    /// <c>Parallel.For</c> body; copies only for a non-trivial sub-view.</summary>
    private static byte[] DenseByteArray(Tensor<byte> t)
    {
        if (System.Runtime.InteropServices.MemoryMarshal.TryGetArray<byte>(t.Buffer, out var seg)
            && seg.Offset == 0 && seg.Array is { } arr && arr.Length == t.Buffer.Length)
            return arr;
        return t.Buffer.Span.ToArray();
    }

    /// <summary>
    /// Unpacks the <paramref name="index"/>-th <paramref name="bits"/>-bit value from the packed
    /// region starting at <paramref name="baseByte"/>. Values are laid out least-significant-first:
    /// for <c>bits = 4</c> the even index is the low nibble and the odd index the high nibble; for
    /// <c>bits = 8</c> one value per byte.
    /// </summary>
    private static int UnpackNBit(ReadOnlySpan<byte> data, int baseByte, int index, int bits, byte mask)
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
