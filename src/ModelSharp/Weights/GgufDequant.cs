using System;

namespace ModelSharp.Weights;

/// <summary>
/// Pure, allocation-light dequantizer for the ggml/llama.cpp block-quantized tensor formats stored
/// in GGUF files. Given the raw on-disk bytes of a quantized tensor, its <see cref="GgmlType"/>, and
/// the logical element count, <see cref="Dequantize"/> reconstructs a <c>float[]</c> in the natural
/// row-major element order (block 0 first, then block 1, …), exactly mirroring the reference
/// <c>dequantize_row_*</c> routines in <c>ggml-quants.c</c>.
/// <para>
/// The on-disk byte/element sizing per block is taken from <see cref="GgmlTypeInfo"/>; this class only
/// owns the bit-unpacking and the affine reconstruction. All scale fields are little-endian IEEE-754
/// half floats decoded with <see cref="DecodeHalf"/> (mirroring <c>GgufFile.DecodeFloat16</c>).
/// </para>
/// <para><b>Supported types</b>: the QK=32 legacy quants <see cref="GgmlType.Q4_0"/>,
/// <see cref="GgmlType.Q4_1"/>, <see cref="GgmlType.Q5_0"/>, <see cref="GgmlType.Q5_1"/>,
/// <see cref="GgmlType.Q8_0"/>, <see cref="GgmlType.Q8_1"/>; the QK_K=256 k-quants
/// <see cref="GgmlType.Q2_K"/>, <see cref="GgmlType.Q3_K"/>, <see cref="GgmlType.Q4_K"/>,
/// <see cref="GgmlType.Q5_K"/>, <see cref="GgmlType.Q6_K"/>, <see cref="GgmlType.Q8_K"/>; and the
/// non-linear-codebook importance-matrix quants <see cref="GgmlType.IQ4_NL"/> and
/// <see cref="GgmlType.IQ4_XS"/>, whose 16-entry codebook (<see cref="KValuesIq4Nl"/>) is exact.</para>
/// <para>
/// The remaining "IQ" families (<see cref="GgmlType.IQ2_XXS"/>, <see cref="GgmlType.IQ2_XS"/>,
/// <see cref="GgmlType.IQ2_S"/>, <see cref="GgmlType.IQ3_XXS"/>, <see cref="GgmlType.IQ3_S"/>,
/// <see cref="GgmlType.IQ1_S"/>, <see cref="GgmlType.IQ1_M"/>) encode each group as an index into a
/// large published lattice/grid constant table rather than an affine scale. Those grids are not
/// transcribed here, so those types — and any unrecognized type — throw rather than silently
/// emitting wrong output.</para>
/// </summary>
public static class GgufDequant
{
    /// <summary>Elements per super-block for the k-quant family.</summary>
    private const int QkK = 256;

    /// <summary>
    /// The 16-entry non-linear codebook shared by IQ4_NL and IQ4_XS (the reference
    /// <c>kvalues_iq4nl</c> in <c>ggml-quants.c</c>). A 4-bit code indexes directly into this table;
    /// the dequantized value is <c>scale · KValuesIq4Nl[code]</c>.
    /// </summary>
    public static ReadOnlySpan<sbyte> KValuesIq4Nl => new sbyte[]
    {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };

    /// <summary>
    /// Returns <c>true</c> if <see cref="Dequantize"/> can reconstruct the given quantized type.
    /// Unquantized scalar types return <c>false</c> (they are not this class's responsibility).
    /// </summary>
    public static bool IsSupported(GgmlType type) => type switch
    {
        GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_0 or GgmlType.Q5_1 or
        GgmlType.Q8_0 or GgmlType.Q8_1 or
        GgmlType.Q2_K or GgmlType.Q3_K or GgmlType.Q4_K or GgmlType.Q5_K or
        GgmlType.Q6_K or GgmlType.Q8_K or
        GgmlType.IQ4_NL or GgmlType.IQ4_XS => true,
        _ => false,
    };

    /// <summary>
    /// Dequantizes <paramref name="raw"/> — the packed bytes of a single quantized tensor of
    /// <paramref name="elementCount"/> elements and ggml type <paramref name="type"/> — into a
    /// freshly-allocated <c>float[elementCount]</c> in row-major order.
    /// </summary>
    /// <exception cref="ModelSharpException">
    /// The type is not a supported quantized type, the element count is not a whole multiple of the
    /// block size, or <paramref name="raw"/> is shorter than the implied number of blocks requires.
    /// </exception>
    public static float[] Dequantize(ReadOnlySpan<byte> raw, GgmlType type, long elementCount)
    {
        if (elementCount < 0)
            throw new ModelSharpException($"Element count must be non-negative (got {elementCount}).");
        if (!IsSupported(type))
            throw new ModelSharpException(
                $"ggml type {type} is not a dequantizable block type supported by ModelSharp. " +
                "Supported: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, " +
                "Q8_K, IQ4_NL, IQ4_XS. The grid-codebook IQ families (IQ2_XXS, IQ2_XS, IQ2_S, " +
                "IQ3_XXS, IQ3_S, IQ1_S, IQ1_M) are not yet implemented and are intentionally not " +
                "approximated.");

        int block = GgmlTypeInfo.BlockSize(type);
        int typeSize = GgmlTypeInfo.TypeSize(type);
        if (elementCount % block != 0)
            throw new ModelSharpException(
                $"ggml type {type}: element count {elementCount} is not a multiple of the block size {block}.");

        int count = checked((int)elementCount);
        int nBlocks = count / block;
        long needed = (long)nBlocks * typeSize;
        if (raw.Length < needed)
            throw new ModelSharpException(
                $"ggml type {type}: raw buffer is {raw.Length} bytes but {needed} are required for " +
                $"{nBlocks} block(s) of {typeSize} bytes.");

        var outArr = new float[count];
        Span<float> output = outArr;

        switch (type)
        {
            case GgmlType.Q4_0: DequantizeQ4_0(raw, output, nBlocks); break;
            case GgmlType.Q4_1: DequantizeQ4_1(raw, output, nBlocks); break;
            case GgmlType.Q5_0: DequantizeQ5_0(raw, output, nBlocks); break;
            case GgmlType.Q5_1: DequantizeQ5_1(raw, output, nBlocks); break;
            case GgmlType.Q8_0: DequantizeQ8_0(raw, output, nBlocks); break;
            case GgmlType.Q8_1: DequantizeQ8_1(raw, output, nBlocks); break;
            case GgmlType.Q2_K: DequantizeQ2_K(raw, output, nBlocks); break;
            case GgmlType.Q3_K: DequantizeQ3_K(raw, output, nBlocks); break;
            case GgmlType.Q4_K: DequantizeQ4_K(raw, output, nBlocks); break;
            case GgmlType.Q5_K: DequantizeQ5_K(raw, output, nBlocks); break;
            case GgmlType.Q6_K: DequantizeQ6_K(raw, output, nBlocks); break;
            case GgmlType.Q8_K: DequantizeQ8_K(raw, output, nBlocks); break;
            case GgmlType.IQ4_NL: DequantizeIq4Nl(raw, output, nBlocks); break;
            case GgmlType.IQ4_XS: DequantizeIq4Xs(raw, output, nBlocks); break;
            default:
                throw new ModelSharpException($"Unhandled ggml type {type}.");
        }

        return outArr;
    }

    // =====================================================================================
    // Legacy QK = 32 quants
    // =====================================================================================

    /// <summary>
    /// Q4_0 block (18 bytes): fp16 scale <c>d</c> (2 bytes) followed by 16 bytes holding 32 4-bit
    /// codes. The 32 codes are split: code <c>j</c> (j&lt;16) is the low nibble of byte <c>j</c>, and
    /// code <c>j+16</c> is the high nibble of byte <c>j</c>. Value = <c>d · (nibble − 8)</c>.
    /// </summary>
    private static void DequantizeQ4_0(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const int qk = 32;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 18;
            float d = DecodeHalf(raw, o);
            ReadOnlySpan<byte> qs = raw.Slice(o + 2, 16);
            int outBase = b * qk;
            for (int j = 0; j < qk / 2; j++)
            {
                int x0 = (qs[j] & 0x0F) - 8;
                int x1 = (qs[j] >> 4) - 8;
                output[outBase + j] = d * x0;
                output[outBase + j + qk / 2] = d * x1;
            }
        }
    }

    /// <summary>
    /// Q4_1 block (20 bytes): fp16 <c>d</c>, fp16 <c>m</c> (min), then 16 bytes of 32 nibbles, split
    /// low-half/high-half exactly as Q4_0. Value = <c>d · nibble + m</c>.
    /// </summary>
    private static void DequantizeQ4_1(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const int qk = 32;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 20;
            float d = DecodeHalf(raw, o);
            float m = DecodeHalf(raw, o + 2);
            ReadOnlySpan<byte> qs = raw.Slice(o + 4, 16);
            int outBase = b * qk;
            for (int j = 0; j < qk / 2; j++)
            {
                int x0 = qs[j] & 0x0F;
                int x1 = qs[j] >> 4;
                output[outBase + j] = d * x0 + m;
                output[outBase + j + qk / 2] = d * x1 + m;
            }
        }
    }

    /// <summary>
    /// Q5_0 block (22 bytes): fp16 <c>d</c>, a 4-byte little-endian high-bit field <c>qh</c>, then 16
    /// bytes of low nibbles. For code <c>j</c> (j&lt;16) the high bit is bit <c>j</c> of <c>qh</c>; for
    /// code <c>j+16</c> it is bit <c>j+16</c>. Value = <c>d · ((low | high&lt;&lt;4) − 16)</c>.
    /// </summary>
    private static void DequantizeQ5_0(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const int qk = 32;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 22;
            float d = DecodeHalf(raw, o);
            uint qh = ReadUInt32LE(raw, o + 2);
            ReadOnlySpan<byte> qs = raw.Slice(o + 6, 16);
            int outBase = b * qk;
            for (int j = 0; j < qk / 2; j++)
            {
                int xh0 = (int)((qh >> j) & 1) << 4;
                int xh1 = (int)((qh >> (j + 16)) & 1) << 4;
                int x0 = ((qs[j] & 0x0F) | xh0) - 16;
                int x1 = ((qs[j] >> 4) | xh1) - 16;
                output[outBase + j] = d * x0;
                output[outBase + j + qk / 2] = d * x1;
            }
        }
    }

    /// <summary>
    /// Q5_1 block (24 bytes): fp16 <c>d</c>, fp16 <c>m</c>, a 4-byte <c>qh</c>, then 16 bytes of low
    /// nibbles, packed as in Q5_0. Value = <c>d · (low | high&lt;&lt;4) + m</c>.
    /// </summary>
    private static void DequantizeQ5_1(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const int qk = 32;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 24;
            float d = DecodeHalf(raw, o);
            float m = DecodeHalf(raw, o + 2);
            uint qh = ReadUInt32LE(raw, o + 4);
            ReadOnlySpan<byte> qs = raw.Slice(o + 8, 16);
            int outBase = b * qk;
            for (int j = 0; j < qk / 2; j++)
            {
                int xh0 = (int)((qh >> j) & 1) << 4;
                int xh1 = (int)((qh >> (j + 16)) & 1) << 4;
                int x0 = (qs[j] & 0x0F) | xh0;
                int x1 = (qs[j] >> 4) | xh1;
                output[outBase + j] = d * x0 + m;
                output[outBase + j + qk / 2] = d * x1 + m;
            }
        }
    }

    /// <summary>
    /// Q8_0 block (34 bytes): fp16 <c>d</c> then 32 signed int8 codes. Value = <c>d · q</c>.
    /// </summary>
    private static void DequantizeQ8_0(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const int qk = 32;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 34;
            float d = DecodeHalf(raw, o);
            int outBase = b * qk;
            for (int j = 0; j < qk; j++)
                output[outBase + j] = d * unchecked((sbyte)raw[o + 2 + j]);
        }
    }

    /// <summary>
    /// Q8_1 block (36 bytes): fp16 <c>d</c>, fp16 <c>s</c> (a precomputed sum used only by the GEMM
    /// kernels, irrelevant to dequant), then 32 signed int8 codes. Value = <c>d · q</c>.
    /// </summary>
    private static void DequantizeQ8_1(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const int qk = 32;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 36;
            float d = DecodeHalf(raw, o);
            int outBase = b * qk;
            for (int j = 0; j < qk; j++)
                output[outBase + j] = d * unchecked((sbyte)raw[o + 4 + j]);
        }
    }

    // =====================================================================================
    // k-quant super-blocks (QK_K = 256)
    // =====================================================================================

    /// <summary>
    /// Q2_K super-block (84 bytes): <c>scales[16]</c> (each byte: low nibble = 4-bit scale, high
    /// nibble = 4-bit min), <c>qs[64]</c> (256 2-bit codes), fp16 <c>d</c>, fp16 <c>dmin</c>. The 256
    /// elements form 16 groups of 16. Group <c>n</c> uses 2-bit codes from bit-plane <c>n/8</c> of
    /// the 64 <c>qs</c> bytes that cover its 32-element half. Value =
    /// <c>d · scale · q − dmin · min</c>.
    /// </summary>
    private static void DequantizeQ2_K(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 84;
            ReadOnlySpan<byte> scales = raw.Slice(o, 16);
            ReadOnlySpan<byte> qs = raw.Slice(o + 16, 64);
            float d = DecodeHalf(raw, o + 80);
            float dmin = DecodeHalf(raw, o + 82);
            int outBase = b * QkK;

            // Mirrors dequantize_row_q2_K: iterate the super-block in two 128-element halves
            // (each half = a 32-byte qs window), then four 32-element shifts, then two 16-element
            // scale groups inside each shift.
            int outIdx = outBase;
            int qsBase = 0;
            int is_ = 0;
            for (int half = 0; half < QkK; half += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte sc0 = scales[is_++];
                    float dl0 = d * (sc0 & 0xF);
                    float ml0 = dmin * (sc0 >> 4);
                    for (int l = 0; l < 16; l++)
                        output[outIdx++] = dl0 * ((qs[qsBase + l] >> shift) & 3) - ml0;

                    byte sc1 = scales[is_++];
                    float dl1 = d * (sc1 & 0xF);
                    float ml1 = dmin * (sc1 >> 4);
                    for (int l = 0; l < 16; l++)
                        output[outIdx++] = dl1 * ((qs[qsBase + 16 + l] >> shift) & 3) - ml1;

                    shift += 2;
                }
                qsBase += 32;
            }
        }
    }

    /// <summary>
    /// Q3_K super-block (110 bytes): <c>hmask[32]</c> (1 high bit per code), <c>qs[64]</c> (256 2-bit
    /// low codes), <c>scales[12]</c> (16 6-bit signed scales packed across 12 bytes), fp16 <c>d</c>.
    /// Each code is a signed 3-bit value <c>(low2 | (highbit ? 0 : 4)) − 4</c>: the high mask bit is
    /// <i>inverted</i>, i.e. value = <c>(low2 − (hmaskbit==0 ? 4 : 0))</c>. The 16 scales are each
    /// <c>scale − 32</c>. Value = <c>d · scale · q</c>.
    /// </summary>
    private static void DequantizeQ3_K(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        Span<sbyte> aux = stackalloc sbyte[16];
        Span<uint> sc = stackalloc uint[4];

        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 110;
            ReadOnlySpan<byte> hmask = raw.Slice(o, 32);
            ReadOnlySpan<byte> qs = raw.Slice(o + 32, 64);
            ReadOnlySpan<byte> scalesRaw = raw.Slice(o + 96, 12);
            float d = DecodeHalf(raw, o + 108);
            int outBase = b * QkK;

            // Unpack the 16 6-bit scales (mirrors the scalar reference in ggml-quants.c).
            // scales layout: 12 bytes -> 4 uint32 words (last word only first 4 bytes).
            uint aux0 = ReadUInt32LE(scalesRaw, 0);
            uint aux1 = ReadUInt32LE(scalesRaw, 4);
            uint aux2 = ReadUInt32LE(scalesRaw, 8);
            uint tmp = aux2;
            // Reconstruct the 16 6-bit values into bytes.
            sc[0] = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
            sc[1] = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
            sc[2] = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            sc[3] = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            for (int i = 0; i < 16; i++)
            {
                int word = i / 4;
                int byteInWord = i % 4;
                aux[i] = (sbyte)((int)((sc[word] >> (byteInWord * 8)) & 0xFF) - 32);
            }

            int outIdx = outBase;
            int qsBase = 0;
            int scIdx = 0;
            byte m = 1;
            for (int half = 0; half < QkK; half += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    float dl0 = d * aux[scIdx++];
                    for (int l = 0; l < 16; l++)
                    {
                        int q = (qs[qsBase + l] >> shift) & 3;
                        if ((hmask[l] & m) == 0) q -= 4;
                        output[outIdx++] = dl0 * q;
                    }
                    float dl1 = d * aux[scIdx++];
                    for (int l = 0; l < 16; l++)
                    {
                        int q = (qs[qsBase + 16 + l] >> shift) & 3;
                        if ((hmask[16 + l] & m) == 0) q -= 4;
                        output[outIdx++] = dl1 * q;
                    }
                    shift += 2;
                    m <<= 1;
                }
                qsBase += 32;
            }
        }
    }

    /// <summary>
    /// Q4_K super-block (144 bytes): fp16 <c>d</c>, fp16 <c>dmin</c>, <c>scales[12]</c> (8 pairs of
    /// 6-bit scale/min packed in 12 bytes), <c>qs[128]</c> (256 4-bit codes). The 256 elements are 8
    /// groups of 32; group <c>j</c> uses scale/min <c>(sc, mn)</c> from
    /// <see cref="GetScaleMinK4"/>. Value = <c>d · sc · q − dmin · mn</c>.
    /// </summary>
    private static void DequantizeQ4_K(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 144;
            float d = DecodeHalf(raw, o);
            float dmin = DecodeHalf(raw, o + 2);
            ReadOnlySpan<byte> scales = raw.Slice(o + 4, 12);
            ReadOnlySpan<byte> qs = raw.Slice(o + 16, 128);
            int outBase = b * QkK;

            int outIdx = outBase;
            int qsBase = 0;
            // Process two 64-element halves; each uses a low and a high nibble plane over 32 qs bytes.
            for (int j = 0; j < QkK / 64; j++)
            {
                GetScaleMinK4(2 * j, scales, out int sc0, out int m0);
                float d0 = d * sc0;
                float dm0 = dmin * m0;
                GetScaleMinK4(2 * j + 1, scales, out int sc1, out int m1);
                float d1 = d * sc1;
                float dm1 = dmin * m1;

                for (int l = 0; l < 32; l++)
                    output[outIdx + l] = d0 * (qs[qsBase + l] & 0x0F) - dm0;
                for (int l = 0; l < 32; l++)
                    output[outIdx + 32 + l] = d1 * (qs[qsBase + l] >> 4) - dm1;

                outIdx += 64;
                qsBase += 32;
            }
        }
    }

    /// <summary>
    /// Q5_K super-block (176 bytes): fp16 <c>d</c>, fp16 <c>dmin</c>, <c>scales[12]</c> (as Q4_K),
    /// <c>qh[32]</c> (1 high bit per code), <c>qs[128]</c> (256 4-bit low codes). Value =
    /// <c>d · sc · (low | high&lt;&lt;4) − dmin · mn</c>, scale/min decoded as in Q4_K.
    /// </summary>
    private static void DequantizeQ5_K(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 176;
            float d = DecodeHalf(raw, o);
            float dmin = DecodeHalf(raw, o + 2);
            ReadOnlySpan<byte> scales = raw.Slice(o + 4, 12);
            ReadOnlySpan<byte> qh = raw.Slice(o + 16, 32);
            ReadOnlySpan<byte> qs = raw.Slice(o + 48, 128);
            int outBase = b * QkK;

            int outIdx = outBase;
            int qsBase = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QkK / 64; j++)
            {
                GetScaleMinK4(2 * j, scales, out int sc0, out int m0);
                float d0 = d * sc0;
                float dm0 = dmin * m0;
                GetScaleMinK4(2 * j + 1, scales, out int sc1, out int m1);
                float d1 = d * sc1;
                float dm1 = dmin * m1;

                for (int l = 0; l < 32; l++)
                {
                    int hi = (qh[l] & u1) != 0 ? 16 : 0;
                    output[outIdx + l] = d0 * ((qs[qsBase + l] & 0x0F) + hi) - dm0;
                }
                for (int l = 0; l < 32; l++)
                {
                    int hi = (qh[l] & u2) != 0 ? 16 : 0;
                    output[outIdx + 32 + l] = d1 * ((qs[qsBase + l] >> 4) + hi) - dm1;
                }

                outIdx += 64;
                qsBase += 32;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    /// <summary>
    /// Q6_K super-block (210 bytes): <c>ql[128]</c> (256 4-bit low codes), <c>qh[64]</c> (256 2-bit
    /// high codes), <c>scales[16]</c> (signed int8), fp16 <c>d</c>. Each code is the signed 6-bit
    /// value <c>(low4 | (high2&lt;&lt;4)) − 32</c>. Value = <c>d · scale · q</c>. The packing walks
    /// the super-block in two 128-element halves; within each half four 16-element scale groups are
    /// read with the reference's interleave.
    /// </summary>
    private static void DequantizeQ6_K(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 210;
            ReadOnlySpan<byte> ql = raw.Slice(o, 128);
            ReadOnlySpan<byte> qh = raw.Slice(o + 128, 64);
            ReadOnlySpan<byte> scales = raw.Slice(o + 192, 16);
            float d = DecodeHalf(raw, o + 208);
            int outBase = b * QkK;

            // Mirrors dequantize_row_q6_K. Two 128-element halves; each half advances ql by 64,
            // qh by 32, and the scale base by 8.
            int qlBase = 0;
            int qhBase = 0;
            int scBase = 0;
            int outIdx = outBase;
            for (int half = 0; half < QkK; half += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int is_ = l / 16;
                    int q1 = ((ql[qlBase + l]      & 0xF) | (((qh[qhBase + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[qlBase + l + 32] & 0xF) | (((qh[qhBase + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[qlBase + l]      >> 4)  | (((qh[qhBase + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[qlBase + l + 32] >> 4)  | (((qh[qhBase + l] >> 6) & 3) << 4)) - 32;
                    output[outIdx + l]      = d * (sbyte)scales[scBase + is_ + 0] * q1;
                    output[outIdx + l + 32] = d * (sbyte)scales[scBase + is_ + 2] * q2;
                    output[outIdx + l + 64] = d * (sbyte)scales[scBase + is_ + 4] * q3;
                    output[outIdx + l + 96] = d * (sbyte)scales[scBase + is_ + 6] * q4;
                }
                qlBase += 64;
                qhBase += 32;
                scBase += 8;
                outIdx += 128;
            }
        }
    }

    /// <summary>
    /// Q8_K super-block (292 bytes): fp32 <c>d</c>, <c>qs[256]</c> (signed int8 codes), then
    /// <c>bsums[16]</c> (int16 partial sums, used only by GEMM and ignored here). Value =
    /// <c>d · q</c>. Q8_K is an intermediate activation-quantization type; included for completeness.
    /// </summary>
    private static void DequantizeQ8_K(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 292;
            float d = BitConverter.UInt32BitsToSingle(ReadUInt32LE(raw, o));
            int outBase = b * QkK;
            for (int j = 0; j < QkK; j++)
                output[outBase + j] = d * unchecked((sbyte)raw[o + 4 + j]);
        }
    }

    // =====================================================================================
    // Non-linear-codebook IQ quants (IQ4_NL, IQ4_XS)
    // =====================================================================================

    /// <summary>
    /// IQ4_NL block (18 bytes, QK=32): fp16 scale <c>d</c> then 16 bytes holding 32 4-bit codes. As in
    /// Q4_0 the codes are split low-half/high-half — code <c>j</c> (j&lt;16) is the low nibble of byte
    /// <c>j</c> and code <c>j+16</c> is the high nibble. Each 4-bit code indexes the non-linear
    /// codebook: value = <c>d · KValuesIq4Nl[code]</c>. Mirrors <c>dequantize_row_iq4_nl</c>.
    /// </summary>
    private static void DequantizeIq4Nl(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        const int qk = 32;
        ReadOnlySpan<sbyte> kv = KValuesIq4Nl;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 18;
            float d = DecodeHalf(raw, o);
            ReadOnlySpan<byte> qs = raw.Slice(o + 2, 16);
            int outBase = b * qk;
            for (int j = 0; j < qk / 2; j++)
            {
                output[outBase + j] = d * kv[qs[j] & 0x0F];
                output[outBase + j + qk / 2] = d * kv[qs[j] >> 4];
            }
        }
    }

    /// <summary>
    /// IQ4_XS super-block (136 bytes, QK_K=256): fp16 <c>d</c>, a 16-bit <c>scales_h</c>, 4 bytes
    /// <c>scales_l</c>, then 128 bytes holding 256 4-bit codes. The 256 elements split into 8 sub-blocks
    /// of 32. Sub-block <c>ib</c> reconstructs a 6-bit scale from <c>scales_l[ib/2]</c> (nibble selected
    /// by <c>ib%2</c>, low 4 bits) and <c>scales_h</c> (2 bits at position <c>2·ib</c>, the high 2 bits);
    /// the signed scale is <c>(ls − 32)</c> and the sub-block delta is <c>dl = d · (ls − 32)</c>. Within
    /// a sub-block the 16 <c>qs</c> bytes split low-half/high-half exactly as IQ4_NL: value =
    /// <c>dl · KValuesIq4Nl[code]</c>. Mirrors <c>dequantize_row_iq4_xs</c>.
    /// </summary>
    private static void DequantizeIq4Xs(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<sbyte> kv = KValuesIq4Nl;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 136;
            float d = DecodeHalf(raw, o);
            int scalesH = raw[o + 2] | (raw[o + 3] << 8);
            ReadOnlySpan<byte> scalesL = raw.Slice(o + 4, 4);
            ReadOnlySpan<byte> qs = raw.Slice(o + 8, 128);

            int outIdx = b * QkK;
            int qsBase = 0;
            for (int ib = 0; ib < QkK / 32; ib++)
            {
                int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scalesH >> (2 * ib)) & 3) << 4);
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++)
                {
                    output[outIdx + j] = dl * kv[qs[qsBase + j] & 0x0F];
                    output[outIdx + j + 16] = dl * kv[qs[qsBase + j] >> 4];
                }
                outIdx += 32;
                qsBase += 16;
            }
        }
    }

    // =====================================================================================
    // Shared helpers
    // =====================================================================================

    /// <summary>
    /// Decodes the <c>j</c>-th 6-bit scale/min pair from the 12-byte <c>scales</c> array shared by
    /// Q4_K and Q5_K (the reference <c>get_scale_min_k4</c>). For <c>j &lt; 4</c> both fields are the
    /// low 6 bits of bytes <c>j</c> and <c>j+4</c>; for <c>j ≥ 4</c> they are reconstructed from the
    /// high 2 bits of bytes <c>j-4</c>/<c>j</c> combined with bytes <c>j+4</c>.
    /// </summary>
    private static void GetScaleMinK4(int j, ReadOnlySpan<byte> scales, out int sc, out int min)
    {
        if (j < 4)
        {
            sc = scales[j] & 63;
            min = scales[j + 4] & 63;
        }
        else
        {
            sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        }
    }

    /// <summary>Decodes a little-endian IEEE-754 half float at <paramref name="offset"/> in
    /// <paramref name="raw"/> to a <see cref="float"/> (mirrors <c>GgufFile.DecodeFloat16</c>).</summary>
    private static float DecodeHalf(ReadOnlySpan<byte> raw, int offset)
    {
        ushort bits = (ushort)(raw[offset] | (raw[offset + 1] << 8));
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }

    /// <summary>Reads a little-endian <see cref="uint"/> at <paramref name="offset"/>.</summary>
    private static uint ReadUInt32LE(ReadOnlySpan<byte> raw, int offset) =>
        (uint)(raw[offset] | (raw[offset + 1] << 8) | (raw[offset + 2] << 16) | (raw[offset + 3] << 24));
}
