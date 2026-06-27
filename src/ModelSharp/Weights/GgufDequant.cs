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
/// The grid-codebook "IQ" families (<see cref="GgmlType.IQ2_XXS"/>, <see cref="GgmlType.IQ2_XS"/>,
/// <see cref="GgmlType.IQ2_S"/>, <see cref="GgmlType.IQ3_XXS"/>, <see cref="GgmlType.IQ3_S"/>,
/// <see cref="GgmlType.IQ1_S"/>, <see cref="GgmlType.IQ1_M"/>) encode each group as an index into a
/// large, fixed lattice/grid constant table (256–2048 entries each) rather than an affine scale. The
/// grids are vendored verbatim from llama.cpp's <c>ggml-common.h</c> in <see cref="GgufIqGrids"/>
/// (MIT — see NOTICE), and each dequant routine below is ported line-by-line from the corresponding
/// <c>dequantize_row_iq*</c> in <c>ggml-quants.c</c> at the pinned upstream commit. Any unrecognized
/// type still throws rather than emitting unverified output.</para>
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
        GgmlType.IQ4_NL or GgmlType.IQ4_XS or
        GgmlType.IQ2_XXS or GgmlType.IQ2_XS or GgmlType.IQ2_S or
        GgmlType.IQ3_XXS or GgmlType.IQ3_S or
        GgmlType.IQ1_S or GgmlType.IQ1_M => true,
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
                "Q8_K, IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M.");

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
            case GgmlType.IQ2_XXS: DequantizeIq2Xxs(raw, output, nBlocks); break;
            case GgmlType.IQ2_XS: DequantizeIq2Xs(raw, output, nBlocks); break;
            case GgmlType.IQ2_S: DequantizeIq2S(raw, output, nBlocks); break;
            case GgmlType.IQ3_XXS: DequantizeIq3Xxs(raw, output, nBlocks); break;
            case GgmlType.IQ3_S: DequantizeIq3S(raw, output, nBlocks); break;
            case GgmlType.IQ1_S: DequantizeIq1S(raw, output, nBlocks); break;
            case GgmlType.IQ1_M: DequantizeIq1M(raw, output, nBlocks); break;
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
    // Grid-codebook IQ families (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M)
    //
    // Each reconstructs a group of weights by indexing a large, fixed lattice "grid" table vendored
    // verbatim in GgufIqGrids (from llama.cpp ggml-common.h, MIT — see NOTICE). The routines below
    // are ported line-by-line from the dequantize_row_iq* reference functions in ggml-quants.c at the
    // pinned commit 050ee92d04c2e1f639025786dea701c70e7d4204. Block layouts (all QK_K = 256, sizes are
    // the GgmlTypeInfo.TypeSize values) match the block_iq* structs in ggml-common.h.
    //
    // Grid entries pack 8 (u64) or 4 (u32) signed int8 lattice codes little-endian. The C reference
    // reinterprets a grid entry as `const uint8_t*` (or `const int8_t*` for the IQ1 grid) and reads
    // byte j; on little-endian that is (entry >> (8*j)) & 0xFF, which is what GridByte/GridSByte do.
    // The IQ2/IQ3 grids hold small non-negative codes (8/0x19/0x2b ...) read as uint8; the IQ1 grid
    // holds signed codes (-1/0/+1 scaled) read as int8.
    // =====================================================================================

    /// <summary>Unpacks the <paramref name="j"/>-th byte (as unsigned 0..255) of a packed 64-bit grid
    /// entry, mirroring the C reinterpret of <c>iq2*_grid[idx]</c> as <c>const uint8_t*</c>.</summary>
    private static int GridByte(ulong entry, int j) => (int)((entry >> (8 * j)) & 0xFF);

    /// <summary>Unpacks the <paramref name="j"/>-th byte (as unsigned 0..255) of a packed 32-bit grid
    /// entry, mirroring the C reinterpret of <c>iq3*_grid[idx]</c> as <c>const uint8_t*</c>.</summary>
    private static int GridByte(uint entry, int j) => (int)((entry >> (8 * j)) & 0xFF);

    /// <summary>Unpacks the <paramref name="j"/>-th byte as <i>signed</i> int8 of a packed 64-bit grid
    /// entry, mirroring the C reinterpret of <c>iq1s_grid[idx]</c> as <c>const int8_t*</c>.</summary>
    private static int GridSByte(ulong entry, int j) => unchecked((sbyte)((entry >> (8 * j)) & 0xFF));

    /// <summary>
    /// IQ2_XXS super-block (66 B): fp16 <c>d</c>; <c>qs[8]</c> uint16 (64 B). For each of the 8
    /// 32-element groups (<c>ib32</c>) the 4 consecutive uint16 form two uint32 <c>aux32[0..1]</c>:
    /// the four bytes of <c>aux32[0]</c> are grid indices into <see cref="GgufIqGrids.Iq2xxsGrid"/>;
    /// <c>aux32[1]</c> packs the block scale in its top nibble and four 7-bit sign selectors. Each
    /// grid point yields 8 codes; signs come from <c>ksigns_iq2xs</c>/<c>kmask_iq2xs</c>. Value =
    /// <c>db · grid[j] · (±1)</c>, <c>db = d · (0.5 + (aux32[1]&gt;&gt;28)) · 0.25</c>.
    /// Ported from <c>dequantize_row_iq2_xxs</c>.
    /// </summary>
    private static void DequantizeIq2Xxs(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<ulong> grid = GgufIqGrids.Iq2xxsGrid;
        ReadOnlySpan<byte> ksigns = GgufIqGrids.KsignsIq2xs;
        ReadOnlySpan<byte> kmask = GgufIqGrids.KmaskIq2xs;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 66;
            float d = DecodeHalf(raw, o);
            int qsBase = o + 2; // qs[QK_K/8] = 32 uint16 (64 bytes); 4 uint16 per ib32.
            int outIdx = b * QkK;
            for (int ib32 = 0; ib32 < QkK / 32; ib32++)
            {
                // 4*ib32 uint16 -> read 2 uint32 (aux32[0], aux32[1]) = 8 bytes.
                int p = qsBase + 8 * ib32;
                uint aux0 = ReadUInt32LE(raw, p);
                uint aux1 = ReadUInt32LE(raw, p + 4);
                float db = d * (0.5f + (aux1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    int gridIdx = (int)((aux0 >> (8 * l)) & 0xFF);
                    ulong g = grid[gridIdx];
                    byte signs = ksigns[(int)((aux1 >> (7 * l)) & 127)];
                    for (int j = 0; j < 8; j++)
                        output[outIdx + j] = db * GridByte(g, j) * ((signs & kmask[j]) != 0 ? -1f : 1f);
                    outIdx += 8;
                }
            }
        }
    }

    /// <summary>
    /// IQ2_XS super-block (74 B): fp16 <c>d</c>; <c>qs[8]</c> uint16 (64 B); <c>scales[8]</c> (8 B).
    /// Per 32-element group <c>ib32</c>: two sub-scales <c>db[0/1] = d · (0.5 + nibble) · 0.25</c> from
    /// <c>scales[ib32]</c>. Each of 4 uint16 <c>qs[4*ib32+l]</c> gives a 9-bit grid index (low 9 bits)
    /// into <see cref="GgufIqGrids.Iq2xsGrid"/> and a 7-bit sign selector (high bits) into
    /// <c>ksigns_iq2xs</c>. Ported from <c>dequantize_row_iq2_xs</c>.
    /// </summary>
    private static void DequantizeIq2Xs(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<ulong> grid = GgufIqGrids.Iq2xsGrid;
        ReadOnlySpan<byte> ksigns = GgufIqGrids.KsignsIq2xs;
        ReadOnlySpan<byte> kmask = GgufIqGrids.KmaskIq2xs;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 74;
            float d = DecodeHalf(raw, o);
            int qsBase = o + 2;          // qs[QK_K/8] = 32 uint16 = 64 bytes
            int scalesBase = o + 2 + 64; // scales[QK_K/32] = 8 bytes
            int outIdx = b * QkK;
            for (int ib32 = 0; ib32 < QkK / 32; ib32++)
            {
                byte sc = raw[scalesBase + ib32];
                float db0 = d * (0.5f + (sc & 0xf)) * 0.25f;
                float db1 = d * (0.5f + (sc >> 4)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    int q = raw[qsBase + 2 * (4 * ib32 + l)] | (raw[qsBase + 2 * (4 * ib32 + l) + 1] << 8);
                    ulong g = grid[q & 511];
                    byte signs = ksigns[(q >> 9) & 127];
                    float dl = (l / 2) == 0 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                        output[outIdx + j] = dl * GridByte(g, j) * ((signs & kmask[j]) != 0 ? -1f : 1f);
                    outIdx += 8;
                }
            }
        }
    }

    /// <summary>
    /// IQ2_S super-block (82 B): fp16 <c>d</c>; <c>qs[64]</c> (the first 32 bytes are grid indices,
    /// the second 32 are sign bytes — <c>signs = qs + QK_K/8</c>); <c>qh[8]</c>; <c>scales[8]</c>.
    /// Per group <c>ib32</c>: two sub-scales from <c>scales[ib32]</c>. The grid index is
    /// <c>qs[l] | ((qh[ib32] &lt;&lt; (8-2l)) &amp; 0x300)</c> into <see cref="GgufIqGrids.Iq2sGrid"/>;
    /// signs are taken directly from the sign bytes (not via ksigns). Ported from
    /// <c>dequantize_row_iq2_s</c>.
    /// </summary>
    private static void DequantizeIq2S(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<ulong> grid = GgufIqGrids.Iq2sGrid;
        ReadOnlySpan<byte> kmask = GgufIqGrids.KmaskIq2xs;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 82;
            float d = DecodeHalf(raw, o);
            int qsBase = o + 2;            // qs[QK_K/4] = 64 bytes
            int signsBase = qsBase + 32;   // signs = qs + QK_K/8 = qs + 32
            int qhBase = o + 2 + 64;       // qh[QK_K/32] = 8 bytes
            int scalesBase = o + 2 + 64 + 8; // scales[QK_K/32] = 8 bytes
            int outIdx = b * QkK;
            // qs/signs advance by 4 per ib32 (mirrors qs += 4; signs += 4).
            int qsPtr = qsBase;
            int signsPtr = signsBase;
            for (int ib32 = 0; ib32 < QkK / 32; ib32++)
            {
                byte sc = raw[scalesBase + ib32];
                float db0 = d * (0.5f + (sc & 0xf)) * 0.25f;
                float db1 = d * (0.5f + (sc >> 4)) * 0.25f;
                byte qh = raw[qhBase + ib32];
                for (int l = 0; l < 4; l++)
                {
                    float dl = (l / 2) == 0 ? db0 : db1;
                    int gridIdx = raw[qsPtr + l] | ((qh << (8 - 2 * l)) & 0x300);
                    ulong g = grid[gridIdx];
                    byte signs = raw[signsPtr + l];
                    for (int j = 0; j < 8; j++)
                        output[outIdx + j] = dl * GridByte(g, j) * ((signs & kmask[j]) != 0 ? -1f : 1f);
                    outIdx += 8;
                }
                qsPtr += 4;
                signsPtr += 4;
            }
        }
    }

    /// <summary>
    /// IQ3_XXS super-block (98 B): fp16 <c>d</c>; <c>qs[96]</c> (3·QK_K/8 grid indices);
    /// <c>scales_and_signs = qs + QK_K/4</c> (the last 32 bytes, one uint32 per group). Per group:
    /// <c>db = d · (0.5 + (aux32&gt;&gt;28)) · 0.5</c>; for each of 4 pairs, two grid indices
    /// <c>qs[2l]</c>,<c>qs[2l+1]</c> select 4-byte points in <see cref="GgufIqGrids.Iq3xxsGrid"/>; signs
    /// from <c>ksigns_iq2xs[(aux32&gt;&gt;7l)&amp;127]</c>. Ported from <c>dequantize_row_iq3_xxs</c>.
    /// </summary>
    private static void DequantizeIq3Xxs(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<uint> grid = GgufIqGrids.Iq3xxsGrid;
        ReadOnlySpan<byte> ksigns = GgufIqGrids.KsignsIq2xs;
        ReadOnlySpan<byte> kmask = GgufIqGrids.KmaskIq2xs;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 98;
            float d = DecodeHalf(raw, o);
            int qsBase = o + 2;             // qs[3*QK_K/8] = 96 bytes
            int ssBase = qsBase + QkK / 4;  // scales_and_signs = qs + 64
            int outIdx = b * QkK;
            int qsPtr = qsBase;             // advances by 8 per ib32
            for (int ib32 = 0; ib32 < QkK / 32; ib32++)
            {
                uint aux32 = ReadUInt32LE(raw, ssBase + 4 * ib32);
                float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++)
                {
                    byte signs = ksigns[(int)((aux32 >> (7 * l)) & 127)];
                    uint g1 = grid[raw[qsPtr + 2 * l + 0]];
                    uint g2 = grid[raw[qsPtr + 2 * l + 1]];
                    for (int j = 0; j < 4; j++)
                    {
                        output[outIdx + j + 0] = db * GridByte(g1, j) * ((signs & kmask[j + 0]) != 0 ? -1f : 1f);
                        output[outIdx + j + 4] = db * GridByte(g2, j) * ((signs & kmask[j + 4]) != 0 ? -1f : 1f);
                    }
                    outIdx += 8;
                }
                qsPtr += 8;
            }
        }
    }

    /// <summary>
    /// IQ3_S super-block (110 B): fp16 <c>d</c>; <c>qs[64]</c>; <c>qh[8]</c>; <c>signs[32]</c>;
    /// <c>scales[4]</c>. Processed two groups at a time (<c>ib32 += 2</c>): each scale byte gives
    /// <c>db1 = d·(1+2·low4)</c>, <c>db2 = d·(1+2·high4)</c>. The grid index is
    /// <c>qs[2l] | ((qh[h] &lt;&lt; (8-2l)) &amp; 256)</c> (and <c>7-2l</c> for the odd member) into
    /// <see cref="GgufIqGrids.Iq3sGrid"/>; signs taken directly from <c>signs[l]</c>. Ported from
    /// <c>dequantize_row_iq3_s</c>.
    /// </summary>
    private static void DequantizeIq3S(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<uint> grid = GgufIqGrids.Iq3sGrid;
        ReadOnlySpan<byte> kmask = GgufIqGrids.KmaskIq2xs;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 110;
            float d = DecodeHalf(raw, o);
            int qsBase = o + 2;                 // qs[QK_K/4] = 64
            int qhBase = qsBase + 64;           // qh[QK_K/32] = 8
            int signsBase = qhBase + 8;         // signs[QK_K/8] = 32
            int scalesBase = signsBase + 32;    // scales[IQ3S_N_SCALE=4]
            int outIdx = b * QkK;
            int qsPtr = qsBase;
            int qhPtr = qhBase;
            int signsPtr = signsBase;
            for (int ib32 = 0; ib32 < QkK / 32; ib32 += 2)
            {
                byte sc = raw[scalesBase + ib32 / 2];
                float db1 = d * (1 + 2 * (sc & 0xf));
                float db2 = d * (1 + 2 * (sc >> 4));
                byte qh0 = raw[qhPtr + 0];
                byte qh1 = raw[qhPtr + 1];
                // First 4 groups (db1, qh[0]).
                for (int l = 0; l < 4; l++)
                {
                    uint g1 = grid[raw[qsPtr + 2 * l + 0] | ((qh0 << (8 - 2 * l)) & 256)];
                    uint g2 = grid[raw[qsPtr + 2 * l + 1] | ((qh0 << (7 - 2 * l)) & 256)];
                    byte signs = raw[signsPtr + l];
                    for (int j = 0; j < 4; j++)
                    {
                        output[outIdx + j + 0] = db1 * GridByte(g1, j) * ((signs & kmask[j + 0]) != 0 ? -1f : 1f);
                        output[outIdx + j + 4] = db1 * GridByte(g2, j) * ((signs & kmask[j + 4]) != 0 ? -1f : 1f);
                    }
                    outIdx += 8;
                }
                qsPtr += 8;
                signsPtr += 4;
                // Second 4 groups (db2, qh[1]).
                for (int l = 0; l < 4; l++)
                {
                    uint g1 = grid[raw[qsPtr + 2 * l + 0] | ((qh1 << (8 - 2 * l)) & 256)];
                    uint g2 = grid[raw[qsPtr + 2 * l + 1] | ((qh1 << (7 - 2 * l)) & 256)];
                    byte signs = raw[signsPtr + l];
                    for (int j = 0; j < 4; j++)
                    {
                        output[outIdx + j + 0] = db2 * GridByte(g1, j) * ((signs & kmask[j + 0]) != 0 ? -1f : 1f);
                        output[outIdx + j + 4] = db2 * GridByte(g2, j) * ((signs & kmask[j + 4]) != 0 ? -1f : 1f);
                    }
                    outIdx += 8;
                }
                qhPtr += 2;
                qsPtr += 8;
                signsPtr += 4;
            }
        }
    }

    /// <summary>IQ1 delta constant (<c>IQ1S_DELTA</c>/<c>IQ1M_DELTA</c> in ggml-common.h).</summary>
    private const float Iq1Delta = 0.125f;

    /// <summary>
    /// IQ1_S super-block (50 B): fp16 <c>d</c>; <c>qs[32]</c>; <c>qh[8]</c> uint16. Per 32-element
    /// group <c>ib</c>: <c>dl = d · (2·((qh[ib]&gt;&gt;12)&amp;7) + 1)</c>; sign bit 0x8000 of
    /// <c>qh[ib]</c> selects <c>±IQ1S_DELTA</c>. Grid index is
    /// <c>qs[l] | (((qh[ib]&gt;&gt;3l)&amp;7) &lt;&lt; 8)</c> into the <i>signed</i>
    /// <see cref="GgufIqGrids.Iq1sGrid"/>; value = <c>dl · (grid[j] + delta)</c>. Ported from
    /// <c>dequantize_row_iq1_s</c>.
    /// </summary>
    private static void DequantizeIq1S(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<ulong> grid = GgufIqGrids.Iq1sGrid;
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 50;
            float d = DecodeHalf(raw, o);
            int qsBase = o + 2;        // qs[QK_K/8] = 32
            int qhBase = qsBase + 32;  // qh[QK_K/32] = 8 uint16 = 16 bytes
            int outIdx = b * QkK;
            int qsPtr = qsBase;
            for (int ib = 0; ib < QkK / 32; ib++)
            {
                int qh = raw[qhBase + 2 * ib] | (raw[qhBase + 2 * ib + 1] << 8);
                float dl = d * (2 * ((qh >> 12) & 7) + 1);
                float delta = (qh & 0x8000) != 0 ? -Iq1Delta : Iq1Delta;
                for (int l = 0; l < 4; l++)
                {
                    int gridIdx = raw[qsPtr + l] | (((qh >> (3 * l)) & 7) << 8);
                    ulong g = grid[gridIdx];
                    for (int j = 0; j < 8; j++)
                        output[outIdx + j] = dl * (GridSByte(g, j) + delta);
                    outIdx += 8;
                }
                qsPtr += 4;
            }
        }
    }

    /// <summary>
    /// IQ1_M super-block (56 B): <c>qs[32]</c>; <c>qh[16]</c>; <c>scales[8]</c> (4 uint16). The fp16
    /// scale is reassembled from the top nibbles of the 4 scale uint16:
    /// <c>(sc0&gt;&gt;12) | ((sc1&gt;&gt;8)&amp;0xf0) | ((sc2&gt;&gt;4)&amp;0xf00) | (sc3&amp;0xf000)</c>.
    /// Per 32-element group <c>ib</c>: two 3-bit sub-scales from <c>sc[ib/2]</c>. Four 8-element grid
    /// points use indices <c>qs[k] | ((qh &lt;&lt; n) &amp; 0x700)</c> and per-point deltas from the
    /// 0x08/0x80 bits of the two <c>qh</c> bytes. Grid is the signed <see cref="GgufIqGrids.Iq1sGrid"/>.
    /// Ported from <c>dequantize_row_iq1_m</c>.
    /// </summary>
    private static void DequantizeIq1M(ReadOnlySpan<byte> raw, Span<float> output, int nBlocks)
    {
        ReadOnlySpan<ulong> grid = GgufIqGrids.Iq1sGrid;
        Span<ushort> idx = stackalloc ushort[4];
        Span<float> delta = stackalloc float[4];
        Span<int> sc = stackalloc int[4];
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 56;
            int qsBase = o;              // qs[QK_K/8] = 32
            int qhBase = o + 32;         // qh[QK_K/16] = 16
            int scalesBase = o + 32 + 16; // scales[QK_K/32] = 8 bytes = 4 uint16
            // scales as uint16[4].
            int sc0 = raw[scalesBase + 0] | (raw[scalesBase + 1] << 8);
            int sc1 = raw[scalesBase + 2] | (raw[scalesBase + 3] << 8);
            int sc2 = raw[scalesBase + 4] | (raw[scalesBase + 5] << 8);
            int sc3 = raw[scalesBase + 6] | (raw[scalesBase + 7] << 8);
            ushort scaleBits = (ushort)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
            float d = (float)BitConverter.UInt16BitsToHalf(scaleBits);
            sc[0] = sc0; sc[1] = sc1; sc[2] = sc2; sc[3] = sc3;

            int outIdx = b * QkK;
            int qsPtr = qsBase;
            int qhPtr = qhBase;
            for (int ib = 0; ib < QkK / 32; ib++)
            {
                float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
                float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
                byte qh0 = raw[qhPtr + 0];
                byte qh1 = raw[qhPtr + 1];
                idx[0] = (ushort)(raw[qsPtr + 0] | ((qh0 << 8) & 0x700));
                idx[1] = (ushort)(raw[qsPtr + 1] | ((qh0 << 4) & 0x700));
                idx[2] = (ushort)(raw[qsPtr + 2] | ((qh1 << 8) & 0x700));
                idx[3] = (ushort)(raw[qsPtr + 3] | ((qh1 << 4) & 0x700));
                delta[0] = (qh0 & 0x08) != 0 ? -Iq1Delta : Iq1Delta;
                delta[1] = (qh0 & 0x80) != 0 ? -Iq1Delta : Iq1Delta;
                delta[2] = (qh1 & 0x08) != 0 ? -Iq1Delta : Iq1Delta;
                delta[3] = (qh1 & 0x80) != 0 ? -Iq1Delta : Iq1Delta;
                for (int l = 0; l < 2; l++)
                {
                    ulong g = grid[idx[l]];
                    for (int j = 0; j < 8; j++)
                        output[outIdx + j] = dl1 * (GridSByte(g, j) + delta[l]);
                    outIdx += 8;
                }
                for (int l = 2; l < 4; l++)
                {
                    ulong g = grid[idx[l]];
                    for (int j = 0; j < 8; j++)
                        output[outIdx + j] = dl2 * (GridSByte(g, j) + delta[l]);
                    outIdx += 8;
                }
                qsPtr += 4;
                qhPtr += 2;
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
