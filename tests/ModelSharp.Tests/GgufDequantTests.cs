using System;
using System.Collections.Generic;
using ModelSharp;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Unit tests for <see cref="GgufDequant"/>. Each test hand-builds one or more synthetic ggml blocks
/// with known scale/quant values and asserts the reconstructed floats against values computed by hand
/// from the documented block layout. Half-float scales are chosen to be exactly representable so the
/// expected products are exact; comparisons use a small tolerance only to absorb float rounding.
/// </summary>
public class GgufDequantTests
{
    private const float Tol = 1e-4f;

    // ---- block-building helpers --------------------------------------------------------

    /// <summary>Appends a little-endian IEEE-754 half float for <paramref name="value"/>.</summary>
    private static void Half(List<byte> dst, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        dst.Add((byte)(bits & 0xFF));
        dst.Add((byte)(bits >> 8));
    }

    private static void U32(List<byte> dst, uint value)
    {
        dst.Add((byte)(value & 0xFF));
        dst.Add((byte)((value >> 8) & 0xFF));
        dst.Add((byte)((value >> 16) & 0xFF));
        dst.Add((byte)((value >> 24) & 0xFF));
    }

    private static void F32(List<byte> dst, float value) => U32(dst, BitConverter.SingleToUInt32Bits(value));

    private static byte Nibbles(int low, int high) => (byte)((low & 0xF) | ((high & 0xF) << 4));

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= Tol,
                $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    // =====================================================================================
    // Legacy QK = 32 quants
    // =====================================================================================

    [Fact]
    public void Q8_0_DequantizesExactly()
    {
        // d = 0.5; codes are -3..28 (32 distinct int8 values).
        var raw = new List<byte>();
        const float d = 0.5f;
        Half(raw, d);
        var expected = new float[32];
        for (int j = 0; j < 32; j++)
        {
            sbyte q = (sbyte)(j - 3);
            raw.Add(unchecked((byte)q));
            expected[j] = d * q;
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q8_0, 32);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q8_1_DequantizesExactly_IgnoringSum()
    {
        // d = 0.25, s = 99 (a bogus sum that must be ignored), then 32 int8 codes.
        var raw = new List<byte>();
        const float d = 0.25f;
        Half(raw, d);
        Half(raw, 99f); // s, ignored by dequant
        var expected = new float[32];
        for (int j = 0; j < 32; j++)
        {
            sbyte q = (sbyte)(j - 16);
            raw.Add(unchecked((byte)q));
            expected[j] = d * q;
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q8_1, 32);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q4_0_DequantizesExactly()
    {
        // d = 0.5. 16 bytes pack 32 nibbles: byte j low nibble -> code j, high nibble -> code j+16.
        // value = d * (nibble - 8).
        var raw = new List<byte>();
        const float d = 0.5f;
        Half(raw, d);

        var lowNibble = new int[16];
        var highNibble = new int[16];
        for (int j = 0; j < 16; j++)
        {
            lowNibble[j] = j;            // 0..15
            highNibble[j] = 15 - j;      // 15..0
            raw.Add(Nibbles(lowNibble[j], highNibble[j]));
        }

        var expected = new float[32];
        for (int j = 0; j < 16; j++)
        {
            expected[j] = d * (lowNibble[j] - 8);
            expected[j + 16] = d * (highNibble[j] - 8);
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q4_0, 32);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q4_1_DequantizesExactly()
    {
        // d = 0.5, m = -1.0. value = d*nibble + m.
        var raw = new List<byte>();
        const float d = 0.5f;
        const float m = -1.0f;
        Half(raw, d);
        Half(raw, m);

        var lowNibble = new int[16];
        var highNibble = new int[16];
        for (int j = 0; j < 16; j++)
        {
            lowNibble[j] = (j * 2) & 0xF;
            highNibble[j] = (j + 1) & 0xF;
            raw.Add(Nibbles(lowNibble[j], highNibble[j]));
        }

        var expected = new float[32];
        for (int j = 0; j < 16; j++)
        {
            expected[j] = d * lowNibble[j] + m;
            expected[j + 16] = d * highNibble[j] + m;
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q4_1, 32);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q5_0_DequantizesExactly()
    {
        // d = 0.25. qh holds the 5th bit per code (bit j -> code j, bit j+16 -> code j+16).
        // value = d * ((low | high<<4) - 16).
        var raw = new List<byte>();
        const float d = 0.25f;
        Half(raw, d);

        var lowNibble = new int[16];
        var highNibble = new int[16];
        var highBit0 = new int[16]; // for codes 0..15
        var highBit1 = new int[16]; // for codes 16..31
        uint qh = 0;
        for (int j = 0; j < 16; j++)
        {
            lowNibble[j] = j;
            highNibble[j] = 15 - j;
            highBit0[j] = j % 2;          // alternate
            highBit1[j] = (j + 1) % 2;
            if (highBit0[j] != 0) qh |= 1u << j;
            if (highBit1[j] != 0) qh |= 1u << (j + 16);
        }
        U32(raw, qh);
        for (int j = 0; j < 16; j++)
            raw.Add(Nibbles(lowNibble[j], highNibble[j]));

        var expected = new float[32];
        for (int j = 0; j < 16; j++)
        {
            expected[j] = d * ((lowNibble[j] | (highBit0[j] << 4)) - 16);
            expected[j + 16] = d * ((highNibble[j] | (highBit1[j] << 4)) - 16);
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q5_0, 32);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q5_1_DequantizesExactly()
    {
        // d = 0.5, m = 2.0. value = d * (low | high<<4) + m.
        var raw = new List<byte>();
        const float d = 0.5f;
        const float m = 2.0f;
        Half(raw, d);
        Half(raw, m);

        var lowNibble = new int[16];
        var highNibble = new int[16];
        var highBit0 = new int[16];
        var highBit1 = new int[16];
        uint qh = 0;
        for (int j = 0; j < 16; j++)
        {
            lowNibble[j] = (j + 3) & 0xF;
            highNibble[j] = (2 * j) & 0xF;
            highBit0[j] = (j / 4) % 2;
            highBit1[j] = j % 2;
            if (highBit0[j] != 0) qh |= 1u << j;
            if (highBit1[j] != 0) qh |= 1u << (j + 16);
        }
        U32(raw, qh);
        for (int j = 0; j < 16; j++)
            raw.Add(Nibbles(lowNibble[j], highNibble[j]));

        var expected = new float[32];
        for (int j = 0; j < 16; j++)
        {
            expected[j] = d * (lowNibble[j] | (highBit0[j] << 4)) + m;
            expected[j + 16] = d * (highNibble[j] | (highBit1[j] << 4)) + m;
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q5_1, 32);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q4_0_TwoBlocks_AreContiguous()
    {
        // Two independent blocks with different scales; checks block striding.
        var raw = new List<byte>();
        float[] ds = { 0.5f, 0.25f };
        var expected = new float[64];
        for (int b = 0; b < 2; b++)
        {
            Half(raw, ds[b]);
            for (int j = 0; j < 16; j++)
            {
                int low = j;
                int high = 15 - j;
                raw.Add(Nibbles(low, high));
                expected[b * 32 + j] = ds[b] * (low - 8);
                expected[b * 32 + j + 16] = ds[b] * (high - 8);
            }
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q4_0, 64);
        AssertClose(expected, got);
    }

    // =====================================================================================
    // k-quant super-blocks (QK_K = 256)
    // =====================================================================================

    [Fact]
    public void Q8_K_DequantizesExactly()
    {
        // fp32 d, then 256 int8 codes, then 16 int16 bsums (ignored).
        var raw = new List<byte>();
        const float d = 0.125f;
        F32(raw, d);
        var expected = new float[256];
        for (int j = 0; j < 256; j++)
        {
            sbyte q = (sbyte)((j % 251) - 120);
            raw.Add(unchecked((byte)q));
            expected[j] = d * q;
        }
        for (int j = 0; j < 16; j++) { raw.Add(0); raw.Add(0); } // bsums

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q8_K, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q2_K_AllZeroQuants_GivesNegativeMinTerms()
    {
        // With all qs codes = 0, value = -dmin * (scales[is] >> 4) per 16-element group.
        // scales[i] low nibble = scale, high nibble = min.
        var raw = new List<byte>();
        const float d = 1.0f;
        const float dmin = 0.5f;

        var mins = new int[16];
        for (int i = 0; i < 16; i++)
        {
            int scale = (i + 1) & 0xF;
            int min = (i + 2) & 0xF;
            mins[i] = min;
            raw.Add((byte)((scale & 0xF) | ((min & 0xF) << 4)));
        }
        for (int i = 0; i < 64; i++) raw.Add(0);   // qs all zero
        Half(raw, d);
        Half(raw, dmin);

        // Group order matches the reference walk: for each 128-half, 4 shifts, each shift two
        // 16-element groups (is increments 0..15 in order).
        var expected = new float[256];
        for (int g = 0; g < 16; g++)
            for (int l = 0; l < 16; l++)
                expected[g * 16 + l] = -dmin * mins[g];

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q2_K, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Q3_K_ZeroScales_AndSetHighMask_DequantizesExactly()
    {
        // hmask[32]=0xFF, qs[64], scales[12]=0, fp16 d. With all scale bytes 0, every unpacked
        // scale is 0 -> (scale - 32) = -32. With every hmask bit set, q = (qs>>shift)&3 (no -4).
        // So value = d * (-32) * ((qs>>shift)&3).
        var raw = new List<byte>();
        const float d = 0.03125f; // 1/32, exactly representable in half
        for (int i = 0; i < 32; i++) raw.Add(0xFF);     // hmask: all high bits set

        // qs[64]: only the low 2 bits of each byte are exercised at shift 0; set them to a pattern.
        // The first 32 elements (n=0 half, shift=0, j=0) read qs[0..15] (is=0) and qs[16..31] (is=1).
        var qs = new byte[64];
        for (int i = 0; i < 64; i++) qs[i] = (byte)(i % 4); // low 2 bits cycle 0,1,2,3
        foreach (byte q in qs) raw.Add(q);

        for (int i = 0; i < 12; i++) raw.Add(0);          // scales all zero
        Half(raw, d);

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q3_K, 256);

        // First 16 elements: shift 0, qs[0..15], q = qs&3 = i%4, scale -32.
        for (int l = 0; l < 16; l++)
        {
            float exp = d * -32 * (qs[l] & 3);
            Assert.True(MathF.Abs(got[l] - exp) <= Tol,
                $"q3_k index {l}: expected {exp}, got {got[l]}");
        }
        // Next 16 elements: shift 0, qs[16..31].
        for (int l = 0; l < 16; l++)
        {
            float exp = d * -32 * (qs[16 + l] & 3);
            Assert.True(MathF.Abs(got[16 + l] - exp) <= Tol,
                $"q3_k index {16 + l}: expected {exp}, got {got[16 + l]}");
        }
    }

    [Fact]
    public void Q5_K_HighBitsAddSixteen_DequantizesExactly()
    {
        // d, dmin, scales[12], qh[32], qs[128]. Set scales[0]=8 (sc0), scales[4]=0 (m0).
        // qh bit 0 (u1) set for all l -> first 32 outputs add 16 to the low nibble.
        var raw = new List<byte>();
        const float d = 0.5f;
        const float dmin = 1.0f;
        Half(raw, d);
        Half(raw, dmin);

        var scales = new byte[12];
        scales[0] = 8; // sc0 = 8
        scales[4] = 0; // m0 = 0
        foreach (byte s in scales) raw.Add(s);

        var qh = new byte[32];
        for (int i = 0; i < 32; i++) qh[i] = 0x01; // u1 bit set, u2 (bit1) clear
        foreach (byte v in qh) raw.Add(v);

        var qs = new byte[128];
        var low = new int[32];
        for (int l = 0; l < 32; l++)
        {
            low[l] = l & 0xF;
            qs[l] = (byte)(low[l] & 0xF); // high nibble 0
        }
        foreach (byte v in qs) raw.Add(v);

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q5_K, 256);

        // First 32 outputs: d * sc0 * ((qs&0xF) + 16) - dmin*0.
        for (int l = 0; l < 32; l++)
        {
            float exp = d * 8 * (low[l] + 16);
            Assert.True(MathF.Abs(got[l] - exp) <= Tol,
                $"q5_k index {l}: expected {exp}, got {got[l]}");
        }
    }

    [Fact]
    public void Q4_K_LowScaleGroup_DequantizesExactly()
    {
        // Build a Q4_K super-block where only the first 6-bit scale (is=0) is non-trivial and all
        // mins are zero, so the first 32 outputs = d * sc0 * (qs low nibble).
        var raw = new List<byte>();
        const float d = 0.5f;
        const float dmin = 1.0f;
        Half(raw, d);
        Half(raw, dmin);

        // 12 scale bytes. get_scale_min_k4(0) reads scales[0]&63 (scale) and scales[4]&63 (min).
        // Set scales[0]=10 (sc0=10), scales[4]=0 (min0=0); leave the rest 0 -> all other scales 0.
        var scales = new byte[12];
        scales[0] = 10; // sc0 = 10
        scales[4] = 0;  // m0 = 0
        // scales[1] -> sc1 = 0, scales[5] -> m1 = 0
        foreach (byte s in scales) raw.Add(s);

        // 128 qs bytes. First 32 bytes' low nibbles map to outputs 0..31 (group 0).
        var qs = new byte[128];
        var lowNibbles = new int[32];
        for (int l = 0; l < 32; l++)
        {
            lowNibbles[l] = l & 0xF;
            qs[l] = (byte)(lowNibbles[l] & 0xF); // high nibble 0 -> group-1 outputs are 0
        }
        foreach (byte q in qs) raw.Add(q);

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q4_K, 256);

        // First 32 outputs: d * 10 * lowNibble - 1.0 * 0.
        for (int l = 0; l < 32; l++)
            Assert.True(MathF.Abs(got[l] - d * 10 * lowNibbles[l]) <= Tol,
                $"q4_k index {l}: expected {d * 10 * lowNibbles[l]}, got {got[l]}");
        // Outputs 32..63 belong to is=1 (sc1=0, m1=0) -> all zero.
        for (int l = 32; l < 64; l++)
            Assert.True(MathF.Abs(got[l]) <= Tol, $"q4_k index {l}: expected 0, got {got[l]}");
    }

    [Fact]
    public void Q6_K_FirstGroup_DequantizesExactly()
    {
        // Build a Q6_K super-block: ql[128], qh[64], scales[16] int8, fp16 d.
        // value = d * sc[is] * ((low4 | (high2<<4)) - 32). Use scales[0]=1 and check q1 for l<16.
        var raw = new List<byte>();
        const float d = 0.25f;

        var ql = new byte[128];
        var qh = new byte[64];
        // For l in 0..31, q1 = (ql[l]&0xF | ((qh[l]&3)<<4)) - 32. Choose low/high to get distinct codes.
        var low4 = new int[32];
        var high2 = new int[32];
        for (int l = 0; l < 32; l++)
        {
            low4[l] = l & 0xF;
            high2[l] = l % 4;
            ql[l] = (byte)(low4[l] & 0xF);   // high nibble 0 so q3 = -32 region not under test
            qh[l] = (byte)(high2[l] & 3);     // bits 0-1 used by q1; others 0
        }

        foreach (byte v in ql) raw.Add(v);
        foreach (byte v in qh) raw.Add(v);
        var scales = new sbyte[16];
        scales[0] = 1; // is=0 group (l 0..15) uses sc[0]
        scales[1] = 2; // is=1 group (l 16..31) uses sc[1]
        foreach (sbyte s in scales) raw.Add(unchecked((byte)s));
        Half(raw, d);

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.Q6_K, 256);

        // q1 outputs land at y[l + 0] for l in 0..31; scale is sc[is] with is = l/16.
        for (int l = 0; l < 32; l++)
        {
            int code = (low4[l] | (high2[l] << 4)) - 32;
            int sc = l < 16 ? scales[0] : scales[1];
            float exp = d * sc * code;
            Assert.True(MathF.Abs(got[l] - exp) <= Tol,
                $"q6_k index {l}: expected {exp}, got {got[l]}");
        }
    }

    // =====================================================================================
    // Error handling
    // =====================================================================================

    [Fact]
    public void UnsupportedType_Throws()
    {
        // F16 is an unquantized scalar type, not this class's responsibility to dequantize.
        Assert.False(GgufDequant.IsSupported(GgmlType.F16));
        Assert.Throws<ModelSharpException>(
            () => GgufDequant.Dequantize(new byte[64], GgmlType.F16, 32));
    }

    [Fact]
    public void NonMultipleOfBlockSize_Throws()
    {
        Assert.Throws<ModelSharpException>(
            () => GgufDequant.Dequantize(new byte[34], GgmlType.Q8_0, 31));
    }

    [Fact]
    public void TruncatedBuffer_Throws()
    {
        // Two Q8_0 blocks declared (64 elements) but only one block of bytes supplied.
        Assert.Throws<ModelSharpException>(
            () => GgufDequant.Dequantize(new byte[34], GgmlType.Q8_0, 64));
    }

    [Fact]
    public void IsSupported_CoversImplementedTypes()
    {
        foreach (GgmlType t in new[]
        {
            GgmlType.Q4_0, GgmlType.Q4_1, GgmlType.Q5_0, GgmlType.Q5_1, GgmlType.Q8_0, GgmlType.Q8_1,
            GgmlType.Q2_K, GgmlType.Q3_K, GgmlType.Q4_K, GgmlType.Q5_K, GgmlType.Q6_K, GgmlType.Q8_K,
        })
            Assert.True(GgufDequant.IsSupported(t), $"{t} should be supported");
    }
}
