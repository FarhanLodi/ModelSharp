using System;
using System.Collections.Generic;
using ModelSharp;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Unit tests for the non-linear-codebook importance-matrix quants in <see cref="GgufDequant"/>
/// (IQ4_NL and IQ4_XS). Each test hand-builds synthetic ggml blocks with known nibbles and scale
/// fields and asserts the reconstructed floats against values computed by hand from the documented
/// block layout and the exact 16-entry codebook. Half-float scales are chosen to be exactly
/// representable; comparisons use a small tolerance only to absorb float rounding.
/// </summary>
public class GgufIqDequantTests
{
    private const float Tol = 1e-4f;

    // The exact reference codebook (mirrors kvalues_iq4nl in ggml-quants.c).
    private static readonly sbyte[] Kv =
    {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };

    private static void Half(List<byte> dst, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        dst.Add((byte)(bits & 0xFF));
        dst.Add((byte)(bits >> 8));
    }

    private static byte Nibbles(int low, int high) => (byte)((low & 0xF) | ((high & 0xF) << 4));

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= Tol,
                $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    [Fact]
    public void Codebook_MatchesReference()
    {
        ReadOnlySpan<sbyte> exposed = GgufDequant.KValuesIq4Nl;
        Assert.Equal(16, exposed.Length);
        for (int i = 0; i < 16; i++)
            Assert.Equal(Kv[i], exposed[i]);
    }

    [Fact]
    public void Iq4Nl_DequantizesExactly()
    {
        // d = 0.5. 16 bytes pack 32 4-bit codes: byte j low nibble -> code j, high nibble -> code j+16.
        // value = d * kvalues_iq4nl[code].
        var raw = new List<byte>();
        const float d = 0.5f;
        Half(raw, d);

        var lowNibble = new int[16];
        var highNibble = new int[16];
        for (int j = 0; j < 16; j++)
        {
            lowNibble[j] = j;          // 0..15 — exercises every codebook entry
            highNibble[j] = 15 - j;    // 15..0
            raw.Add(Nibbles(lowNibble[j], highNibble[j]));
        }

        var expected = new float[32];
        for (int j = 0; j < 16; j++)
        {
            expected[j] = d * Kv[lowNibble[j]];
            expected[j + 16] = d * Kv[highNibble[j]];
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ4_NL, 32);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq4Nl_TwoBlocks_AreContiguous()
    {
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
                expected[b * 32 + j] = ds[b] * Kv[low];
                expected[b * 32 + j + 16] = ds[b] * Kv[high];
            }
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ4_NL, 64);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq4Xs_DequantizesExactly()
    {
        // Super-block layout (136 bytes): fp16 d, uint16 scales_h, scales_l[4], qs[128].
        // 8 sub-blocks of 32. For sub-block ib:
        //   ls = (scales_l[ib/2] >> 4*(ib%2) & 0xF) | (((scales_h >> 2*ib) & 3) << 4)
        //   dl = d * (ls - 32)
        //   y[j]    = dl * kv[qs[j] & 0xF]   (j in 0..15)
        //   y[j+16] = dl * kv[qs[j] >> 4]
        var raw = new List<byte>();
        const float d = 0.25f;
        Half(raw, d);

        // Choose a low nibble (scales_l) and high 2 bits (scales_h) per sub-block so reconstructed
        // 6-bit scales span a range including values below and above 32 (the bias).
        // ib:        0   1   2   3   4   5   6   7
        // low4:      0   1   2   3   4   5   6   7
        // high2:     0   1   2   3   0   1   2   3
        var low4 = new int[8] { 0, 1, 2, 3, 4, 5, 6, 7 };
        var high2 = new int[8] { 0, 1, 2, 3, 0, 1, 2, 3 };

        // Pack scales_l: byte (ib/2) holds sub-block (2k) in its low nibble, (2k+1) in its high nibble.
        var scalesL = new byte[4];
        for (int ib = 0; ib < 8; ib++)
            scalesL[ib / 2] |= (byte)((low4[ib] & 0xF) << (4 * (ib % 2)));

        // Pack scales_h: 2 bits per sub-block at position 2*ib.
        int scalesH = 0;
        for (int ib = 0; ib < 8; ib++)
            scalesH |= (high2[ib] & 3) << (2 * ib);

        raw.Add((byte)(scalesH & 0xFF));
        raw.Add((byte)((scalesH >> 8) & 0xFF));
        foreach (byte sl in scalesL) raw.Add(sl);

        // qs[128]: 16 bytes per sub-block. Vary nibbles so every codebook entry is hit.
        var qsLow = new int[8, 16];
        var qsHigh = new int[8, 16];
        for (int ib = 0; ib < 8; ib++)
        {
            for (int j = 0; j < 16; j++)
            {
                qsLow[ib, j] = (j + ib) & 0xF;
                qsHigh[ib, j] = (15 - j) & 0xF;
                raw.Add(Nibbles(qsLow[ib, j], qsHigh[ib, j]));
            }
        }

        var expected = new float[256];
        for (int ib = 0; ib < 8; ib++)
        {
            int ls = (low4[ib] & 0xF) | ((high2[ib] & 3) << 4);
            float dl = d * (ls - 32);
            int outBase = ib * 32;
            for (int j = 0; j < 16; j++)
            {
                expected[outBase + j] = dl * Kv[qsLow[ib, j]];
                expected[outBase + j + 16] = dl * Kv[qsHigh[ib, j]];
            }
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ4_XS, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void IsSupported_CoversIq4Family()
    {
        Assert.True(GgufDequant.IsSupported(GgmlType.IQ4_NL));
        Assert.True(GgufDequant.IsSupported(GgmlType.IQ4_XS));
    }

    [Fact]
    public void GridCodebookIqFamilies_NowSupported()
    {
        // The grid-codebook families are now backed by the verbatim-vendored ggml grids
        // (GgufIqGrids), so they dequantize rather than throw. A full super-block of zero bytes is a
        // valid input (grid index 0, sign 0, scale 0) and must produce QK_K=256 finite floats.
        foreach (GgmlType t in new[]
        {
            GgmlType.IQ2_XXS, GgmlType.IQ2_XS, GgmlType.IQ2_S,
            GgmlType.IQ3_XXS, GgmlType.IQ3_S, GgmlType.IQ1_S, GgmlType.IQ1_M,
        })
        {
            Assert.True(GgufDequant.IsSupported(t), $"{t} should now be supported");
            int typeSize = t switch
            {
                GgmlType.IQ2_XXS => 66,
                GgmlType.IQ2_XS => 74,
                GgmlType.IQ2_S => 82,
                GgmlType.IQ3_XXS => 98,
                GgmlType.IQ3_S => 110,
                GgmlType.IQ1_S => 50,
                GgmlType.IQ1_M => 56,
                _ => 0,
            };
            float[] got = GgufDequant.Dequantize(new byte[typeSize], t, 256);
            Assert.Equal(256, got.Length);
            foreach (float f in got) Assert.True(float.IsFinite(f));
        }
    }
}
