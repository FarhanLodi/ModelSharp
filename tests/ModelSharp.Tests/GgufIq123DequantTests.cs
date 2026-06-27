using System;
using System.Collections.Generic;
using ModelSharp;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// By-construction correctness tests for the grid-codebook IQ families (IQ2_XXS, IQ2_XS, IQ2_S,
/// IQ3_XXS, IQ3_S, IQ1_S, IQ1_M). Each test hand-builds a synthetic ggml super-block with chosen
/// grid indices, sign selectors and scales, then computes the expected float output independently
/// from the <i>same</i> vendored grid (<see cref="GgufIqGrids"/>) by re-deriving the bit-unpacking /
/// sign / scale / delta math from the ggml-quants.c reference. Equality of the two paths validates
/// the dequantizer's bit packing against the verbatim-vendored constants end-to-end.
///
/// <para>
/// Half-float scales are chosen exactly representable; comparisons use a tiny tolerance only to
/// absorb float rounding. The grid byte unpacking mirrors the C reinterpret-cast: byte j of a packed
/// grid entry is <c>(entry &gt;&gt; 8*j) &amp; 0xFF</c> (unsigned for IQ2/IQ3, signed int8 for IQ1).
/// </para>
/// </summary>
public class GgufIq123DequantTests
{
    private const float Tol = 1e-4f;

    private static void Half(List<byte> dst, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        dst.Add((byte)(bits & 0xFF));
        dst.Add((byte)(bits >> 8));
    }

    private static int GByte(ulong e, int j) => (int)((e >> (8 * j)) & 0xFF);
    private static int GByte(uint e, int j) => (int)((e >> (8 * j)) & 0xFF);
    private static int GSByte(ulong e, int j) => unchecked((sbyte)((e >> (8 * j)) & 0xFF));

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= Tol,
                $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    // Sign convention shared by IQ2_XXS/IQ2_XS/IQ3_XXS via ksigns_iq2xs + kmask_iq2xs.
    private static float SignAt(byte signs, int j) =>
        (signs & GgufIqGrids.KmaskIq2xs[j]) != 0 ? -1f : 1f;

    [Fact]
    public void Iq2Xxs_DequantizesExactly()
    {
        // Layout (66 B): fp16 d; qs[32] uint16 (64 B). Per ib32 we form aux32[0] (4 grid indices) and
        // aux32[1] (scale in top nibble + four 7-bit sign selectors).
        const float d = 0.5f;
        var grid = GgufIqGrids.Iq2xxsGrid;

        // Choose, per ib32, four grid indices and four 7-bit sign selectors plus a 4-bit scale nibble.
        var gridIdx = new int[8, 4];
        var signSel = new int[8, 4];
        var scaleNib = new uint[8];
        var rng = new Random(1234);
        for (int ib = 0; ib < 8; ib++)
        {
            scaleNib[ib] = (uint)rng.Next(0, 16);
            for (int l = 0; l < 4; l++)
            {
                gridIdx[ib, l] = rng.Next(0, 256);
                signSel[ib, l] = rng.Next(0, 128);
            }
        }

        var raw = new List<byte>();
        Half(raw, d);
        for (int ib = 0; ib < 8; ib++)
        {
            uint aux0 = 0, aux1 = 0;
            for (int l = 0; l < 4; l++)
            {
                aux0 |= (uint)(gridIdx[ib, l] & 0xFF) << (8 * l);
                aux1 |= (uint)(signSel[ib, l] & 127) << (7 * l);
            }
            aux1 |= scaleNib[ib] << 28;
            for (int k = 0; k < 4; k++) raw.Add((byte)((aux0 >> (8 * k)) & 0xFF));
            for (int k = 0; k < 4; k++) raw.Add((byte)((aux1 >> (8 * k)) & 0xFF));
        }

        var expected = new float[256];
        int oi = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            float db = d * (0.5f + scaleNib[ib]) * 0.25f;
            for (int l = 0; l < 4; l++)
            {
                ulong g = grid[gridIdx[ib, l]];
                byte signs = GgufIqGrids.KsignsIq2xs[signSel[ib, l]];
                for (int j = 0; j < 8; j++)
                    expected[oi++] = db * GByte(g, j) * SignAt(signs, j);
            }
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ2_XXS, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq2Xs_DequantizesExactly()
    {
        // Layout (74 B): fp16 d; qs[32] uint16 (64 B); scales[8]. Each uint16 = 9-bit grid index (low)
        // + 7-bit sign selector (high). scales byte ib32 = two 4-bit sub-scales.
        const float d = 0.25f;
        var grid = GgufIqGrids.Iq2xsGrid;
        var rng = new Random(99);

        var gridIdx = new int[8, 4];
        var signSel = new int[8, 4];
        var scLo = new int[8];
        var scHi = new int[8];
        for (int ib = 0; ib < 8; ib++)
        {
            scLo[ib] = rng.Next(0, 16);
            scHi[ib] = rng.Next(0, 16);
            for (int l = 0; l < 4; l++)
            {
                gridIdx[ib, l] = rng.Next(0, 512);
                signSel[ib, l] = rng.Next(0, 128);
            }
        }

        var raw = new List<byte>();
        Half(raw, d);
        for (int ib = 0; ib < 8; ib++)
            for (int l = 0; l < 4; l++)
            {
                int q = (gridIdx[ib, l] & 511) | ((signSel[ib, l] & 127) << 9);
                raw.Add((byte)(q & 0xFF));
                raw.Add((byte)((q >> 8) & 0xFF));
            }
        for (int ib = 0; ib < 8; ib++)
            raw.Add((byte)((scLo[ib] & 0xF) | ((scHi[ib] & 0xF) << 4)));

        var expected = new float[256];
        int oi = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            float db0 = d * (0.5f + scLo[ib]) * 0.25f;
            float db1 = d * (0.5f + scHi[ib]) * 0.25f;
            for (int l = 0; l < 4; l++)
            {
                ulong g = grid[gridIdx[ib, l]];
                byte signs = GgufIqGrids.KsignsIq2xs[signSel[ib, l]];
                float dl = (l / 2) == 0 ? db0 : db1;
                for (int j = 0; j < 8; j++)
                    expected[oi++] = dl * GByte(g, j) * SignAt(signs, j);
            }
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ2_XS, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq2S_DequantizesExactly()
    {
        // Layout (82 B): fp16 d; qs[64] (first 32 = grid-low, next 32 = sign bytes); qh[8]; scales[8].
        // Grid index = qs[l] | ((qh[ib32] << (8-2l)) & 0x300). Signs come from the sign bytes directly.
        const float d = 0.5f;
        var grid = GgufIqGrids.Iq2sGrid;
        var rng = new Random(7);

        var qsLow = new int[8, 4];   // 32 grid-low bytes
        var qhHi = new int[8];       // per-ib32 high bits (0..3 packed across the 4 l's)
        var signByte = new int[8, 4];
        var scLo = new int[8];
        var scHi = new int[8];
        // qh[ib32] is consumed as (qh << (8-2l)) & 0x300 for l=0..3 -> contributes bits 8-9 of index.
        // For l: shift 8,6,4,2. &0x300 keeps the two bits landing at 0x100/0x200.
        // To exercise the high bits, give qh a random byte and derive the resulting index.
        var qhByte = new int[8];
        for (int ib = 0; ib < 8; ib++)
        {
            scLo[ib] = rng.Next(0, 16);
            scHi[ib] = rng.Next(0, 16);
            qhByte[ib] = rng.Next(0, 256);
            for (int l = 0; l < 4; l++)
            {
                qsLow[ib, l] = rng.Next(0, 256);
                signByte[ib, l] = rng.Next(0, 256);
            }
        }
        _ = qhHi;

        var raw = new List<byte>();
        Half(raw, d);
        // qs grid-low: 4 per ib32 = 32 bytes
        for (int ib = 0; ib < 8; ib++)
            for (int l = 0; l < 4; l++)
                raw.Add((byte)qsLow[ib, l]);
        // sign bytes: 4 per ib32 = 32 bytes
        for (int ib = 0; ib < 8; ib++)
            for (int l = 0; l < 4; l++)
                raw.Add((byte)signByte[ib, l]);
        // qh[8]
        for (int ib = 0; ib < 8; ib++) raw.Add((byte)qhByte[ib]);
        // scales[8]
        for (int ib = 0; ib < 8; ib++)
            raw.Add((byte)((scLo[ib] & 0xF) | ((scHi[ib] & 0xF) << 4)));

        var expected = new float[256];
        int oi = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            float db0 = d * (0.5f + scLo[ib]) * 0.25f;
            float db1 = d * (0.5f + scHi[ib]) * 0.25f;
            for (int l = 0; l < 4; l++)
            {
                int gridIndex = qsLow[ib, l] | ((qhByte[ib] << (8 - 2 * l)) & 0x300);
                ulong g = grid[gridIndex];
                byte signs = (byte)signByte[ib, l];
                float dl = (l / 2) == 0 ? db0 : db1;
                for (int j = 0; j < 8; j++)
                    expected[oi++] = dl * GByte(g, j) * SignAt(signs, j);
            }
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ2_S, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq3Xxs_DequantizesExactly()
    {
        // Layout (98 B): fp16 d; qs[96] (3*32 grid indices, 8 per ib32); scales_and_signs = qs+64
        // (one uint32 per ib32: scale top nibble + four 7-bit sign selectors).
        const float d = 0.5f;
        var grid = GgufIqGrids.Iq3xxsGrid;
        var rng = new Random(424242);

        var qs = new int[8, 8];     // 8 grid indices per ib32
        var signSel = new int[8, 4];
        var scaleNib = new uint[8];
        for (int ib = 0; ib < 8; ib++)
        {
            scaleNib[ib] = (uint)rng.Next(0, 16);
            for (int k = 0; k < 8; k++) qs[ib, k] = rng.Next(0, 256);
            for (int l = 0; l < 4; l++) signSel[ib, l] = rng.Next(0, 128);
        }

        var raw = new List<byte>();
        Half(raw, d);
        // qs[96]: 8 bytes per ib32
        for (int ib = 0; ib < 8; ib++)
            for (int k = 0; k < 8; k++)
                raw.Add((byte)qs[ib, k]);
        // scales_and_signs: one uint32 per ib32
        for (int ib = 0; ib < 8; ib++)
        {
            uint aux = 0;
            for (int l = 0; l < 4; l++) aux |= (uint)(signSel[ib, l] & 127) << (7 * l);
            aux |= scaleNib[ib] << 28;
            for (int k = 0; k < 4; k++) raw.Add((byte)((aux >> (8 * k)) & 0xFF));
        }

        var expected = new float[256];
        int oi = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            float db = d * (0.5f + scaleNib[ib]) * 0.5f;
            for (int l = 0; l < 4; l++)
            {
                byte signs = GgufIqGrids.KsignsIq2xs[signSel[ib, l]];
                uint g1 = grid[qs[ib, 2 * l + 0]];
                uint g2 = grid[qs[ib, 2 * l + 1]];
                for (int j = 0; j < 4; j++)
                {
                    expected[oi + j + 0] = db * GByte(g1, j) * SignAt(signs, j + 0);
                    expected[oi + j + 4] = db * GByte(g2, j) * SignAt(signs, j + 4);
                }
                oi += 8;
            }
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ3_XXS, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq3S_DequantizesExactly()
    {
        // Layout (110 B): fp16 d; qs[64]; qh[8]; signs[32]; scales[4]. Processed two ib32 at a time.
        // Grid index = qs[2l] | ((qh[h] << (8-2l)) & 256) ; second member uses (7-2l).
        const float d = 0.5f;
        var grid = GgufIqGrids.Iq3sGrid;
        var rng = new Random(31337);

        var qs = new int[64];
        var qh = new int[8];
        var signs = new int[32];
        var scales = new int[4];
        for (int i = 0; i < 64; i++) qs[i] = rng.Next(0, 256);
        for (int i = 0; i < 8; i++) qh[i] = rng.Next(0, 256);
        for (int i = 0; i < 32; i++) signs[i] = rng.Next(0, 256);
        for (int i = 0; i < 4; i++) scales[i] = rng.Next(0, 256);

        var raw = new List<byte>();
        Half(raw, d);
        for (int i = 0; i < 64; i++) raw.Add((byte)qs[i]);
        for (int i = 0; i < 8; i++) raw.Add((byte)qh[i]);
        for (int i = 0; i < 32; i++) raw.Add((byte)signs[i]);
        for (int i = 0; i < 4; i++) raw.Add((byte)scales[i]);

        var expected = new float[256];
        int oi = 0;
        int qsPtr = 0, qhPtr = 0, signPtr = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2)
        {
            byte sc = (byte)scales[ib32 / 2];
            float db1 = d * (1 + 2 * (sc & 0xf));
            float db2 = d * (1 + 2 * (sc >> 4));
            int qh0 = qh[qhPtr + 0];
            int qh1 = qh[qhPtr + 1];
            for (int l = 0; l < 4; l++)
            {
                uint g1 = grid[qs[qsPtr + 2 * l + 0] | ((qh0 << (8 - 2 * l)) & 256)];
                uint g2 = grid[qs[qsPtr + 2 * l + 1] | ((qh0 << (7 - 2 * l)) & 256)];
                byte s = (byte)signs[signPtr + l];
                for (int j = 0; j < 4; j++)
                {
                    expected[oi + j + 0] = db1 * GByte(g1, j) * SignAt(s, j + 0);
                    expected[oi + j + 4] = db1 * GByte(g2, j) * SignAt(s, j + 4);
                }
                oi += 8;
            }
            qsPtr += 8; signPtr += 4;
            for (int l = 0; l < 4; l++)
            {
                uint g1 = grid[qs[qsPtr + 2 * l + 0] | ((qh1 << (8 - 2 * l)) & 256)];
                uint g2 = grid[qs[qsPtr + 2 * l + 1] | ((qh1 << (7 - 2 * l)) & 256)];
                byte s = (byte)signs[signPtr + l];
                for (int j = 0; j < 4; j++)
                {
                    expected[oi + j + 0] = db2 * GByte(g1, j) * SignAt(s, j + 0);
                    expected[oi + j + 4] = db2 * GByte(g2, j) * SignAt(s, j + 4);
                }
                oi += 8;
            }
            qhPtr += 2; qsPtr += 8; signPtr += 4;
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ3_S, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq1S_DequantizesExactly()
    {
        // Layout (50 B): fp16 d; qs[32]; qh[8] uint16. Per ib: dl = d*(2*((qh>>12)&7)+1);
        // delta = (qh & 0x8000) ? -0.125 : 0.125. Grid index = qs[l] | (((qh>>3l)&7)<<8). Signed grid.
        const float d = 1.0f;
        var grid = GgufIqGrids.Iq1sGrid;
        var rng = new Random(2024);

        var qs = new int[32];
        var qh = new int[8];
        for (int i = 0; i < 32; i++) qs[i] = rng.Next(0, 256);
        for (int i = 0; i < 8; i++) qh[i] = rng.Next(0, 0x10000);

        var raw = new List<byte>();
        Half(raw, d);
        for (int i = 0; i < 32; i++) raw.Add((byte)qs[i]);
        for (int i = 0; i < 8; i++) { raw.Add((byte)(qh[i] & 0xFF)); raw.Add((byte)((qh[i] >> 8) & 0xFF)); }

        var expected = new float[256];
        int oi = 0;
        int qsPtr = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
            float delta = (qh[ib] & 0x8000) != 0 ? -0.125f : 0.125f;
            for (int l = 0; l < 4; l++)
            {
                int gridIndex = qs[qsPtr + l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                ulong g = grid[gridIndex];
                for (int j = 0; j < 8; j++)
                    expected[oi++] = dl * (GSByte(g, j) + delta);
            }
            qsPtr += 4;
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ1_S, 256);
        AssertClose(expected, got);
    }

    [Fact]
    public void Iq1M_DequantizesExactly()
    {
        // Layout (56 B): qs[32]; qh[16]; scales[8] (4 uint16). fp16 scale reassembled from the top
        // nibbles of the 4 scale uint16. Per ib: two 3-bit sub-scales; four grid points with per-point
        // delta from the 0x08/0x80 bits of two qh bytes. Signed grid.
        var grid = GgufIqGrids.Iq1sGrid;
        var rng = new Random(555);

        var qs = new int[32];
        var qh = new int[16];
        for (int i = 0; i < 32; i++) qs[i] = rng.Next(0, 256);
        for (int i = 0; i < 16; i++) qh[i] = rng.Next(0, 256);

        // Build the four scale uint16 so the reassembled fp16 is exactly 1.0 and the 3-bit sub-scales
        // are controllable. fp16(1.0) = 0x3C00. The reassembly takes:
        //   scaleBits = (sc0>>12) | ((sc1>>8)&0x00f0) | ((sc2>>4)&0x0f00) | (sc3&0xf000)
        // i.e. nibble0 from sc0[15:12], nibble1 from sc1[15:12], nibble2 from sc2[15:12], nibble3 from
        // sc3[15:12]. For 0x3C00: n0=0, n1=0, n2=0xC, n3=0x3.
        // The low 12 bits of each sc carry the 3-bit sub-scales: sub-scale for ib uses
        //   (sc[ib/2] >> (6*(ib%2)+0)) & 7  and  (sc[ib/2] >> (6*(ib%2)+3)) & 7.
        var subLo = new int[8];
        var subHi = new int[8];
        for (int ib = 0; ib < 8; ib++) { subLo[ib] = rng.Next(0, 8); subHi[ib] = rng.Next(0, 8); }

        int[] sc = new int[4];
        for (int k = 0; k < 4; k++)
        {
            // low 12 bits: ib = 2k (even, shift 0) and ib = 2k+1 (odd, shift 6)
            int evenLo = subLo[2 * k], evenHi = subHi[2 * k];
            int oddLo = subLo[2 * k + 1], oddHi = subHi[2 * k + 1];
            int low12 = (evenLo & 7) | ((evenHi & 7) << 3) | ((oddLo & 7) << 6) | ((oddHi & 7) << 9);
            sc[k] = low12; // top nibble set below
        }
        // Set top nibbles to encode fp16 1.0 = 0x3C00: n0=0,n1=0,n2=0xC,n3=0x3
        sc[0] |= 0x0 << 12;
        sc[1] |= 0x0 << 12;
        sc[2] |= 0xC << 12;
        sc[3] |= 0x3 << 12;

        var raw = new List<byte>();
        for (int i = 0; i < 32; i++) raw.Add((byte)qs[i]);
        for (int i = 0; i < 16; i++) raw.Add((byte)qh[i]);
        for (int k = 0; k < 4; k++) { raw.Add((byte)(sc[k] & 0xFF)); raw.Add((byte)((sc[k] >> 8) & 0xFF)); }

        // Independent expected computation.
        ushort scaleBits = (ushort)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        float d = (float)BitConverter.UInt16BitsToHalf(scaleBits);
        Assert.Equal(1.0f, d); // sanity: our nibble choice reassembles to fp16 1.0

        var expected = new float[256];
        int oi = 0;
        int qsPtr = 0, qhPtr = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            int qh0 = qh[qhPtr + 0];
            int qh1 = qh[qhPtr + 1];
            int[] idx =
            {
                qs[qsPtr + 0] | ((qh0 << 8) & 0x700),
                qs[qsPtr + 1] | ((qh0 << 4) & 0x700),
                qs[qsPtr + 2] | ((qh1 << 8) & 0x700),
                qs[qsPtr + 3] | ((qh1 << 4) & 0x700),
            };
            float[] delta =
            {
                (qh0 & 0x08) != 0 ? -0.125f : 0.125f,
                (qh0 & 0x80) != 0 ? -0.125f : 0.125f,
                (qh1 & 0x08) != 0 ? -0.125f : 0.125f,
                (qh1 & 0x80) != 0 ? -0.125f : 0.125f,
            };
            for (int l = 0; l < 2; l++)
            {
                ulong g = grid[idx[l]];
                for (int j = 0; j < 8; j++) expected[oi++] = dl1 * (GSByte(g, j) + delta[l]);
            }
            for (int l = 2; l < 4; l++)
            {
                ulong g = grid[idx[l]];
                for (int j = 0; j < 8; j++) expected[oi++] = dl2 * (GSByte(g, j) + delta[l]);
            }
            qsPtr += 4; qhPtr += 2;
        }

        float[] got = GgufDequant.Dequantize(raw.ToArray(), GgmlType.IQ1_M, 256);
        AssertClose(expected, got);
    }
}
