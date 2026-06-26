using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using ModelSharp;
using ModelSharp.Tensors;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

public class QuantizedWeightsTests
{
    // ---- helpers ---------------------------------------------------------------------

    private static byte[] ToBytes<T>(params T[] values) where T : unmanaged =>
        MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static ushort F16(float v) => BitConverter.HalfToUInt16Bits((Half)v);

    /// <summary>Assembles a safetensors buffer: 8-byte LE header length + JSON + data.</summary>
    private static byte[] Build(string json, byte[] data)
    {
        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
        var buf = new byte[8 + jsonBytes.Length + data.Length];
        BinaryPrimitives.WriteUInt64LittleEndian(buf, (ulong)jsonBytes.Length);
        Buffer.BlockCopy(jsonBytes, 0, buf, 8, jsonBytes.Length);
        Buffer.BlockCopy(data, 0, buf, 8 + jsonBytes.Length, data.Length);
        return buf;
    }

    /// <summary>Packs 4-bit codes (LSB-first, eight per int32) at the given bit positions.
    /// <paramref name="order"/> maps logical field index → bit field index inside the word.</summary>
    private static int Pack4(ReadOnlySpan<int> codes, int[] order)
    {
        int word = 0;
        for (int j = 0; j < codes.Length; j++)
            word |= (codes[j] & 0xF) << (order[j] * 4);
        return word;
    }

    private static readonly int[] Natural8 = { 0, 1, 2, 3, 4, 5, 6, 7 };
    private static readonly int[] AwqOrder = { 0, 4, 1, 5, 2, 6, 3, 7 };

    // ---- UnpackBits / UnpackNibbles --------------------------------------------------

    [Fact]
    public void UnpackNibbles_Extracts_Eight_LowToHigh()
    {
        // nibbles 0..7 placed at fields 0..7 -> 0x76543210
        int packed = 0;
        for (int i = 0; i < 8; i++) packed |= i << (i * 4);
        int[] got = QuantizedWeights.UnpackNibbles(packed);
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, got);
    }

    [Fact]
    public void UnpackNibbles_Handles_High_Nibble_Without_Sign_Extension()
    {
        // top nibble = 0xF means the int32 is negative; must not sign-extend.
        int packed = unchecked((int)0xF000_0000);
        int[] got = QuantizedWeights.UnpackNibbles(packed);
        Assert.Equal(new[] { 0, 0, 0, 0, 0, 0, 0, 15 }, got);
    }

    [Fact]
    public void UnpackBits_4Bit_Reads_Across_Words()
    {
        // word0: fields 1,2,3,...  word1: continues
        int[] codes0 = { 1, 2, 3, 4, 5, 6, 7, 8 };
        int[] codes1 = { 9, 10, 11, 12, 13, 14, 15, 0 };
        int w0 = Pack4(codes0, Natural8);
        int w1 = Pack4(codes1, Natural8);

        int[] got = QuantizedWeights.UnpackBits(new[] { w0, w1 }, bits: 4, count: 16);
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 }, got);
    }

    [Fact]
    public void UnpackBits_8Bit_Reads_Four_Per_Word()
    {
        int word = (0x12) | (0x34 << 8) | (0x56 << 16) | (0x78 << 24);
        int[] got = QuantizedWeights.UnpackBits(new[] { word }, bits: 8, count: 4);
        Assert.Equal(new[] { 0x12, 0x34, 0x56, 0x78 }, got);
    }

    [Fact]
    public void UnpackBits_Partial_Count_Stops_Early()
    {
        int word = Pack4(new[] { 5, 6, 7, 8, 0, 0, 0, 0 }, Natural8);
        int[] got = QuantizedWeights.UnpackBits(new[] { word }, bits: 4, count: 3);
        Assert.Equal(new[] { 5, 6, 7 }, got);
    }

    [Fact]
    public void UnpackBits_Rejects_Bad_Bits()
    {
        Assert.Throws<ModelSharpException>(() => QuantizedWeights.UnpackBits(new[] { 0 }, bits: 3, count: 1));
    }

    [Fact]
    public void UnpackBits_Rejects_Short_Span()
    {
        Assert.Throws<ModelSharpException>(() => QuantizedWeights.UnpackBits(new[] { 0 }, bits: 4, count: 9));
    }

    // ---- GPTQ ------------------------------------------------------------------------

    [Fact]
    public void DequantizeGptq_Reconstructs_Hand_Computed_Matrix()
    {
        // in_features = 8, out_features = 2, group_size = 8 -> 1 group.
        // qweight packed along rows: shape [in/8, out] = [1, 2].
        // Column 0 codes over the 8 input rows; column 1 codes.
        int inFeatures = 8, outFeatures = 2, groupSize = 8;
        int[] col0 = { 0, 1, 2, 3, 4, 5, 6, 7 };
        int[] col1 = { 8, 9, 10, 11, 12, 13, 14, 15 };

        int qwCol0 = Pack4(col0, Natural8);
        int qwCol1 = Pack4(col1, Natural8);
        // row-major [1,2]: [col0, col1]
        int[] qweight = { qwCol0, qwCol1 };

        // qzeros: shape [groups=1, out/8 = ceil(2/8)=1]. zeros: col0=2, col1=3.
        int qzWord = Pack4(new[] { 2, 3, 0, 0, 0, 0, 0, 0 }, Natural8);
        int[] qzeros = { qzWord };

        // scales: shape [1, 2]. col0=0.5, col1=0.25
        float s0 = 0.5f, s1 = 0.25f;
        ushort[] scales = { F16(s0), F16(s1) };

        byte[] data = Concat(ToBytes(qweight), ToBytes(qzeros), ToBytes(scales));
        int o1 = qweight.Length * 4;
        int o2 = o1 + qzeros.Length * 4;
        int o3 = o2 + scales.Length * 2;
        string json =
            "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[1,2],\"data_offsets\":[0," + o1 + "]}," +
            "\"L.qzeros\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[" + o1 + "," + o2 + "]}," +
            "\"L.scales\":{\"dtype\":\"F16\",\"shape\":[1,2],\"data_offsets\":[" + o2 + "," + o3 + "]}}";

        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, data));
        Tensor<float> w = QuantizedWeights.DequantizeGptq(f, "L", bits: 4, groupSize: groupSize);

        Assert.Equal(new TensorShape(inFeatures, outFeatures), w.Shape);
        ReadOnlySpan<float> got = w.Span;
        for (int r = 0; r < inFeatures; r++)
        {
            float exp0 = (col0[r] - 2) * s0;
            float exp1 = (col1[r] - 3) * s1;
            Assert.Equal(exp0, got[r * outFeatures + 0], 3);
            Assert.Equal(exp1, got[r * outFeatures + 1], 3);
        }
    }

    [Fact]
    public void DequantizeGptq_Two_Groups_Use_Distinct_Scales()
    {
        // in_features = 16, out_features = 1, group_size = 8 -> 2 groups.
        int inFeatures = 16, outFeatures = 1, groupSize = 8;
        int[] rows = { 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5 };
        int w0 = Pack4(new ReadOnlySpan<int>(rows, 0, 8), Natural8);
        int w1 = Pack4(new ReadOnlySpan<int>(rows, 8, 8), Natural8);
        int[] qweight = { w0, w1 }; // shape [2,1]

        // qzeros shape [groups=2, ceil(1/8)=1]; group0 zero=1, group1 zero=2.
        int[] qzeros =
        {
            Pack4(new[] { 1, 0, 0, 0, 0, 0, 0, 0 }, Natural8),
            Pack4(new[] { 2, 0, 0, 0, 0, 0, 0, 0 }, Natural8),
        };
        ushort[] scales = { F16(0.5f), F16(2.0f) }; // shape [2,1]

        byte[] data = Concat(ToBytes(qweight), ToBytes(qzeros), ToBytes(scales));
        int o1 = qweight.Length * 4, o2 = o1 + qzeros.Length * 4, o3 = o2 + scales.Length * 2;
        string json =
            "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[2,1],\"data_offsets\":[0," + o1 + "]}," +
            "\"L.qzeros\":{\"dtype\":\"I32\",\"shape\":[2,1],\"data_offsets\":[" + o1 + "," + o2 + "]}," +
            "\"L.scales\":{\"dtype\":\"F16\",\"shape\":[2,1],\"data_offsets\":[" + o2 + "," + o3 + "]}}";

        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, data));
        Tensor<float> w = QuantizedWeights.DequantizeGptq(f, "L", bits: 4, groupSize: groupSize);

        ReadOnlySpan<float> got = w.Span;
        // group0: (3-1)*0.5 = 1.0  ; group1: (5-2)*2.0 = 6.0
        for (int r = 0; r < 8; r++) Assert.Equal(1.0f, got[r], 3);
        for (int r = 8; r < inFeatures; r++) Assert.Equal(6.0f, got[r], 3);
    }

    [Fact]
    public void DequantizeGptq_Mismatched_Scales_Out_Dim_Throws()
    {
        // qweight out=2 but scales out=1.
        int[] qweight = { Pack4(new[] { 1, 1, 1, 1, 1, 1, 1, 1 }, Natural8), 0 }; // [1,2]
        int[] qzeros = { 0 }; // [1,1]
        ushort[] scales = { F16(1.0f) }; // [1,1] -- wrong
        byte[] data = Concat(ToBytes(qweight), ToBytes(qzeros), ToBytes(scales));
        int o1 = 8, o2 = 12, o3 = 14;
        string json =
            "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[1,2],\"data_offsets\":[0," + o1 + "]}," +
            "\"L.qzeros\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[" + o1 + "," + o2 + "]}," +
            "\"L.scales\":{\"dtype\":\"F16\",\"shape\":[1,1],\"data_offsets\":[" + o2 + "," + o3 + "]}}";

        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, data));
        Assert.Throws<ModelSharpException>(() => QuantizedWeights.DequantizeGptq(f, "L", 4, 8));
    }

    [Fact]
    public void DequantizeGptq_Bad_Group_Size_Throws()
    {
        // in_features = 8 not divisible by group size 3.
        int[] qweight = { 0, 0 };
        int[] qzeros = { 0 };
        ushort[] scales = { F16(1.0f), F16(1.0f) };
        byte[] data = Concat(ToBytes(qweight), ToBytes(qzeros), ToBytes(scales));
        int o1 = 8, o2 = 12, o3 = 16;
        string json =
            "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[1,2],\"data_offsets\":[0," + o1 + "]}," +
            "\"L.qzeros\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[" + o1 + "," + o2 + "]}," +
            "\"L.scales\":{\"dtype\":\"F16\",\"shape\":[1,2],\"data_offsets\":[" + o2 + "," + o3 + "]}}";

        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, data));
        Assert.Throws<ModelSharpException>(() => QuantizedWeights.DequantizeGptq(f, "L", 4, 3));
    }

    [Fact]
    public void DequantizeGptq_Missing_Tensor_Throws()
    {
        string json = "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}";
        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, ToBytes(new int[] { 0 })));
        Assert.Throws<ModelSharpException>(() => QuantizedWeights.DequantizeGptq(f, "L", 4, 8));
    }

    // ---- AWQ -------------------------------------------------------------------------

    [Fact]
    public void DequantizeAwq_Applies_Interleave_Order()
    {
        // in_features = 1, out_features = 8, group_size = 1 -> 1 group.
        // qweight shape [in=1, out/8 = 1]. Logical out codes 0..7 packed at AWQ positions.
        int inFeatures = 1, outFeatures = 8, groupSize = 1;
        int[] outCodes = { 1, 2, 3, 4, 5, 6, 7, 8 }; // logical out channel j -> code
        int qwWord = Pack4(outCodes, AwqOrder); // logical j stored at field AwqOrder[j]
        int[] qweight = { qwWord }; // [1,1]

        // qzeros shape [1,1] with same interleave; all zeros = 1.
        int qzWord = Pack4(new[] { 1, 1, 1, 1, 1, 1, 1, 1 }, AwqOrder);
        int[] qzeros = { qzWord };

        // scales shape [1, 8], NOT packed, natural order.
        float[] sVals = { 0.5f, 0.5f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f, 2.0f };
        var scales = new ushort[8];
        for (int i = 0; i < 8; i++) scales[i] = F16(sVals[i]);

        byte[] data = Concat(ToBytes(qweight), ToBytes(qzeros), ToBytes(scales));
        int o1 = 4, o2 = 8, o3 = 8 + 16;
        string json =
            "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[0," + o1 + "]}," +
            "\"L.qzeros\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[" + o1 + "," + o2 + "]}," +
            "\"L.scales\":{\"dtype\":\"F16\",\"shape\":[1,8],\"data_offsets\":[" + o2 + "," + o3 + "]}}";

        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, data));
        Tensor<float> w = QuantizedWeights.DequantizeAwq(f, "L", bits: 4, groupSize: groupSize);

        Assert.Equal(new TensorShape(inFeatures, outFeatures), w.Shape);
        ReadOnlySpan<float> got = w.Span;
        for (int j = 0; j < outFeatures; j++)
        {
            float exp = (outCodes[j] - 1) * sVals[j];
            Assert.Equal(exp, got[j], 3);
        }
    }

    [Fact]
    public void DequantizeAwq_Interleave_Differs_From_Natural_Packing()
    {
        // Pack codes in NATURAL order, then verify AWQ dequant reads them permuted (proving
        // the interleave is actually applied, not a no-op).
        int[] outCodes = { 0, 1, 2, 3, 4, 5, 6, 7 };
        int qwWord = Pack4(outCodes, Natural8); // packed natural
        int[] qweight = { qwWord };
        int[] qzeros = { 0 }; // all zero points = 0
        var scales = new ushort[8];
        for (int i = 0; i < 8; i++) scales[i] = F16(1.0f);

        byte[] data = Concat(ToBytes(qweight), ToBytes(qzeros), ToBytes(scales));
        int o1 = 4, o2 = 8, o3 = 8 + 16;
        string json =
            "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[0," + o1 + "]}," +
            "\"L.qzeros\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[" + o1 + "," + o2 + "]}," +
            "\"L.scales\":{\"dtype\":\"F16\",\"shape\":[1,8],\"data_offsets\":[" + o2 + "," + o3 + "]}}";

        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, data));
        Tensor<float> w = QuantizedWeights.DequantizeAwq(f, "L", bits: 4, groupSize: 1);

        // logical out j reads field AwqOrder[j]; with natural packing field k holds code k.
        ReadOnlySpan<float> got = w.Span;
        for (int j = 0; j < 8; j++)
            Assert.Equal(AwqOrder[j], got[j], 3);
    }

    [Fact]
    public void DequantizeAwq_Mismatched_Qzeros_Throws()
    {
        // qweight [1,1] (out=8), but qzeros declared [1,2] which is wrong.
        int[] qweight = { 0 };
        int[] qzeros = { 0, 0 };
        var scales = new ushort[8];
        for (int i = 0; i < 8; i++) scales[i] = F16(1.0f);

        byte[] data = Concat(ToBytes(qweight), ToBytes(qzeros), ToBytes(scales));
        int o1 = 4, o2 = 12, o3 = 12 + 16;
        string json =
            "{\"L.qweight\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[0," + o1 + "]}," +
            "\"L.qzeros\":{\"dtype\":\"I32\",\"shape\":[1,2],\"data_offsets\":[" + o1 + "," + o2 + "]}," +
            "\"L.scales\":{\"dtype\":\"F16\",\"shape\":[1,8],\"data_offsets\":[" + o2 + "," + o3 + "]}}";

        SafetensorsFile f = SafetensorsFile.FromBytes(Build(json, data));
        Assert.Throws<ModelSharpException>(() => QuantizedWeights.DequantizeAwq(f, "L", 4, 1));
    }

    // ---- shared ----------------------------------------------------------------------

    private static byte[] Concat(params byte[][] parts)
    {
        var list = new List<byte>();
        foreach (byte[] p in parts) list.AddRange(p);
        return list.ToArray();
    }
}
