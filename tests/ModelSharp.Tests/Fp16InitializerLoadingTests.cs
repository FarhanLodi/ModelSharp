using System;
using System.Collections.Generic;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Verifies that half-precision (FLOAT16, data_type 10) and bfloat16 (BFLOAT16,
/// data_type 16) ONNX initializers are decoded to float32 tensors at load time.
/// Tests drive <see cref="ModelSharp.Onnx.OnnxModelLoader"/> by hand-encoding a minimal
/// ModelProto whose graph carries a single initializer, mirroring the encoding approach
/// in <c>ControlFlowOpsTests.Loader_Parses_Nested_Subgraph_From_Attribute</c>.
/// </summary>
public class Fp16InitializerLoadingTests
{
    // ONNX TensorProto.DataType
    private const int Float16 = 10;
    private const int BFloat16 = 16;

    [Fact]
    public void Loader_Decodes_Float16_RawData_To_Float32()
    {
        // Known IEEE half bit patterns: 1.0=0x3C00, 2.0=0x4000, -0.5=0xB800, 0.0=0x0000.
        ushort[] bits = { 0x3C00, 0x4000, 0xB800, 0x0000 };
        byte[] raw = RawLe16(bits);

        Tensor<float> t = LoadInitializer("w", Float16, dims: new[] { 4L }, raw: raw);

        Assert.Equal(ElementType.Float32, t.Dtype);
        var span = t.Span;
        Assert.Equal(1.0f, span[0]);
        Assert.Equal(2.0f, span[1]);
        Assert.Equal(-0.5f, span[2]);
        Assert.Equal(0.0f, span[3]);
    }

    [Fact]
    public void Loader_Decodes_BFloat16_RawData_To_Float32()
    {
        // Known bfloat16 bit patterns: 1.0=0x3F80, -2.0=0xC000, 0.0=0x0000.
        ushort[] bits = { 0x3F80, 0xC000, 0x0000 };
        byte[] raw = RawLe16(bits);

        Tensor<float> t = LoadInitializer("w", BFloat16, dims: new[] { 3L }, raw: raw);

        Assert.Equal(ElementType.Float32, t.Dtype);
        var span = t.Span;
        Assert.Equal(1.0f, span[0]);
        Assert.Equal(-2.0f, span[1]);
        Assert.Equal(0.0f, span[2]);
    }

    [Fact]
    public void Loader_Decodes_Float16_Int32Data_Packing_To_Float32()
    {
        // When not raw, ONNX packs FLOAT16 into int32_data (low 16 bits per int32).
        int[] int32Data = { 0x3C00, 0x4000, 0xB800 }; // 1.0, 2.0, -0.5

        Tensor<float> t = LoadInitializer("w", Float16, dims: new[] { 3L }, int32Data: int32Data);

        Assert.Equal(ElementType.Float32, t.Dtype);
        var span = t.Span;
        Assert.Equal(1.0f, span[0]);
        Assert.Equal(2.0f, span[1]);
        Assert.Equal(-0.5f, span[2]);
    }

    [Fact]
    public void Loader_Decodes_BFloat16_Int32Data_Packing_To_Float32()
    {
        int[] int32Data = { 0x3F80, 0xC000 }; // 1.0, -2.0

        Tensor<float> t = LoadInitializer("w", BFloat16, dims: new[] { 2L }, int32Data: int32Data);

        Assert.Equal(ElementType.Float32, t.Dtype);
        var span = t.Span;
        Assert.Equal(1.0f, span[0]);
        Assert.Equal(-2.0f, span[1]);
    }

    // -- helpers --

    /// <summary>Builds a one-initializer ModelProto, parses it, returns the named tensor as float32.</summary>
    private static Tensor<float> LoadInitializer(
        string name, int dataType, long[] dims, byte[]? raw = null, int[]? int32Data = null)
    {
        byte[] model = BuildModelWithInitializer(name, dataType, dims, raw, int32Data);
        ModelGraph g = ModelSharp.Onnx.OnnxModelLoader.ParseModel(model);
        Assert.True(g.Initializers.TryGetValue(name, out Tensor? tensor), $"initializer '{name}' not loaded");
        return tensor!.AsFloat();
    }

    private static byte[] BuildModelWithInitializer(
        string name, int dataType, long[] dims, byte[]? raw, int[]? int32Data)
    {
        // TensorProto: dims(1, varint, repeated), data_type(2, varint), name(8, string),
        // int32_data(5, varint, repeated) and/or raw_data(9, bytes).
        var parts = new List<byte[]>();
        foreach (long d in dims) parts.Add(VarintField(1, (ulong)d));
        parts.Add(VarintField(2, (ulong)dataType));
        parts.Add(LenField(8, Str(name)));
        if (int32Data is not null)
            foreach (int v in int32Data) parts.Add(VarintField(5, (ulong)(uint)v));
        if (raw is not null) parts.Add(LenField(9, raw));
        byte[] tensorProto = Concat(parts.ToArray());

        // GraphProto: initializer(5) = tensorProto.
        byte[] graph = LenField(5, tensorProto);

        // ModelProto: graph(7) = graph.
        return LenField(7, graph);
    }

    private static byte[] RawLe16(ushort[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            bytes[i * 2] = (byte)(values[i] & 0xFF);
            bytes[i * 2 + 1] = (byte)(values[i] >> 8);
        }
        return bytes;
    }

    private static byte[] Str(string s) => System.Text.Encoding.UTF8.GetBytes(s);

    private static byte[] LenField(int fieldNo, byte[] payload)
        => Concat(Varint((uint)((fieldNo << 3) | 2)), Varint((uint)payload.Length), payload);

    private static byte[] VarintField(int fieldNo, ulong value)
        => Concat(Varint((uint)((fieldNo << 3) | 0)), Varint(value));

    private static byte[] Varint(ulong v)
    {
        var bytes = new List<byte>();
        do { byte b = (byte)(v & 0x7F); v >>= 7; if (v != 0) b |= 0x80; bytes.Add(b); } while (v != 0);
        return bytes.ToArray();
    }

    private static byte[] Concat(params byte[][] parts)
    {
        int n = 0;
        foreach (var p in parts) n += p.Length;
        var outBuf = new byte[n];
        int off = 0;
        foreach (var p in parts) { Array.Copy(p, 0, outBuf, off, p.Length); off += p.Length; }
        return outBuf;
    }
}
