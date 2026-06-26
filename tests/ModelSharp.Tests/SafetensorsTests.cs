using System;
using System.Buffers.Binary;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using ModelSharp;
using ModelSharp.Tensors;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

public class SafetensorsTests
{
    // ---- helpers ---------------------------------------------------------------------

    private static byte[] ToBytes<T>(params T[] values) where T : unmanaged =>
        MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

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

    private static ushort BF16(float v) => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
    private static ushort F16(float v) => BitConverter.HalfToUInt16Bits((Half)v);

    // ---- round-trips -----------------------------------------------------------------

    [Fact]
    public void F32_Tensor_Reads_Back_Exactly()
    {
        byte[] data = ToBytes(1.5f, -2.25f, 0.75f, 4.0f);
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}", data);

        SafetensorsFile f = SafetensorsFile.FromBytes(buf);

        Assert.True(f.Contains("w"));
        Assert.Equal(new[] { "w" }, f.Names);
        SafetensorsTensorInfo info = f.GetInfo("w");
        Assert.Equal(SafetensorsDtype.Float32, info.Dtype);
        Assert.Equal(new TensorShape(2, 2), info.Shape);
        Assert.Equal(16, info.ByteLength);

        var t = Assert.IsType<Tensor<float>>(f.GetTensor("w"));
        Assert.Equal(ElementType.Float32, t.Dtype);
        Assert.Equal(new[] { 1.5f, -2.25f, 0.75f, 4.0f }, t.Span.ToArray());
    }

    [Fact]
    public void I64_Tensor_Reads_Back_Exactly()
    {
        byte[] data = ToBytes(1L, -2L, 3L);
        byte[] buf = Build(
            "{\"ids\":{\"dtype\":\"I64\",\"shape\":[3],\"data_offsets\":[0,24]}}", data);

        SafetensorsFile f = SafetensorsFile.FromBytes(buf);

        var t = Assert.IsType<Tensor<long>>(f.GetTensor("ids"));
        Assert.Equal(ElementType.Int64, t.Dtype);
        Assert.Equal(new[] { 1L, -2L, 3L }, t.Span.ToArray());
    }

    [Fact]
    public void BFloat16_Decodes_To_Expected_Floats()
    {
        byte[] data = ToBytes(BF16(1.0f), BF16(-2.0f), BF16(0.5f));
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[0,6]}}", data);

        SafetensorsFile f = SafetensorsFile.FromBytes(buf);

        var t = Assert.IsType<Tensor<float>>(f.GetTensor("w"));
        Assert.Equal(new[] { 1.0f, -2.0f, 0.5f }, t.Span.ToArray());
        Assert.Equal(SafetensorsDtype.BFloat16, f.GetInfo("w").Dtype);
    }

    [Fact]
    public void Float16_Decodes_To_Expected_Floats()
    {
        byte[] data = ToBytes(F16(1.0f), F16(-2.0f), F16(0.5f));
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F16\",\"shape\":[3],\"data_offsets\":[0,6]}}", data);

        SafetensorsFile f = SafetensorsFile.FromBytes(buf);

        var t = Assert.IsType<Tensor<float>>(f.GetTensor("w"));
        Assert.Equal(new[] { 1.0f, -2.0f, 0.5f }, t.Span.ToArray());
    }

    [Fact]
    public void Float16_Raw_Preserves_Bits()
    {
        byte[] data = ToBytes(F16(1.0f), F16(-2.0f), F16(0.5f));
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F16\",\"shape\":[3],\"data_offsets\":[0,6]}}", data);

        SafetensorsFile f = SafetensorsFile.FromBytes(buf);

        var raw = Assert.IsType<Tensor<ushort>>(f.GetTensorRaw("w"));
        Assert.Equal(new[] { F16(1.0f), F16(-2.0f), F16(0.5f) }, raw.Span.ToArray());
    }

    [Fact]
    public void Bool_Tensor_Reads_Back()
    {
        byte[] data = { 1, 0, 1 };
        byte[] buf = Build(
            "{\"m\":{\"dtype\":\"BOOL\",\"shape\":[3],\"data_offsets\":[0,3]}}", data);

        SafetensorsFile f = SafetensorsFile.FromBytes(buf);

        var t = Assert.IsType<Tensor<bool>>(f.GetTensor("m"));
        Assert.Equal(new[] { true, false, true }, t.Span.ToArray());
    }

    // ---- metadata --------------------------------------------------------------------

    [Fact]
    public void Metadata_Is_Exposed_And_Not_A_Tensor()
    {
        byte[] data = ToBytes(1.0f);
        byte[] buf = Build(
            "{\"__metadata__\":{\"format\":\"pt\",\"author\":\"x\"}," +
            "\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", data);

        SafetensorsFile f = SafetensorsFile.FromBytes(buf);

        Assert.Equal("pt", f.Metadata["format"]);
        Assert.Equal("x", f.Metadata["author"]);
        Assert.False(f.Contains("__metadata__"));
        Assert.DoesNotContain("__metadata__", f.Names);
        Assert.Equal(1, f.Count);
    }

    // ---- error handling --------------------------------------------------------------

    [Fact]
    public void Truncated_Size_Prefix_Throws()
    {
        Assert.Throws<ModelSharpException>(() => SafetensorsFile.FromBytes(new byte[] { 1, 2, 3 }));
    }

    [Fact]
    public void Header_Length_Beyond_Buffer_Throws()
    {
        var buf = new byte[12];
        BinaryPrimitives.WriteUInt64LittleEndian(buf, 9999); // claims a huge header
        Assert.Throws<ModelSharpException>(() => SafetensorsFile.FromBytes(buf));
    }

    [Fact]
    public void Bad_Json_Throws()
    {
        byte[] buf = Build("this is not json", Array.Empty<byte>());
        Assert.Throws<ModelSharpException>(() => SafetensorsFile.FromBytes(buf));
    }

    [Fact]
    public void Bad_Data_Offsets_Length_Throws()
    {
        // shape [2,2] F32 needs 16 bytes, but data_offsets span only 8.
        byte[] data = ToBytes(1.0f, 2.0f, 3.0f, 4.0f);
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,8]}}", data);
        Assert.Throws<ModelSharpException>(() => SafetensorsFile.FromBytes(buf));
    }

    [Fact]
    public void Offset_Out_Of_Range_Throws()
    {
        byte[] data = ToBytes(1.0f, 2.0f); // only 8 bytes in the data section
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", data);
        Assert.Throws<ModelSharpException>(() => SafetensorsFile.FromBytes(buf));
    }

    [Fact]
    public void Unsupported_Dtype_Throws()
    {
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F8_E4M3\",\"shape\":[1],\"data_offsets\":[0,1]}}", new byte[] { 0 });
        Assert.Throws<ModelSharpException>(() => SafetensorsFile.FromBytes(buf));
    }

    [Fact]
    public void Missing_Tensor_Throws()
    {
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", ToBytes(1.0f));
        SafetensorsFile f = SafetensorsFile.FromBytes(buf);
        Assert.Throws<ModelSharpException>(() => f.GetTensor("missing"));
        Assert.False(f.TryGetInfo("missing", out _));
    }

    // ---- streams and files -----------------------------------------------------------

    [Fact]
    public void FromStream_Round_Trips()
    {
        byte[] data = ToBytes(7.0f, 8.0f);
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", data);

        using var ms = new MemoryStream(buf, writable: false);
        SafetensorsFile f = SafetensorsFile.FromStream(ms);

        var t = Assert.IsType<Tensor<float>>(f.GetTensor("w"));
        Assert.Equal(new[] { 7.0f, 8.0f }, t.Span.ToArray());
    }

    [Fact]
    public void FromFiles_Reads_Back_A_Temp_File()
    {
        byte[] data = ToBytes(10L, 20L, 30L);
        byte[] buf = Build(
            "{\"ids\":{\"dtype\":\"I64\",\"shape\":[3],\"data_offsets\":[0,24]}}", data);

        string path = Path.Combine(Path.GetTempPath(), $"modelsharp_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, buf);
        try
        {
            using SafetensorsFile f = SafetensorsFile.FromFiles(path);
            var t = Assert.IsType<Tensor<long>>(f.GetTensor("ids"));
            Assert.Equal(new[] { 10L, 20L, 30L }, t.Span.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ---- memory-mapped / long-offset path --------------------------------------------

    [Fact]
    public void FromFile_Mmap_Path_Decodes_Identically_To_FromBytes()
    {
        // A multi-tensor file so several entries decode through the file-backed data section.
        byte[] floats = ToBytes(1.5f, -2.25f, 0.75f, 4.0f);
        byte[] longs = ToBytes(10L, 20L, 30L);
        byte[] half = ToBytes(F16(1.0f), F16(-2.0f), F16(0.5f));
        var data = new byte[floats.Length + longs.Length + half.Length];
        Buffer.BlockCopy(floats, 0, data, 0, floats.Length);
        Buffer.BlockCopy(longs, 0, data, floats.Length, longs.Length);
        Buffer.BlockCopy(half, 0, data, floats.Length + longs.Length, half.Length);

        // offsets: w [0,16), ids [16,40), h [40,46)
        string json =
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}," +
            "\"ids\":{\"dtype\":\"I64\",\"shape\":[3],\"data_offsets\":[16,40]}," +
            "\"h\":{\"dtype\":\"F16\",\"shape\":[3],\"data_offsets\":[40,46]}}";
        byte[] buf = Build(json, data);

        SafetensorsFile inMem = SafetensorsFile.FromBytes(buf);

        string path = Path.Combine(Path.GetTempPath(), $"modelsharp_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, buf);
        try
        {
            using SafetensorsFile mapped = SafetensorsFile.FromFile(path);

            Assert.Equal(inMem.Count, mapped.Count);

            var wMem = Assert.IsType<Tensor<float>>(inMem.GetTensor("w"));
            var wMap = Assert.IsType<Tensor<float>>(mapped.GetTensor("w"));
            Assert.Equal(wMem.Span.ToArray(), wMap.Span.ToArray());

            var idsMem = Assert.IsType<Tensor<long>>(inMem.GetTensor("ids"));
            var idsMap = Assert.IsType<Tensor<long>>(mapped.GetTensor("ids"));
            Assert.Equal(idsMem.Span.ToArray(), idsMap.Span.ToArray());

            var hMem = Assert.IsType<Tensor<float>>(inMem.GetTensor("h"));
            var hMap = Assert.IsType<Tensor<float>>(mapped.GetTensor("h"));
            Assert.Equal(hMem.Span.ToArray(), hMap.Span.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Dispose_Releases_The_Mapped_File()
    {
        byte[] data = ToBytes(1.0f);
        byte[] buf = Build(
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", data);

        string path = Path.Combine(Path.GetTempPath(), $"modelsharp_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, buf);
        try
        {
            SafetensorsFile f = SafetensorsFile.FromFile(path);
            var t = Assert.IsType<Tensor<float>>(f.GetTensor("w"));
            Assert.Equal(new[] { 1.0f }, t.Span.ToArray());
            f.Dispose();

            // After dispose the mapping is released: the file can be rewritten/deleted, and
            // accessing a tensor throws ObjectDisposedException.
            File.WriteAllBytes(path, buf);
            Assert.Throws<ObjectDisposedException>(() => f.GetTensor("w"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ---- sharded index.json ------------------------------------------------------------

    [Fact]
    public void FromIndex_Exposes_Union_Of_Shards()
    {
        byte[] shard1 = Build(
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", ToBytes(1.0f, 2.0f));
        byte[] shard2 = Build(
            "{\"b\":{\"dtype\":\"I64\",\"shape\":[2],\"data_offsets\":[0,16]}}", ToBytes(7L, 8L));

        string dir = Path.Combine(Path.GetTempPath(), $"modelsharp_idx_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string s1 = Path.Combine(dir, "model-00001-of-00002.safetensors");
        string s2 = Path.Combine(dir, "model-00002-of-00002.safetensors");
        string idx = Path.Combine(dir, "model.safetensors.index.json");
        File.WriteAllBytes(s1, shard1);
        File.WriteAllBytes(s2, shard2);
        File.WriteAllText(idx,
            "{\"metadata\":{\"total_size\":24}," +
            "\"weight_map\":{" +
            "\"a\":\"model-00001-of-00002.safetensors\"," +
            "\"b\":\"model-00002-of-00002.safetensors\"}}");

        try
        {
            using SafetensorsFile f = SafetensorsFile.FromIndex(idx);

            Assert.Equal(2, f.Count);
            Assert.True(f.Contains("a"));
            Assert.True(f.Contains("b"));

            var a = Assert.IsType<Tensor<float>>(f.GetTensor("a"));
            Assert.Equal(new[] { 1.0f, 2.0f }, a.Span.ToArray());

            var b = Assert.IsType<Tensor<long>>(f.GetTensor("b"));
            Assert.Equal(new[] { 7L, 8L }, b.Span.ToArray());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void FromIndex_Missing_Tensor_Throws()
    {
        byte[] shard1 = Build(
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", ToBytes(1.0f, 2.0f));

        string dir = Path.Combine(Path.GetTempPath(), $"modelsharp_idx_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string s1 = Path.Combine(dir, "shard1.safetensors");
        string idx = Path.Combine(dir, "bad.index.json");
        File.WriteAllBytes(s1, shard1);
        // weight_map references a tensor 'c' that no shard actually contains.
        File.WriteAllText(idx,
            "{\"weight_map\":{\"a\":\"shard1.safetensors\",\"c\":\"shard1.safetensors\"}}");

        try
        {
            Assert.Throws<ModelSharpException>(() => SafetensorsFile.FromIndex(idx));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
