using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using ModelSharp;
using ModelSharp.Tensors;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

public class GgufTests
{
    // ---- a tiny little-endian GGUF writer ---------------------------------------------

    private sealed class GgufWriter
    {
        private readonly MemoryStream _s = new();

        public void U8(byte v) => _s.WriteByte(v);

        public void U32(uint v)
        {
            Span<byte> b = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(b, v);
            _s.Write(b);
        }

        public void I32(int v) => U32(unchecked((uint)v));

        public void U64(ulong v)
        {
            Span<byte> b = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(b, v);
            _s.Write(b);
        }

        public void F32(float v) => U32(BitConverter.SingleToUInt32Bits(v));

        public void Str(string v)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(v);
            U64((ulong)bytes.Length);
            _s.Write(bytes);
        }

        public void Raw(byte[] bytes) => _s.Write(bytes);

        public long Position => _s.Position;

        public byte[] ToArray() => _s.ToArray();

        // Metadata helpers: type tag then value.
        public void KvU32(string key, uint v) { Str(key); U32((uint)GgufValueType.UInt32); U32(v); }
        public void KvI32(string key, int v) { Str(key); U32((uint)GgufValueType.Int32); I32(v); }
        public void KvF32(string key, float v) { Str(key); U32((uint)GgufValueType.Float32); F32(v); }
        public void KvBool(string key, bool v) { Str(key); U32((uint)GgufValueType.Bool); U8((byte)(v ? 1 : 0)); }
        public void KvString(string key, string v) { Str(key); U32((uint)GgufValueType.String); Str(v); }

        public void KvStringArray(string key, params string[] values)
        {
            Str(key);
            U32((uint)GgufValueType.Array);
            U32((uint)GgufValueType.String);
            U64((ulong)values.Length);
            foreach (string s in values) Str(s);
        }

        public void KvI32Array(string key, params int[] values)
        {
            Str(key);
            U32((uint)GgufValueType.Array);
            U32((uint)GgufValueType.Int32);
            U64((ulong)values.Length);
            foreach (int i in values) I32(i);
        }
    }

    private static byte[] ToBytes<T>(params T[] values) where T : unmanaged =>
        MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static string WriteTempFile(byte[] bytes)
    {
        string path = Path.Combine(Path.GetTempPath(), $"modelsharp_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, bytes);
        return path;
    }

    // ---- round-trip --------------------------------------------------------------------

    [Fact]
    public void Reads_Metadata_And_F32_Tensor()
    {
        float[] values = { 1.5f, -2.25f, 0.75f, 4.0f, 5.0f, 6.0f };
        var w = new GgufWriter();

        // Fixed header.
        w.U32(GgufFile.Magic);
        w.U32(3);              // version 3
        w.U64(1);              // tensor_count
        w.U64(5);              // metadata_kv_count

        // Metadata: a string, an int, a float, a bool, and a string array.
        w.KvString("general.architecture", "llama");
        w.KvU32("llama.block_count", 32u);
        w.KvF32("llama.rope.freq_base", 10000.0f);
        w.KvBool("general.quantized", false);
        w.KvStringArray("tokenizer.ggml.tokens", "<s>", "</s>", "hello");

        // One F32 tensor, shape [2,3] (6 elements). GGUF dims are fastest-first; we want
        // row-major [2,3], so write dims in reversed order: 3 then 2.
        w.Str("weight");
        w.U32(2);              // n_dims
        w.U64(3);              // dim[0] fastest
        w.U64(2);              // dim[1]
        w.U32((uint)GgmlType.F32);
        w.U64(0);              // offset within data blob

        // Align to 32 (default), then write the tensor data.
        long pos = w.Position;
        long pad = (32 - (pos % 32)) % 32;
        if (pad > 0) w.Raw(new byte[pad]);
        w.Raw(ToBytes(values));

        string path = WriteTempFile(w.ToArray());
        try
        {
            using GgufFile f = GgufFile.FromFile(path);

            Assert.Equal(3u, f.Version);
            Assert.Equal(1, f.Count);

            Assert.Equal("llama", f.GetMetadataString("general.architecture"));
            Assert.Equal(32L, f.GetMetadataInt("llama.block_count"));

            Assert.True(f.TryGetMetadata("llama.rope.freq_base", out GgufMetadataValue freq));
            Assert.Equal(GgufValueType.Float32, freq.Type);
            Assert.Equal(10000.0f, Assert.IsType<float>(freq.Value));

            Assert.True(f.TryGetMetadata("general.quantized", out GgufMetadataValue q));
            Assert.False(Assert.IsType<bool>(q.Value));

            Assert.True(f.TryGetMetadata("tokenizer.ggml.tokens", out GgufMetadataValue toks));
            Assert.Equal(GgufValueType.Array, toks.Type);
            Assert.Equal(GgufValueType.String, toks.ArrayElementType);
            Assert.Equal(new[] { "<s>", "</s>", "hello" }, Assert.IsType<string[]>(toks.Value));

            GgufTensorInfo info = f.GetInfo("weight");
            Assert.Equal(GgmlType.F32, info.Type);
            Assert.Equal(new TensorShape(2, 3), info.Shape);
            Assert.Equal(24, info.ByteLength);

            var t = Assert.IsType<Tensor<float>>(f.GetTensor("weight"));
            Assert.Equal(ElementType.Float32, t.Dtype);
            Assert.Equal(values, t.Span.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Reads_F16_Tensor_As_Floats()
    {
        ushort[] half =
        {
            BitConverter.HalfToUInt16Bits((Half)1.0f),
            BitConverter.HalfToUInt16Bits((Half)(-2.0f)),
            BitConverter.HalfToUInt16Bits((Half)0.5f),
        };
        var w = new GgufWriter();
        w.U32(GgufFile.Magic);
        w.U32(3);
        w.U64(1);
        w.U64(0);

        w.Str("h");
        w.U32(1);
        w.U64(3);
        w.U32((uint)GgmlType.F16);
        w.U64(0);

        long pos = w.Position;
        long pad = (32 - (pos % 32)) % 32;
        if (pad > 0) w.Raw(new byte[pad]);
        w.Raw(ToBytes(half));

        string path = WriteTempFile(w.ToArray());
        try
        {
            using GgufFile f = GgufFile.FromFile(path);
            var t = Assert.IsType<Tensor<float>>(f.GetTensor("h"));
            Assert.Equal(new[] { 1.0f, -2.0f, 0.5f }, t.Span.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ---- validation --------------------------------------------------------------------

    [Fact]
    public void Bad_Magic_Throws()
    {
        var w = new GgufWriter();
        w.U32(0xDEADBEEF);
        w.U32(3);
        w.U64(0);
        w.U64(0);
        // pad to the 24-byte minimum header so we reach the magic check, not a length check
        w.Raw(new byte[8]);

        string path = WriteTempFile(w.ToArray());
        try
        {
            Assert.Throws<ModelSharpException>(() => GgufFile.FromFile(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Unsupported_Version_Throws()
    {
        var w = new GgufWriter();
        w.U32(GgufFile.Magic);
        w.U32(1);              // version 1 is not supported
        w.U64(0);
        w.U64(0);
        w.Raw(new byte[8]);

        string path = WriteTempFile(w.ToArray());
        try
        {
            Assert.Throws<ModelSharpException>(() => GgufFile.FromFile(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Materializing_Quantized_Tensor_Dequantizes()
    {
        var w = new GgufWriter();
        w.U32(GgufFile.Magic);
        w.U32(3);
        w.U64(1);
        w.U64(0);

        // A Q4_0 tensor: block size 32, type size 18 bytes. One block = 32 elements.
        w.Str("q");
        w.U32(1);
        w.U64(32);
        w.U32((uint)GgmlType.Q4_0);
        w.U64(0);

        long pos = w.Position;
        long pad = (32 - (pos % 32)) % 32;
        if (pad > 0) w.Raw(new byte[pad]);
        w.Raw(new byte[18]);   // one Q4_0 block

        string path = WriteTempFile(w.ToArray());
        try
        {
            using GgufFile f = GgufFile.FromFile(path);

            GgufTensorInfo info = f.GetInfo("q");
            Assert.Equal(GgmlType.Q4_0, info.Type);
            Assert.Equal(18, info.ByteLength);

            // Raw bytes are surfaced...
            byte[] raw = f.GetRawTensorBytes("q");
            Assert.Equal(18, raw.Length);

            // ...and materializing now dequantizes to a float tensor (an all-zero
            // Q4_0 block has scale d == 0, so every element decodes to 0).
            Tensor t = f.GetTensor("q");
            Tensor<float> ft = t.AsFloat();
            Assert.Equal(32L, ft.Length);
            Assert.All(ft.Buffer.ToArray(), v => Assert.Equal(0f, v));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Custom_Alignment_Is_Honored()
    {
        float[] values = { 3.0f, 4.0f };
        var w = new GgufWriter();
        w.U32(GgufFile.Magic);
        w.U32(3);
        w.U64(1);
        w.U64(1);              // one metadata KV: general.alignment

        w.KvU32("general.alignment", 64u);

        w.Str("v");
        w.U32(1);
        w.U64(2);
        w.U32((uint)GgmlType.F32);
        w.U64(0);

        long pos = w.Position;
        long pad = (64 - (pos % 64)) % 64;
        if (pad > 0) w.Raw(new byte[pad]);
        w.Raw(ToBytes(values));

        string path = WriteTempFile(w.ToArray());
        try
        {
            using GgufFile f = GgufFile.FromFile(path);
            var t = Assert.IsType<Tensor<float>>(f.GetTensor("v"));
            Assert.Equal(values, t.Span.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }
}
