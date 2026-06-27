using System;
using System.Collections.Generic;
using System.IO;
using ModelSharp;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Verifies that ONNX external-data initializers load: weights live in a sibling data file
/// referenced by <c>TensorProto.data_location = EXTERNAL</c> plus <c>external_data</c> entries
/// (<c>location</c>/<c>offset</c>/<c>length</c>), NOT inline <c>raw_data</c>. Each test writes a
/// real sidecar data file and a hand-encoded ModelProto to a temp dir, calls
/// <see cref="ModelSharp.Onnx.OnnxModelLoader.LoadModel(string)"/>, and asserts exact values.
/// Encoding helpers mirror <c>Fp16InitializerLoadingTests</c> / <c>ControlFlowOpsTests</c>.
/// </summary>
public class ExternalDataLoadingTests
{
    // ONNX TensorProto.DataType
    private const int Float = 1;
    private const int Uint8 = 2;

    [Fact]
    public void Loader_Reads_Float32_External_Initializer()
    {
        float[] values = { 1.5f, -2.25f, 3.0f, 42.0f };
        byte[] weights = FloatsToLe(values);

        using var dir = new TempDir();
        File.WriteAllBytes(dir.File("weights.bin"), weights);
        string modelPath = dir.File("model.onnx");
        File.WriteAllBytes(modelPath, BuildModelWithExternalInitializer(
            "w", Float, dims: new[] { 4L }, location: "weights.bin", offset: 0, length: weights.Length));

        ModelGraph g = ModelSharp.Onnx.OnnxModelLoader.LoadModel(modelPath);

        Assert.True(g.Initializers.TryGetValue("w", out Tensor? tensor));
        Tensor<float> t = tensor!.AsFloat();
        Assert.Equal(values, t.Span.ToArray());
    }

    [Fact]
    public void Loader_Reads_Float32_External_Initializer_At_Nonzero_Offset()
    {
        // The wanted tensor is a 3-float slice that begins 16 bytes into the file (after 4
        // leading "padding" floats), exercising offset > 0 and length sub-slicing.
        float[] leading = { 100f, 101f, 102f, 103f };
        float[] wanted = { 7.0f, 8.5f, -9.0f };
        byte[] weights = Concat(FloatsToLe(leading), FloatsToLe(wanted));
        long offset = leading.Length * sizeof(float); // 16

        using var dir = new TempDir();
        File.WriteAllBytes(dir.File("weights.bin"), weights);
        string modelPath = dir.File("model.onnx");
        File.WriteAllBytes(modelPath, BuildModelWithExternalInitializer(
            "w", Float, dims: new[] { 3L }, location: "weights.bin",
            offset: offset, length: wanted.Length * sizeof(float)));

        ModelGraph g = ModelSharp.Onnx.OnnxModelLoader.LoadModel(modelPath);

        Tensor<float> t = g.Initializers["w"].AsFloat();
        Assert.Equal(wanted, t.Span.ToArray());
    }

    [Fact]
    public void Loader_Reads_Uint8_External_Initializer()
    {
        byte[] values = { 0, 1, 127, 200, 255 };

        using var dir = new TempDir();
        File.WriteAllBytes(dir.File("q.bin"), values);
        string modelPath = dir.File("model.onnx");
        File.WriteAllBytes(modelPath, BuildModelWithExternalInitializer(
            "q", Uint8, dims: new[] { 5L }, location: "q.bin", offset: 0, length: values.Length));

        ModelGraph g = ModelSharp.Onnx.OnnxModelLoader.LoadModel(modelPath);

        var t = (Tensor<byte>)g.Initializers["q"];
        Assert.Equal(values, t.Span.ToArray());
    }

    [Fact]
    public void Loader_Reads_External_Initializer_With_Omitted_Length_To_Eof()
    {
        // When external_data has no "length" key the slice runs from offset to end-of-file.
        float[] wanted = { 11f, 12f, 13f };
        byte[] weights = FloatsToLe(wanted);

        using var dir = new TempDir();
        File.WriteAllBytes(dir.File("w.bin"), weights);
        string modelPath = dir.File("model.onnx");
        File.WriteAllBytes(modelPath, BuildModelWithExternalInitializer(
            "w", Float, dims: new[] { 3L }, location: "w.bin", offset: 0, length: null));

        ModelGraph g = ModelSharp.Onnx.OnnxModelLoader.LoadModel(modelPath);

        Assert.Equal(wanted, g.Initializers["w"].AsFloat().Span.ToArray());
    }

    [Fact]
    public void ParseModel_From_Bytes_Throws_Clear_Error_On_External_Reference()
    {
        // No backing file path => external data can't be resolved; must throw a clear error.
        byte[] model = BuildModelWithExternalInitializer(
            "w", Float, dims: new[] { 1L }, location: "weights.bin", offset: 0, length: 4);

        ModelSharpException ex = Assert.Throws<ModelSharpException>(
            () => ModelSharp.Onnx.OnnxModelLoader.ParseModel(model));
        Assert.Contains("external data", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    // -- temp-dir cleanup --

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } =
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ms_extdata_" + Guid.NewGuid().ToString("N"));

        public TempDir() => Directory.CreateDirectory(Path);
        public string File(string name) => System.IO.Path.Combine(Path, name);

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { /* best-effort cleanup */ }
        }
    }

    // -- protobuf encoders (mirror Fp16InitializerLoadingTests) --

    private static byte[] BuildModelWithExternalInitializer(
        string name, int dataType, long[] dims, string location, long offset, long? length)
    {
        // external_data StringStringEntryProto: key(1)=string, value(2)=string.
        byte[] Entry(string key, string value) =>
            Concat(LenField(1, Str(key)), LenField(2, Str(value)));

        var parts = new List<byte[]>();
        foreach (long d in dims) parts.Add(VarintField(1, (ulong)d));
        parts.Add(VarintField(2, (ulong)dataType));            // data_type
        parts.Add(LenField(8, Str(name)));                     // name
        parts.Add(LenField(13, Entry("location", location)));  // external_data: location
        parts.Add(LenField(13, Entry("offset", offset.ToString(System.Globalization.CultureInfo.InvariantCulture))));
        if (length is long len)
            parts.Add(LenField(13, Entry("length", len.ToString(System.Globalization.CultureInfo.InvariantCulture))));
        parts.Add(VarintField(14, 1));                         // data_location = EXTERNAL
        byte[] tensorProto = Concat(parts.ToArray());

        byte[] graph = LenField(5, tensorProto);               // GraphProto.initializer(5)
        return LenField(7, graph);                             // ModelProto.graph(7)
    }

    private static byte[] FloatsToLe(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes; // x86/x64 + ARM little-endian; matches loader's MemoryMarshal.Cast decode.
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
