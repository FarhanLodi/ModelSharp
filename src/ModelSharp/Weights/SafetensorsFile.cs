using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using ModelSharp.Tensors;

namespace ModelSharp.Weights;

/// <summary>
/// Reader for the <c>safetensors</c> weight format used by virtually every HuggingFace
/// model. The on-disk layout is:
/// <list type="number">
///   <item>8 bytes — a little-endian <see cref="ulong"/> giving the header length <c>N</c>.</item>
///   <item><c>N</c> bytes — a UTF-8 JSON object mapping each tensor name to
///         <c>{ "dtype", "shape", "data_offsets":[begin,end] }</c>, plus an optional
///         <c>"__metadata__"</c> string→string map.</item>
///   <item>The remaining bytes — the packed, little-endian tensor data buffer; every
///         tensor's <c>data_offsets</c> are byte ranges relative to the start of this section.</item>
/// </list>
/// The header is parsed and validated eagerly; tensor data is materialized lazily and only
/// copied (never mutated in place) when requested.
/// <para>
/// <b>Endianness:</b> the data section is little-endian. On a little-endian host (the .NET
/// norm) a direct reinterpret/copy is correct; reading multi-byte tensors on a big-endian
/// host is rejected rather than producing byte-swapped garbage.
/// </para>
/// </summary>
public sealed class SafetensorsFile
{
    /// <summary>Size of the leading little-endian header-length prefix.</summary>
    private const int HeaderSizePrefix = 8;

    /// <summary>Reserved header key carrying free-form string metadata, not a tensor.</summary>
    private const string MetadataKey = "__metadata__";

    private readonly Dictionary<string, Entry> _entries;
    private readonly List<string> _names;
    private readonly Dictionary<string, string> _metadata;

    private SafetensorsFile(
        Dictionary<string, Entry> entries,
        List<string> names,
        Dictionary<string, string> metadata)
    {
        _entries = entries;
        _names = names;
        _metadata = metadata;
    }

    /// <summary>The tensor names, in header order. Does not include <c>"__metadata__"</c>.</summary>
    public IReadOnlyCollection<string> Names => _names;

    /// <summary>The file's free-form metadata (the <c>"__metadata__"</c> map); empty if absent.</summary>
    public IReadOnlyDictionary<string, string> Metadata => _metadata;

    /// <summary>Number of tensors in the file.</summary>
    public int Count => _names.Count;

    // -------------------------------------------------------------------------------------
    // Factories
    // -------------------------------------------------------------------------------------

    /// <summary>Loads and parses a <c>.safetensors</c> file from a path.</summary>
    public static SafetensorsFile FromFile(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        return FromBytes(File.ReadAllBytes(path));
    }

    /// <summary>
    /// Loads one or more <c>.safetensors</c> shards (e.g. <c>model-00001-of-00002.safetensors</c>)
    /// and exposes them as a single merged view. Tensor names must be unique across shards.
    /// </summary>
    public static SafetensorsFile FromFiles(params string[] paths)
    {
        if (paths is null) throw new ArgumentNullException(nameof(paths));
        if (paths.Length == 0)
            throw new ArgumentException("At least one path is required.", nameof(paths));
        if (paths.Length == 1) return FromFile(paths[0]);

        var entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        var names = new List<string>();
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (string path in paths)
        {
            SafetensorsFile shard = FromFile(path);
            foreach (string name in shard._names)
            {
                if (!entries.TryAdd(name, shard._entries[name]))
                    throw new ModelSharpException(
                        $"Tensor '{name}' appears in more than one safetensors shard.");
                names.Add(name);
            }
            foreach (KeyValuePair<string, string> kv in shard._metadata)
                metadata[kv.Key] = kv.Value;
        }

        return new SafetensorsFile(entries, names, metadata);
    }

    /// <summary>Reads and parses a <c>.safetensors</c> payload from a stream (read to the end).</summary>
    public static SafetensorsFile FromStream(Stream stream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));

        // Fast path: a publicly-backed MemoryStream lets us avoid the extra copy.
        if (stream is MemoryStream ms && ms.TryGetBuffer(out ArraySegment<byte> seg))
            return FromBytes(new ReadOnlyMemory<byte>(seg.Array!, seg.Offset, seg.Count));

        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return FromBytes(buffer.ToArray());
    }

    /// <summary>
    /// Parses a complete <c>.safetensors</c> payload held in memory. The buffer is retained;
    /// tensor accessors slice into it directly, so it must remain valid for the lifetime of
    /// the returned <see cref="SafetensorsFile"/>.
    /// </summary>
    public static SafetensorsFile FromBytes(ReadOnlyMemory<byte> data)
    {
        if (data.Length < HeaderSizePrefix)
            throw new ModelSharpException(
                $"Truncated safetensors buffer: need at least {HeaderSizePrefix} bytes for the " +
                $"header-length prefix, got {data.Length}.");

        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(data.Span);
        if (headerLen > (ulong)(data.Length - HeaderSizePrefix))
            throw new ModelSharpException(
                $"Truncated safetensors buffer: header declares {headerLen} bytes but only " +
                $"{data.Length - HeaderSizePrefix} bytes follow the size prefix.");

        int headerLenInt = checked((int)headerLen);
        ReadOnlyMemory<byte> headerJson = data.Slice(HeaderSizePrefix, headerLenInt);
        ReadOnlyMemory<byte> dataSection = data.Slice(HeaderSizePrefix + headerLenInt);

        return Parse(headerJson, dataSection);
    }

    // -------------------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------------------

    /// <summary>Returns <c>true</c> if a tensor with the given name is present.</summary>
    public bool Contains(string name) => name is not null && _entries.ContainsKey(name);

    /// <summary>Returns the header info (dtype + shape + byte length) for a tensor.</summary>
    /// <exception cref="ModelSharpException">No tensor with that name exists.</exception>
    public SafetensorsTensorInfo GetInfo(string name) => GetEntry(name).Info;

    /// <summary>Tries to get the header info for a tensor without throwing.</summary>
    public bool TryGetInfo(string name, out SafetensorsTensorInfo info)
    {
        if (name is not null && _entries.TryGetValue(name, out Entry e))
        {
            info = e.Info;
            return true;
        }
        info = null!;
        return false;
    }

    // -------------------------------------------------------------------------------------
    // Tensor materialization
    // -------------------------------------------------------------------------------------

    /// <summary>
    /// Materializes a tensor in a sensible managed dtype. The low-precision float weights that
    /// dominate LLMs — <see cref="SafetensorsDtype.Float16"/> and <see cref="SafetensorsDtype.BFloat16"/> —
    /// are decoded to <see cref="Tensor{Single}"/>; <see cref="SafetensorsDtype.Int16"/> is widened
    /// to <see cref="Tensor{Int32}"/> (there is no 16-bit managed tensor). All other dtypes map
    /// directly: F32→float, F64→double, I64→long, I32→int, I8→sbyte, U8→byte, BOOL→bool.
    /// </summary>
    /// <exception cref="ModelSharpException">No tensor with that name exists.</exception>
    public Tensor GetTensor(string name)
    {
        Entry e = GetEntry(name);
        ReadOnlySpan<byte> bytes = e.Bytes.Span;
        TensorShape shape = e.Info.Shape;

        return e.Info.Dtype switch
        {
            SafetensorsDtype.Float32 => Reinterpret<float>(bytes, shape),
            SafetensorsDtype.Float64 => Reinterpret<double>(bytes, shape),
            SafetensorsDtype.Int64 => Reinterpret<long>(bytes, shape),
            SafetensorsDtype.Int32 => Reinterpret<int>(bytes, shape),
            SafetensorsDtype.Int16 => WidenInt16(bytes, shape),
            SafetensorsDtype.Int8 => Reinterpret<sbyte>(bytes, shape),
            SafetensorsDtype.UInt8 => Reinterpret<byte>(bytes, shape),
            SafetensorsDtype.Bool => ReadBool(bytes, shape),
            SafetensorsDtype.Float16 => DecodeFloat16(bytes, shape),
            SafetensorsDtype.BFloat16 => DecodeBFloat16(bytes, shape),
            _ => throw new ModelSharpException($"Unsupported dtype for tensor '{name}'."),
        };
    }

    /// <summary>
    /// Materializes a tensor preserving its on-disk bit pattern wherever a matching managed type
    /// exists. Differs from <see cref="GetTensor"/> only for the 16-bit types: F16/BF16 are
    /// returned as their raw 16-bit patterns in a <see cref="Tensor{UInt16}"/>, and I16 as a
    /// <see cref="Tensor{Int16}"/> — letting callers decode or requantize without precision loss.
    /// (Those carry <see cref="ElementType.Unknown"/> since ModelSharp has no native 16-bit dtype.)
    /// </summary>
    /// <exception cref="ModelSharpException">No tensor with that name exists.</exception>
    public Tensor GetTensorRaw(string name)
    {
        Entry e = GetEntry(name);
        ReadOnlySpan<byte> bytes = e.Bytes.Span;
        TensorShape shape = e.Info.Shape;

        return e.Info.Dtype switch
        {
            SafetensorsDtype.Float32 => Reinterpret<float>(bytes, shape),
            SafetensorsDtype.Float64 => Reinterpret<double>(bytes, shape),
            SafetensorsDtype.Int64 => Reinterpret<long>(bytes, shape),
            SafetensorsDtype.Int32 => Reinterpret<int>(bytes, shape),
            SafetensorsDtype.Int16 => Reinterpret<short>(bytes, shape),
            SafetensorsDtype.Int8 => Reinterpret<sbyte>(bytes, shape),
            SafetensorsDtype.UInt8 => Reinterpret<byte>(bytes, shape),
            SafetensorsDtype.Bool => ReadBool(bytes, shape),
            SafetensorsDtype.Float16 => Reinterpret<ushort>(bytes, shape),
            SafetensorsDtype.BFloat16 => Reinterpret<ushort>(bytes, shape),
            _ => throw new ModelSharpException($"Unsupported dtype for tensor '{name}'."),
        };
    }

    // -------------------------------------------------------------------------------------
    // Header parsing
    // -------------------------------------------------------------------------------------

    private static SafetensorsFile Parse(ReadOnlyMemory<byte> headerJson, ReadOnlyMemory<byte> dataSection)
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(headerJson);
        }
        catch (JsonException ex)
        {
            throw new ModelSharpException("Safetensors header is not valid JSON.", ex);
        }

        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                throw new ModelSharpException("Safetensors header JSON must be an object.");

            var entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
            var names = new List<string>();
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            long dataLen = dataSection.Length;

            foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Name == MetadataKey)
                {
                    ReadMetadata(prop.Value, metadata);
                    continue;
                }

                Entry entry = ReadEntry(prop.Name, prop.Value, dataSection, dataLen);
                if (!entries.TryAdd(prop.Name, entry))
                    throw new ModelSharpException($"Duplicate tensor name '{prop.Name}' in safetensors header.");
                names.Add(prop.Name);
            }

            return new SafetensorsFile(entries, names, metadata);
        }
    }

    private static void ReadMetadata(JsonElement value, Dictionary<string, string> metadata)
    {
        if (value.ValueKind != JsonValueKind.Object)
            throw new ModelSharpException("Safetensors '__metadata__' must be a JSON object of string values.");

        foreach (JsonProperty p in value.EnumerateObject())
        {
            string v = p.Value.ValueKind == JsonValueKind.String
                ? p.Value.GetString()!
                : p.Value.GetRawText();
            metadata[p.Name] = v;
        }
    }

    private static Entry ReadEntry(string name, JsonElement value, ReadOnlyMemory<byte> dataSection, long dataLen)
    {
        if (value.ValueKind != JsonValueKind.Object)
            throw new ModelSharpException($"Safetensors entry '{name}' must be a JSON object.");

        if (!value.TryGetProperty("dtype", out JsonElement dtypeEl) || dtypeEl.ValueKind != JsonValueKind.String)
            throw new ModelSharpException($"Safetensors entry '{name}' is missing a string 'dtype'.");
        if (!value.TryGetProperty("shape", out JsonElement shapeEl) || shapeEl.ValueKind != JsonValueKind.Array)
            throw new ModelSharpException($"Safetensors entry '{name}' is missing an array 'shape'.");
        if (!value.TryGetProperty("data_offsets", out JsonElement offEl) || offEl.ValueKind != JsonValueKind.Array)
            throw new ModelSharpException($"Safetensors entry '{name}' is missing an array 'data_offsets'.");

        SafetensorsDtype dtype = SafetensorsDtypeInfo.Parse(dtypeEl.GetString()!, name);

        int rank = shapeEl.GetArrayLength();
        var dims = new int[rank];
        int i = 0;
        foreach (JsonElement d in shapeEl.EnumerateArray())
        {
            if (d.ValueKind != JsonValueKind.Number || !d.TryGetInt32(out int dim) || dim < 0)
                throw new ModelSharpException($"Safetensors entry '{name}' has an invalid shape dimension.");
            dims[i++] = dim;
        }
        var shape = new TensorShape(dims);

        if (offEl.GetArrayLength() != 2)
            throw new ModelSharpException($"Safetensors entry '{name}' must have exactly two 'data_offsets'.");

        JsonElement beginEl = offEl[0];
        JsonElement endEl = offEl[1];
        if (beginEl.ValueKind != JsonValueKind.Number || endEl.ValueKind != JsonValueKind.Number ||
            !beginEl.TryGetInt64(out long begin) || !endEl.TryGetInt64(out long end))
            throw new ModelSharpException($"Safetensors entry '{name}' has non-integer 'data_offsets'.");

        if (begin < 0 || end < begin || end > dataLen)
            throw new ModelSharpException(
                $"Safetensors entry '{name}' data_offsets [{begin}, {end}] are out of range " +
                $"(data section is {dataLen} bytes).");

        long actual = end - begin;
        long expected = checked(shape.Length * SafetensorsDtypeInfo.ByteSize(dtype));
        if (actual != expected)
            throw new ModelSharpException(
                $"Safetensors entry '{name}' byte length {actual} does not match shape {shape} " +
                $"of {dtype} ({expected} bytes).");

        ReadOnlyMemory<byte> slice = dataSection.Slice((int)begin, (int)actual);
        var info = new SafetensorsTensorInfo(name, dtype, shape, actual);
        return new Entry(info, slice);
    }

    // -------------------------------------------------------------------------------------
    // Decoding helpers
    // -------------------------------------------------------------------------------------

    private Entry GetEntry(string name)
    {
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (!_entries.TryGetValue(name, out Entry e))
            throw new ModelSharpException($"Safetensors file does not contain a tensor named '{name}'.");
        return e;
    }

    /// <summary>Validated single-copy reinterpret of a little-endian byte slice into a <c>T[]</c>.</summary>
    private static Tensor<T> Reinterpret<T>(ReadOnlySpan<byte> bytes, TensorShape shape) where T : unmanaged
    {
        if (Unsafe.SizeOf<T>() > 1) EnsureLittleEndian();
        int count = checked((int)shape.Length);
        var data = new T[count];
        if (count > 0)
            MemoryMarshal.Cast<byte, T>(bytes).Slice(0, count).CopyTo(data);
        return new Tensor<T>(shape, data);
    }

    private static Tensor<bool> ReadBool(ReadOnlySpan<byte> bytes, TensorShape shape)
    {
        int count = checked((int)shape.Length);
        var data = new bool[count];
        for (int i = 0; i < count; i++) data[i] = bytes[i] != 0;
        return new Tensor<bool>(shape, data);
    }

    private static Tensor<int> WidenInt16(ReadOnlySpan<byte> bytes, TensorShape shape)
    {
        EnsureLittleEndian();
        int count = checked((int)shape.Length);
        var data = new int[count];
        ReadOnlySpan<short> src = MemoryMarshal.Cast<byte, short>(bytes);
        for (int i = 0; i < count; i++) data[i] = src[i];
        return new Tensor<int>(shape, data);
    }

    private static Tensor<float> DecodeFloat16(ReadOnlySpan<byte> bytes, TensorShape shape)
    {
        EnsureLittleEndian();
        int count = checked((int)shape.Length);
        var data = new float[count];
        ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<byte, ushort>(bytes);
        for (int i = 0; i < count; i++)
            data[i] = (float)BitConverter.UInt16BitsToHalf(bits[i]);
        return new Tensor<float>(shape, data);
    }

    private static Tensor<float> DecodeBFloat16(ReadOnlySpan<byte> bytes, TensorShape shape)
    {
        EnsureLittleEndian();
        int count = checked((int)shape.Length);
        var data = new float[count];
        ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<byte, ushort>(bytes);
        // bf16 is the upper 16 bits of a float32; widen back by shifting into the high half.
        for (int i = 0; i < count; i++)
            data[i] = BitConverter.UInt32BitsToSingle((uint)bits[i] << 16);
        return new Tensor<float>(shape, data);
    }

    private static void EnsureLittleEndian()
    {
        if (!BitConverter.IsLittleEndian)
            throw new ModelSharpException(
                "Safetensors data is little-endian; reading multi-byte tensors on a " +
                "big-endian host is not supported.");
    }

    /// <summary>A parsed header entry plus its (already validated) slice of the data section.</summary>
    private readonly struct Entry
    {
        public Entry(SafetensorsTensorInfo info, ReadOnlyMemory<byte> bytes)
        {
            Info = info;
            Bytes = bytes;
        }

        public SafetensorsTensorInfo Info { get; }

        public ReadOnlyMemory<byte> Bytes { get; }
    }
}
