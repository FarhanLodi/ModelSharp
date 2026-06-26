using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
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
/// <b>Large files:</b> <see cref="FromFile(string)"/> memory-maps the file and serves tensor
/// bytes through an internal 64-bit-addressable data section, so a single shard larger than
/// 2 GB loads without allocating the whole file as a managed array. Because that mapping is a
/// native resource, <see cref="SafetensorsFile"/> implements <see cref="IDisposable"/>; the
/// in-memory <see cref="FromBytes"/>/<see cref="FromStream"/> paths have a no-op
/// <see cref="Dispose"/>.
/// </para>
/// <para>
/// <b>Endianness:</b> the data section is little-endian. On a little-endian host (the .NET
/// norm) a direct reinterpret/copy is correct; reading multi-byte tensors on a big-endian
/// host is rejected rather than producing byte-swapped garbage.
/// </para>
/// </summary>
public sealed class SafetensorsFile : IDisposable
{
    /// <summary>Size of the leading little-endian header-length prefix.</summary>
    private const int HeaderSizePrefix = 8;

    /// <summary>Reserved header key carrying free-form string metadata, not a tensor.</summary>
    private const string MetadataKey = "__metadata__";

    private readonly Dictionary<string, Entry> _entries;
    private readonly List<string> _names;
    private readonly Dictionary<string, string> _metadata;

    /// <summary>
    /// Native resources (memory-mapped files and their data sections) that must be kept alive
    /// for the lifetime of this instance and released on <see cref="Dispose"/>. Empty for
    /// purely in-memory instances. May contain the disposables of several shards in a merged view.
    /// </summary>
    private readonly List<IDisposable> _owned;

    private bool _disposed;

    private SafetensorsFile(
        Dictionary<string, Entry> entries,
        List<string> names,
        Dictionary<string, string> metadata,
        List<IDisposable> owned)
    {
        _entries = entries;
        _names = names;
        _metadata = metadata;
        _owned = owned;
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

    /// <summary>
    /// Memory-maps and parses a <c>.safetensors</c> file from a path. The returned instance owns
    /// the mapping and must be disposed; tensors larger than 2 GB total file size are supported.
    /// </summary>
    public static SafetensorsFile FromFile(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));

        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? accessor = null;
        MappedDataSection? section = null;
        try
        {
            long fileLength = new FileInfo(path).Length;
            if (fileLength < HeaderSizePrefix)
                throw new ModelSharpException(
                    $"Truncated safetensors file '{path}': need at least {HeaderSizePrefix} bytes " +
                    $"for the header-length prefix, got {fileLength}.");

            mmf = MemoryMappedFile.CreateFromFile(
                path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);

            // Read the 8-byte length prefix and the header JSON eagerly.
            ulong headerLen;
            using (var prefix = mmf.CreateViewAccessor(0, HeaderSizePrefix, MemoryMappedFileAccess.Read))
            {
                Span<byte> tmp = stackalloc byte[HeaderSizePrefix];
                for (int i = 0; i < HeaderSizePrefix; i++) tmp[i] = prefix.ReadByte(i);
                headerLen = BinaryPrimitives.ReadUInt64LittleEndian(tmp);
            }

            long afterHeader = (long)HeaderSizePrefix + (long)headerLen;
            if (headerLen > (ulong)(fileLength - HeaderSizePrefix))
                throw new ModelSharpException(
                    $"Truncated safetensors file '{path}': header declares {headerLen} bytes but " +
                    $"only {fileLength - HeaderSizePrefix} bytes follow the size prefix.");

            int headerLenInt = checked((int)headerLen);
            var headerBytes = new byte[headerLenInt];
            using (var headerView = mmf.CreateViewAccessor(HeaderSizePrefix, headerLenInt, MemoryMappedFileAccess.Read))
            {
                if (headerLenInt > 0) headerView.ReadArray(0, headerBytes, 0, headerLenInt);
            }

            long dataLen = fileLength - afterHeader;

            // A single view covering the whole file, from which the data section reads with
            // 64-bit offsets relative to the start of the data blob.
            accessor = mmf.CreateViewAccessor(0, fileLength, MemoryMappedFileAccess.Read);
            section = new MappedDataSection(accessor, afterHeader, dataLen);

            var owned = new List<IDisposable> { section, mmf };
            SafetensorsFile result = Parse(headerBytes, section, owned);

            // Ownership transferred into result; clear locals so the catch/finally below
            // doesn't double-dispose.
            mmf = null;
            accessor = null;
            section = null;
            return result;
        }
        catch
        {
            section?.Dispose();
            accessor?.Dispose();
            mmf?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Memory-maps one or more <c>.safetensors</c> shards (e.g. <c>model-00001-of-00002.safetensors</c>)
    /// and exposes them as a single merged, disposable view. Tensor names must be unique across shards;
    /// disposing the merged view releases every shard's mapping.
    /// </summary>
    public static SafetensorsFile FromFiles(params string[] paths)
    {
        if (paths is null) throw new ArgumentNullException(nameof(paths));
        if (paths.Length == 0)
            throw new ArgumentException("At least one path is required.", nameof(paths));
        if (paths.Length == 1) return FromFile(paths[0]);

        return Merge(paths);
    }

    /// <summary>
    /// Loads a HuggingFace sharded-checkpoint index (a <c>*.index.json</c> with a
    /// <c>"weight_map"</c> of <c>tensorName → shardFileName</c> and optional <c>"metadata"</c>),
    /// resolves the distinct shard files relative to the index's directory, memory-maps them, and
    /// exposes the merged view. Every tensor named in the weight map must be present after loading.
    /// </summary>
    public static SafetensorsFile FromIndex(string indexJsonPath)
    {
        if (indexJsonPath is null) throw new ArgumentNullException(nameof(indexJsonPath));

        byte[] indexBytes = File.ReadAllBytes(indexJsonPath);
        string? directory = Path.GetDirectoryName(Path.GetFullPath(indexJsonPath));

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(indexBytes);
        }
        catch (JsonException ex)
        {
            throw new ModelSharpException($"Safetensors index '{indexJsonPath}' is not valid JSON.", ex);
        }

        List<string> shardPaths;
        HashSet<string> mappedTensors;
        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                throw new ModelSharpException($"Safetensors index '{indexJsonPath}' must be a JSON object.");

            if (!doc.RootElement.TryGetProperty("weight_map", out JsonElement weightMap) ||
                weightMap.ValueKind != JsonValueKind.Object)
                throw new ModelSharpException(
                    $"Safetensors index '{indexJsonPath}' is missing an object 'weight_map'.");

            mappedTensors = new HashSet<string>(StringComparer.Ordinal);
            var distinctShards = new List<string>();
            var seenShards = new HashSet<string>(StringComparer.Ordinal);

            foreach (JsonProperty kv in weightMap.EnumerateObject())
            {
                if (kv.Value.ValueKind != JsonValueKind.String)
                    throw new ModelSharpException(
                        $"Safetensors index '{indexJsonPath}' weight_map entry '{kv.Name}' must be a string.");

                mappedTensors.Add(kv.Name);
                string shardFile = kv.Value.GetString()!;
                if (seenShards.Add(shardFile))
                    distinctShards.Add(shardFile);
            }

            shardPaths = new List<string>(distinctShards.Count);
            foreach (string shardFile in distinctShards)
                shardPaths.Add(directory is null ? shardFile : Path.Combine(directory, shardFile));
        }

        if (shardPaths.Count == 0)
            throw new ModelSharpException($"Safetensors index '{indexJsonPath}' weight_map is empty.");

        SafetensorsFile merged = Merge(shardPaths.ToArray());
        try
        {
            foreach (string tensor in mappedTensors)
            {
                if (!merged._entries.ContainsKey(tensor))
                    throw new ModelSharpException(
                        $"Safetensors index '{indexJsonPath}' references tensor '{tensor}', " +
                        $"but it was not found in the resolved shards.");
            }
        }
        catch
        {
            merged.Dispose();
            throw;
        }

        return merged;
    }

    /// <summary>Memory-maps several shards and merges their entries into one disposable view.</summary>
    private static SafetensorsFile Merge(string[] paths)
    {
        var entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        var names = new List<string>();
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        var owned = new List<IDisposable>();
        var shards = new List<SafetensorsFile>();

        try
        {
            foreach (string path in paths)
            {
                SafetensorsFile shard = FromFile(path);
                shards.Add(shard);

                foreach (string name in shard._names)
                {
                    if (!entries.TryAdd(name, shard._entries[name]))
                        throw new ModelSharpException(
                            $"Tensor '{name}' appears in more than one safetensors shard.");
                    names.Add(name);
                }
                foreach (KeyValuePair<string, string> kv in shard._metadata)
                    metadata[kv.Key] = kv.Value;

                // Adopt the shard's native resources into the merged view, then neutralize the
                // shard wrapper so disposing it does not release the mapping the merged view uses.
                owned.AddRange(shard._owned);
                shard._owned.Clear();
            }

            return new SafetensorsFile(entries, names, metadata, owned);
        }
        catch
        {
            foreach (IDisposable d in owned) d.Dispose();
            throw;
        }
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
    /// tensor accessors copy out of it on demand, so it must remain valid for the lifetime of
    /// the returned <see cref="SafetensorsFile"/>. The returned instance owns no native
    /// resources and its <see cref="Dispose"/> is a no-op.
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

        var section = new MemoryDataSection(dataSection);
        // MemoryDataSection.Dispose is a no-op, so no native ownership is required.
        return Parse(headerJson.ToArray(), section, new List<IDisposable>());
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
        TensorShape shape = e.Info.Shape;

        return e.Info.Dtype switch
        {
            SafetensorsDtype.Float32 => Reinterpret<float>(e, shape),
            SafetensorsDtype.Float64 => Reinterpret<double>(e, shape),
            SafetensorsDtype.Int64 => Reinterpret<long>(e, shape),
            SafetensorsDtype.Int32 => Reinterpret<int>(e, shape),
            SafetensorsDtype.Int16 => WidenInt16(e, shape),
            SafetensorsDtype.Int8 => Reinterpret<sbyte>(e, shape),
            SafetensorsDtype.UInt8 => Reinterpret<byte>(e, shape),
            SafetensorsDtype.Bool => ReadBool(e, shape),
            SafetensorsDtype.Float16 => DecodeFloat16(e, shape),
            SafetensorsDtype.BFloat16 => DecodeBFloat16(e, shape),
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
        TensorShape shape = e.Info.Shape;

        return e.Info.Dtype switch
        {
            SafetensorsDtype.Float32 => Reinterpret<float>(e, shape),
            SafetensorsDtype.Float64 => Reinterpret<double>(e, shape),
            SafetensorsDtype.Int64 => Reinterpret<long>(e, shape),
            SafetensorsDtype.Int32 => Reinterpret<int>(e, shape),
            SafetensorsDtype.Int16 => Reinterpret<short>(e, shape),
            SafetensorsDtype.Int8 => Reinterpret<sbyte>(e, shape),
            SafetensorsDtype.UInt8 => Reinterpret<byte>(e, shape),
            SafetensorsDtype.Bool => ReadBool(e, shape),
            SafetensorsDtype.Float16 => Reinterpret<ushort>(e, shape),
            SafetensorsDtype.BFloat16 => Reinterpret<ushort>(e, shape),
            _ => throw new ModelSharpException($"Unsupported dtype for tensor '{name}'."),
        };
    }

    // -------------------------------------------------------------------------------------
    // Header parsing
    // -------------------------------------------------------------------------------------

    private static SafetensorsFile Parse(
        ReadOnlyMemory<byte> headerJson, IDataSection dataSection, List<IDisposable> owned)
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

            return new SafetensorsFile(entries, names, metadata, owned);
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

    private static Entry ReadEntry(string name, JsonElement value, IDataSection dataSection, long dataLen)
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

        var info = new SafetensorsTensorInfo(name, dtype, shape, actual);
        return new Entry(info, dataSection, begin, actual);
    }

    // -------------------------------------------------------------------------------------
    // Decoding helpers
    // -------------------------------------------------------------------------------------

    private Entry GetEntry(string name)
    {
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (_disposed) throw new ObjectDisposedException(nameof(SafetensorsFile));
        if (!_entries.TryGetValue(name, out Entry e))
            throw new ModelSharpException($"Safetensors file does not contain a tensor named '{name}'.");
        return e;
    }

    /// <summary>Reads an entry's raw bytes out of its data section into a freshly allocated array.</summary>
    private static byte[] ReadRawBytes(in Entry e) =>
        e.Section.ReadBytes(e.Begin, checked((int)e.Length));

    /// <summary>Validated single-copy reinterpret of a little-endian byte range into a <c>T[]</c>.</summary>
    private static Tensor<T> Reinterpret<T>(in Entry e, TensorShape shape) where T : unmanaged
    {
        if (Unsafe.SizeOf<T>() > 1) EnsureLittleEndian();
        int count = checked((int)shape.Length);
        byte[] bytes = ReadRawBytes(e);
        var data = new T[count];
        if (count > 0)
            MemoryMarshal.Cast<byte, T>(bytes).Slice(0, count).CopyTo(data);
        return new Tensor<T>(shape, data);
    }

    private static Tensor<bool> ReadBool(in Entry e, TensorShape shape)
    {
        int count = checked((int)shape.Length);
        byte[] bytes = ReadRawBytes(e);
        var data = new bool[count];
        for (int i = 0; i < count; i++) data[i] = bytes[i] != 0;
        return new Tensor<bool>(shape, data);
    }

    private static Tensor<int> WidenInt16(in Entry e, TensorShape shape)
    {
        EnsureLittleEndian();
        int count = checked((int)shape.Length);
        byte[] bytes = ReadRawBytes(e);
        var data = new int[count];
        ReadOnlySpan<short> src = MemoryMarshal.Cast<byte, short>(bytes);
        for (int i = 0; i < count; i++) data[i] = src[i];
        return new Tensor<int>(shape, data);
    }

    private static Tensor<float> DecodeFloat16(in Entry e, TensorShape shape)
    {
        EnsureLittleEndian();
        int count = checked((int)shape.Length);
        byte[] bytes = ReadRawBytes(e);
        var data = new float[count];
        ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<byte, ushort>(bytes);
        for (int i = 0; i < count; i++)
            data[i] = (float)BitConverter.UInt16BitsToHalf(bits[i]);
        return new Tensor<float>(shape, data);
    }

    private static Tensor<float> DecodeBFloat16(in Entry e, TensorShape shape)
    {
        EnsureLittleEndian();
        int count = checked((int)shape.Length);
        byte[] bytes = ReadRawBytes(e);
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

    // -------------------------------------------------------------------------------------
    // Disposal
    // -------------------------------------------------------------------------------------

    /// <summary>
    /// Releases any memory-mapped file resources held by this instance (and, for a merged view,
    /// all of its shards). In-memory instances created via <see cref="FromBytes"/> /
    /// <see cref="FromStream"/> own nothing, so disposing them is a no-op. After disposal, tensor
    /// accessors throw <see cref="ObjectDisposedException"/>.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (IDisposable d in _owned) d.Dispose();
        _owned.Clear();
    }

    /// <summary>A parsed header entry plus the (validated) location of its bytes within a data section.</summary>
    private readonly struct Entry
    {
        public Entry(SafetensorsTensorInfo info, IDataSection section, long begin, long length)
        {
            Info = info;
            Section = section;
            Begin = begin;
            Length = length;
        }

        public SafetensorsTensorInfo Info { get; }

        /// <summary>The data section this tensor's bytes live in (may differ per shard in a merged view).</summary>
        public IDataSection Section { get; }

        /// <summary>Byte offset of this tensor within <see cref="Section"/>.</summary>
        public long Begin { get; }

        /// <summary>Byte length of this tensor.</summary>
        public long Length { get; }
    }
}
