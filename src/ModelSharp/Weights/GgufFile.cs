using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using ModelSharp.Tensors;

namespace ModelSharp.Weights;

/// <summary>
/// Reader for the <c>GGUF</c> weight format used by llama.cpp. The little-endian on-disk layout is:
/// <list type="number">
///   <item>magic <c>uint32</c> <c>0x46554747</c> ("GGUF"), <c>uint32</c> version (2 or 3).</item>
///   <item><c>uint64</c> tensor count, <c>uint64</c> metadata key/value count.</item>
///   <item>The metadata key/value pairs: each a length-prefixed UTF-8 key, a <c>uint32</c> value
///         type tag, then the value (scalars, strings, or homogeneous arrays).</item>
///   <item>The tensor infos: each a name, <c>uint32</c> dimension count, that many <c>uint64</c>
///         dimensions, a <c>uint32</c> ggml type, and a <c>uint64</c> offset into the data blob.</item>
///   <item>Padding to <c>general.alignment</c> (default 32), then the packed tensor-data blob.</item>
/// </list>
/// The header is parsed eagerly; the data blob is served through a 64-bit-addressable
/// memory-mapped <see cref="IDataSection"/> (the same abstraction the safetensors reader uses),
/// so files larger than 2 GB load without allocating the whole file as a managed array. Native
/// resources are released on <see cref="Dispose"/>.
/// <para>
/// The unquantized ggml types materialize directly: <see cref="GgmlType.F32"/> and
/// <see cref="GgmlType.F16"/> decode to <see cref="Tensor{Single}"/>. The common block-quantized
/// types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 and the Q2_K…Q6_K / Q8_K k-quants) are dequantized to
/// <see cref="Tensor{Single}"/> by <see cref="GgufDequant"/>; the importance-matrix "IQ" families are
/// still surfaced only as raw bytes via <see cref="GetRawTensorBytes"/>, and asking
/// <see cref="GetTensor"/> to materialize one of those throws.
/// </para>
/// </summary>
public sealed class GgufFile : IDisposable
{
    /// <summary>The GGUF magic, "GGUF" as a little-endian <c>uint32</c>.</summary>
    public const uint Magic = 0x46554747;

    /// <summary>Default tensor-data alignment when <c>general.alignment</c> is absent.</summary>
    private const long DefaultAlignment = 32;

    private readonly Dictionary<string, GgufMetadataValue> _metadata;
    private readonly Dictionary<string, GgufTensorInfo> _tensors;
    private readonly List<string> _tensorNames;
    private readonly IDataSection _dataSection;
    private readonly List<IDisposable> _owned;
    private bool _disposed;

    private GgufFile(
        uint version,
        Dictionary<string, GgufMetadataValue> metadata,
        Dictionary<string, GgufTensorInfo> tensors,
        List<string> tensorNames,
        IDataSection dataSection,
        List<IDisposable> owned)
    {
        Version = version;
        _metadata = metadata;
        _tensors = tensors;
        _tensorNames = tensorNames;
        _dataSection = dataSection;
        _owned = owned;
    }

    /// <summary>The GGUF format version (2 or 3).</summary>
    public uint Version { get; }

    /// <summary>The metadata key/value map.</summary>
    public IReadOnlyDictionary<string, GgufMetadataValue> Metadata => _metadata;

    /// <summary>The tensor names, in file order.</summary>
    public IReadOnlyCollection<string> TensorNames => _tensorNames;

    /// <summary>Number of tensors in the file.</summary>
    public int Count => _tensorNames.Count;

    // -------------------------------------------------------------------------------------
    // Factories
    // -------------------------------------------------------------------------------------

    /// <summary>
    /// Memory-maps and parses a <c>.gguf</c> file from a path. The returned instance owns the
    /// mapping and must be disposed.
    /// </summary>
    public static GgufFile FromFile(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));

        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? accessor = null;
        MappedDataSection? section = null;
        try
        {
            long fileLength = new FileInfo(path).Length;
            if (fileLength < 24)
                throw new ModelSharpException(
                    $"Truncated GGUF file '{path}': need at least 24 bytes for the fixed header, " +
                    $"got {fileLength}.");

            // Read the full header (everything up to the data blob) into memory. GGUF headers are
            // small relative to the weight data, so an eager read is cheap and keeps parsing simple.
            // We don't know the header length up front, so read incrementally via a growing cursor
            // over a memory-mapped header view; cap it at the file length.
            mmf = MemoryMappedFile.CreateFromFile(
                path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);

            byte[] all = ReadAllBytes(mmf, fileLength);

            long dataStart = ParseHeader(all, fileLength, out uint version,
                out var metadata, out var tensorOrder, out var tensorRaw);

            long dataLen = fileLength - dataStart;

            accessor = mmf.CreateViewAccessor(0, fileLength, MemoryMappedFileAccess.Read);
            section = new MappedDataSection(accessor, dataStart, dataLen);

            // Validate and finalize tensor infos against the now-known blob length.
            var tensors = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
            foreach (string name in tensorOrder)
            {
                (TensorShape shape, GgmlType type, long offset) = tensorRaw[name];

                if (!GgmlTypeInfo.TryByteLength(type, shape.Length, out long byteLen))
                    throw new ModelSharpException(
                        $"GGUF tensor '{name}' has unsupported or misaligned ggml type {type} " +
                        $"for {shape.Length} elements.");

                if (offset < 0 || offset > dataLen || offset + byteLen > dataLen)
                    throw new ModelSharpException(
                        $"GGUF tensor '{name}' data range [{offset}, {offset + byteLen}) is out of " +
                        $"range (data blob is {dataLen} bytes).");

                tensors[name] = new GgufTensorInfo(name, shape, type, offset, byteLen);
            }

            var owned = new List<IDisposable> { section, mmf };
            var result = new GgufFile(version, metadata, tensors, tensorOrder, section, owned);

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

    private static byte[] ReadAllBytes(MemoryMappedFile mmf, long fileLength)
    {
        // fileLength is bounded by a real file; headers are tiny but the blob can be huge, so
        // only read the header region. We don't yet know its length, so read up to a sane cap
        // and re-read if the parser needs more. In practice GGUF headers are < a few MB; to stay
        // simple and correct we read the whole file when it is small, otherwise a generous window.
        int window = (int)Math.Min(fileLength, 64L * 1024 * 1024);
        var buf = new byte[window];
        using MemoryMappedViewAccessor view = mmf.CreateViewAccessor(0, window, MemoryMappedFileAccess.Read);
        view.ReadArray(0, buf, 0, window);
        return buf;
    }

    // -------------------------------------------------------------------------------------
    // Header parsing
    // -------------------------------------------------------------------------------------

    private static long ParseHeader(
        byte[] all, long fileLength, out uint version,
        out Dictionary<string, GgufMetadataValue> metadata,
        out List<string> tensorOrder,
        out Dictionary<string, (TensorShape Shape, GgmlType Type, long Offset)> tensorRaw)
    {
        EnsureLittleEndian();

        var c = new Cursor(all);

        uint magic = c.ReadUInt32();
        if (magic != Magic)
            throw new ModelSharpException(
                $"Not a GGUF file: magic was 0x{magic:X8}, expected 0x{Magic:X8}.");

        version = c.ReadUInt32();
        if (version != 2 && version != 3)
            throw new ModelSharpException(
                $"Unsupported GGUF version {version}; only versions 2 and 3 are supported.");

        ulong tensorCount = c.ReadUInt64();
        ulong kvCount = c.ReadUInt64();

        metadata = new Dictionary<string, GgufMetadataValue>(StringComparer.Ordinal);
        for (ulong i = 0; i < kvCount; i++)
        {
            string key = c.ReadGgufString();
            GgufMetadataValue value = ReadMetadataValue(ref c, key);
            metadata[key] = value;
        }

        tensorOrder = new List<string>(checked((int)tensorCount));
        tensorRaw = new Dictionary<string, (TensorShape, GgmlType, long)>(StringComparer.Ordinal);

        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = c.ReadGgufString();
            uint nDims = c.ReadUInt32();
            if (nDims > 8)
                throw new ModelSharpException($"GGUF tensor '{name}' declares {nDims} dimensions (>8).");

            var dims = new int[nDims];
            // GGUF stores dimensions fastest-varying first; reverse into row-major order.
            for (int d = (int)nDims - 1; d >= 0; d--)
            {
                ulong dim = c.ReadUInt64();
                if (dim > int.MaxValue)
                    throw new ModelSharpException($"GGUF tensor '{name}' dimension {dim} exceeds Int32 range.");
                dims[d] = (int)dim;
            }
            var shape = new TensorShape(dims);

            uint typeTag = c.ReadUInt32();
            var type = (GgmlType)typeTag;
            if (GgmlTypeInfo.BlockSize(type) == 0)
                throw new ModelSharpException($"GGUF tensor '{name}' has unknown ggml type tag {typeTag}.");

            ulong offset = c.ReadUInt64();
            if (offset > long.MaxValue)
                throw new ModelSharpException($"GGUF tensor '{name}' offset {offset} exceeds Int64 range.");

            if (!tensorRaw.TryAdd(name, (shape, type, (long)offset)))
                throw new ModelSharpException($"Duplicate GGUF tensor name '{name}'.");
            tensorOrder.Add(name);
        }

        // Align the data-blob start to general.alignment (default 32).
        long alignment = DefaultAlignment;
        if (metadata.TryGetValue("general.alignment", out GgufMetadataValue? alignVal) &&
            TryToInt64(alignVal, out long a) && a > 0)
            alignment = a;

        long headerEnd = c.Position;
        long padded = headerEnd;
        long rem = headerEnd % alignment;
        if (rem != 0) padded = headerEnd + (alignment - rem);

        if (padded > fileLength)
            throw new ModelSharpException(
                $"GGUF header end {padded} (aligned to {alignment}) exceeds file length {fileLength}.");

        return padded;
    }

    private static GgufMetadataValue ReadMetadataValue(ref Cursor c, string key)
    {
        var type = (GgufValueType)c.ReadUInt32();
        return type switch
        {
            GgufValueType.UInt8 => new GgufMetadataValue(type, c.ReadByte()),
            GgufValueType.Int8 => new GgufMetadataValue(type, unchecked((sbyte)c.ReadByte())),
            GgufValueType.UInt16 => new GgufMetadataValue(type, c.ReadUInt16()),
            GgufValueType.Int16 => new GgufMetadataValue(type, unchecked((short)c.ReadUInt16())),
            GgufValueType.UInt32 => new GgufMetadataValue(type, c.ReadUInt32()),
            GgufValueType.Int32 => new GgufMetadataValue(type, unchecked((int)c.ReadUInt32())),
            GgufValueType.Float32 => new GgufMetadataValue(type, BitConverter.UInt32BitsToSingle(c.ReadUInt32())),
            GgufValueType.Bool => new GgufMetadataValue(type, c.ReadByte() != 0),
            GgufValueType.String => new GgufMetadataValue(type, c.ReadGgufString()),
            GgufValueType.UInt64 => new GgufMetadataValue(type, c.ReadUInt64()),
            GgufValueType.Int64 => new GgufMetadataValue(type, unchecked((long)c.ReadUInt64())),
            GgufValueType.Float64 => new GgufMetadataValue(type, BitConverter.UInt64BitsToDouble(c.ReadUInt64())),
            GgufValueType.Array => ReadArrayValue(ref c, key),
            _ => throw new ModelSharpException(
                $"GGUF metadata key '{key}' has unknown value type {(uint)type}."),
        };
    }

    private static GgufMetadataValue ReadArrayValue(ref Cursor c, string key)
    {
        var elemType = (GgufValueType)c.ReadUInt32();
        ulong count = c.ReadUInt64();
        int n = checked((int)count);

        switch (elemType)
        {
            case GgufValueType.UInt8:
            {
                var a = new byte[n];
                for (int i = 0; i < n; i++) a[i] = c.ReadByte();
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.Int8:
            {
                var a = new sbyte[n];
                for (int i = 0; i < n; i++) a[i] = unchecked((sbyte)c.ReadByte());
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.UInt16:
            {
                var a = new ushort[n];
                for (int i = 0; i < n; i++) a[i] = c.ReadUInt16();
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.Int16:
            {
                var a = new short[n];
                for (int i = 0; i < n; i++) a[i] = unchecked((short)c.ReadUInt16());
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.UInt32:
            {
                var a = new uint[n];
                for (int i = 0; i < n; i++) a[i] = c.ReadUInt32();
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.Int32:
            {
                var a = new int[n];
                for (int i = 0; i < n; i++) a[i] = unchecked((int)c.ReadUInt32());
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.Float32:
            {
                var a = new float[n];
                for (int i = 0; i < n; i++) a[i] = BitConverter.UInt32BitsToSingle(c.ReadUInt32());
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.Bool:
            {
                var a = new bool[n];
                for (int i = 0; i < n; i++) a[i] = c.ReadByte() != 0;
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.String:
            {
                var a = new string[n];
                for (int i = 0; i < n; i++) a[i] = c.ReadGgufString();
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.UInt64:
            {
                var a = new ulong[n];
                for (int i = 0; i < n; i++) a[i] = c.ReadUInt64();
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.Int64:
            {
                var a = new long[n];
                for (int i = 0; i < n; i++) a[i] = unchecked((long)c.ReadUInt64());
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            case GgufValueType.Float64:
            {
                var a = new double[n];
                for (int i = 0; i < n; i++) a[i] = BitConverter.UInt64BitsToDouble(c.ReadUInt64());
                return new GgufMetadataValue(GgufValueType.Array, a, elemType);
            }
            default:
                throw new ModelSharpException(
                    $"GGUF metadata key '{key}' is an array of unsupported element type {(uint)elemType}.");
        }
    }

    // -------------------------------------------------------------------------------------
    // Metadata accessors
    // -------------------------------------------------------------------------------------

    /// <summary>Returns the string value for a metadata key.</summary>
    /// <exception cref="ModelSharpException">The key is absent or not a string.</exception>
    public string GetMetadataString(string key)
    {
        GgufMetadataValue v = RequireMetadata(key);
        if (v.Type != GgufValueType.String || v.Value is not string s)
            throw new ModelSharpException($"GGUF metadata key '{key}' is {v.Type}, not a string.");
        return s;
    }

    /// <summary>Returns the integer value for a metadata key, widening any GGUF integer type to <see cref="long"/>.</summary>
    /// <exception cref="ModelSharpException">The key is absent or not an integer.</exception>
    public long GetMetadataInt(string key)
    {
        GgufMetadataValue v = RequireMetadata(key);
        if (!TryToInt64(v, out long result))
            throw new ModelSharpException($"GGUF metadata key '{key}' is {v.Type}, not an integer.");
        return result;
    }

    /// <summary>Tries to get a metadata value without throwing.</summary>
    public bool TryGetMetadata(string key, out GgufMetadataValue value)
    {
        if (key is not null && _metadata.TryGetValue(key, out GgufMetadataValue? v))
        {
            value = v;
            return true;
        }
        value = null!;
        return false;
    }

    private GgufMetadataValue RequireMetadata(string key)
    {
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (!_metadata.TryGetValue(key, out GgufMetadataValue? v))
            throw new ModelSharpException($"GGUF file has no metadata key '{key}'.");
        return v;
    }

    private static bool TryToInt64(GgufMetadataValue v, out long result)
    {
        switch (v.Value)
        {
            case byte b: result = b; return true;
            case sbyte sb: result = sb; return true;
            case ushort us: result = us; return true;
            case short s: result = s; return true;
            case uint ui: result = ui; return true;
            case int i: result = i; return true;
            case ulong ul when ul <= long.MaxValue: result = (long)ul; return true;
            case long l: result = l; return true;
            default: result = 0; return false;
        }
    }

    // -------------------------------------------------------------------------------------
    // Tensor access
    // -------------------------------------------------------------------------------------

    /// <summary>Returns <c>true</c> if a tensor with the given name is present.</summary>
    public bool Contains(string name) => name is not null && _tensors.ContainsKey(name);

    /// <summary>Returns the info (name, shape, ggml type, offset) for a tensor.</summary>
    /// <exception cref="ModelSharpException">No tensor with that name exists.</exception>
    public GgufTensorInfo GetInfo(string name) => GetEntry(name);

    /// <summary>Tries to get tensor info without throwing.</summary>
    public bool TryGetInfo(string name, out GgufTensorInfo info)
    {
        if (name is not null && _tensors.TryGetValue(name, out GgufTensorInfo? i))
        {
            info = i;
            return true;
        }
        info = null!;
        return false;
    }

    /// <summary>
    /// Returns a freshly-allocated copy of a tensor's raw on-disk bytes — including for quantized
    /// block types, so a future dequantization kernel can consume them alongside
    /// <see cref="GgufTensorInfo.Type"/>.
    /// </summary>
    /// <exception cref="ModelSharpException">No tensor with that name exists.</exception>
    public byte[] GetRawTensorBytes(string name)
    {
        GgufTensorInfo info = GetEntry(name);
        return _dataSection.ReadBytes(info.Offset, checked((int)info.ByteLength));
    }

    /// <summary>
    /// Materializes a tensor as a managed tensor: <see cref="GgmlType.F32"/> →
    /// <see cref="Tensor{Single}"/>, <see cref="GgmlType.F16"/> → <see cref="Tensor{Single}"/>
    /// (half-decoded), <see cref="GgmlType.F64"/> → <see cref="Tensor{Double}"/>, and the integer
    /// types to their managed equivalents. The ggml block-quantized types supported by
    /// <see cref="GgufDequant"/> — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
    /// and Q8_K — are dequantized to a <see cref="Tensor{Single}"/> of the tensor's shape. The
    /// importance-matrix "IQ" quant families are not yet implemented and still throw.
    /// </summary>
    /// <exception cref="ModelSharpException">
    /// No tensor with that name exists, or the tensor is a quantized type that ModelSharp does not
    /// yet dequantize (use <see cref="GetRawTensorBytes"/> and dequantize separately).
    /// </exception>
    public Tensor GetTensor(string name)
    {
        GgufTensorInfo info = GetEntry(name);

        if (GgmlTypeInfo.IsQuantized(info.Type))
        {
            if (!GgufDequant.IsSupported(info.Type))
                throw new ModelSharpException(
                    $"GGUF tensor '{name}' is quantized type {info.Type}; ModelSharp does not yet " +
                    "dequantize it. Supported quantized types are Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, " +
                    "Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K. Use GetRawTensorBytes and the " +
                    "ggml type to dequantize separately.");

            byte[] raw = GetRawTensorBytes(name);
            float[] dequant = GgufDequant.Dequantize(raw, info.Type, info.Shape.Length);
            return new Tensor<float>(info.Shape, dequant);
        }

        byte[] bytes = GetRawTensorBytes(name);
        TensorShape shape = info.Shape;

        return info.Type switch
        {
            GgmlType.F32 => Reinterpret<float>(bytes, shape),
            GgmlType.F64 => Reinterpret<double>(bytes, shape),
            GgmlType.F16 => DecodeFloat16(bytes, shape),
            GgmlType.BF16 => DecodeBFloat16(bytes, shape),
            GgmlType.I8 => Reinterpret<sbyte>(bytes, shape),
            GgmlType.I16 => WidenInt16(bytes, shape),
            GgmlType.I32 => Reinterpret<int>(bytes, shape),
            GgmlType.I64 => Reinterpret<long>(bytes, shape),
            _ => throw new ModelSharpException(
                $"GGUF tensor '{name}' has unsupported ggml type {info.Type} for materialization."),
        };
    }

    private GgufTensorInfo GetEntry(string name)
    {
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (_disposed) throw new ObjectDisposedException(nameof(GgufFile));
        if (!_tensors.TryGetValue(name, out GgufTensorInfo? info))
            throw new ModelSharpException($"GGUF file does not contain a tensor named '{name}'.");
        return info;
    }

    // -------------------------------------------------------------------------------------
    // Decoding helpers (little-endian; mirror SafetensorsFile)
    // -------------------------------------------------------------------------------------

    private static Tensor<T> Reinterpret<T>(ReadOnlySpan<byte> bytes, TensorShape shape) where T : unmanaged
    {
        EnsureLittleEndian();
        int count = checked((int)shape.Length);
        var data = new T[count];
        if (count > 0)
            MemoryMarshal.Cast<byte, T>(bytes).Slice(0, count).CopyTo(data);
        return new Tensor<T>(shape, data);
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
        for (int i = 0; i < count; i++)
            data[i] = BitConverter.UInt32BitsToSingle((uint)bits[i] << 16);
        return new Tensor<float>(shape, data);
    }

    private static void EnsureLittleEndian()
    {
        if (!BitConverter.IsLittleEndian)
            throw new ModelSharpException(
                "GGUF data is little-endian; reading multi-byte values on a big-endian host is not supported.");
    }

    // -------------------------------------------------------------------------------------
    // Disposal
    // -------------------------------------------------------------------------------------

    /// <summary>Releases the memory-mapped file resources held by this instance.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (IDisposable d in _owned) d.Dispose();
        _owned.Clear();
    }

    /// <summary>
    /// A forward-only little-endian cursor over the eagerly-read header bytes. Bounds are checked
    /// so a truncated/malformed header throws a clear <see cref="ModelSharpException"/> rather than
    /// reading past the buffer.
    /// </summary>
    private struct Cursor
    {
        private readonly byte[] _buffer;
        private int _pos;

        public Cursor(byte[] buffer)
        {
            _buffer = buffer;
            _pos = 0;
        }

        public long Position => _pos;

        private void Require(int n)
        {
            if (n < 0 || _pos + n > _buffer.Length)
                throw new ModelSharpException(
                    "Truncated GGUF header: ran out of bytes while reading the metadata/tensor table " +
                    "(the header may be larger than the read window or the file is corrupt).");
        }

        public byte ReadByte()
        {
            Require(1);
            return _buffer[_pos++];
        }

        public ushort ReadUInt16()
        {
            Require(2);
            ushort v = BinaryPrimitives.ReadUInt16LittleEndian(_buffer.AsSpan(_pos, 2));
            _pos += 2;
            return v;
        }

        public uint ReadUInt32()
        {
            Require(4);
            uint v = BinaryPrimitives.ReadUInt32LittleEndian(_buffer.AsSpan(_pos, 4));
            _pos += 4;
            return v;
        }

        public ulong ReadUInt64()
        {
            Require(8);
            ulong v = BinaryPrimitives.ReadUInt64LittleEndian(_buffer.AsSpan(_pos, 8));
            _pos += 8;
            return v;
        }

        public string ReadGgufString()
        {
            ulong len = ReadUInt64();
            if (len > int.MaxValue)
                throw new ModelSharpException($"GGUF string length {len} exceeds Int32 range.");
            int n = (int)len;
            Require(n);
            string s = Encoding.UTF8.GetString(_buffer, _pos, n);
            _pos += n;
            return s;
        }
    }
}
