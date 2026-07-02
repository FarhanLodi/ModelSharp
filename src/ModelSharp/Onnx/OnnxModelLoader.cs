using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Onnx;

/// <summary>
/// Loads an ONNX model (or a standalone TensorProto) into ModelSharp's engine-agnostic
/// <see cref="ModelGraph"/> using a hand-rolled protobuf reader — no native or
/// Google.Protobuf dependency. Phase 1 materializes everything as float32.
/// </summary>
public static class OnnxModelLoader
{
    /// <summary>Loads an ONNX model from a file path.</summary>
    public static ModelGraph LoadModel(string path)
    {
        // External-data initializers (TensorProto.data_location == EXTERNAL) reference a
        // sibling weights file relative to the .onnx file's directory. The resolver memory-maps
        // each referenced file once and reuses the mapping across every tensor in this load.
        string? baseDir = Path.GetDirectoryName(Path.GetFullPath(path));
        using var resolver = new ExternalDataResolver(baseDir);
        return ParseModel(File.ReadAllBytes(path), resolver);
    }

    /// <summary>Parses an ONNX ModelProto from bytes.</summary>
    public static ModelGraph ParseModel(ReadOnlySpan<byte> modelProto)
        => ParseModel(modelProto, resolver: null);

    /// <summary>
    /// Parses an ONNX ModelProto from bytes, resolving external-data initializers via
    /// <paramref name="resolver"/>. When <paramref name="resolver"/> is null (in-memory parse with
    /// no backing file path), any EXTERNAL initializer encountered throws a clear error.
    /// </summary>
    private static ModelGraph ParseModel(ReadOnlySpan<byte> modelProto, ExternalDataResolver? resolver)
    {
        ReadOnlySpan<byte> graphBytes = default;
        bool hasGraph = false;
        Dictionary<string, string>? metadata = null;

        var r = new ProtoReader(modelProto);
        while (r.TryReadTag(out int field, out int wire))
        {
            if (field == 7 && wire == 2) { graphBytes = r.ReadLengthDelimited(); hasGraph = true; }
            // ModelProto.metadata_props (field 14): repeated StringStringEntryProto.
            else if (field == 14 && wire == 2)
            {
                (string key, string value) = ParseMetadataEntry(r.ReadLengthDelimited());
                if (key.Length != 0) (metadata ??= new Dictionary<string, string>())[key] = value;
            }
            else r.SkipField(wire);
        }

        if (!hasGraph) throw new ModelSharpException("ONNX model contains no graph (field 7).");
        return ParseGraph(graphBytes, metadata, resolver);
    }

    /// <summary>Loads a standalone TensorProto (e.g. an ONNX test-data <c>.pb</c>) as a float32 tensor.</summary>
    public static Tensor<float> LoadTensor(string path)
    {
        using var resolver = new ExternalDataResolver(Path.GetDirectoryName(Path.GetFullPath(path)));
        return ParseTensor(File.ReadAllBytes(path), resolver).Tensor.AsFloat();
    }

    /// <summary>Loads a standalone TensorProto preserving its declared dtype.</summary>
    public static Tensor LoadTensorTyped(string path)
    {
        using var resolver = new ExternalDataResolver(Path.GetDirectoryName(Path.GetFullPath(path)));
        return ParseTensor(File.ReadAllBytes(path), resolver).Tensor;
    }

    private static ModelGraph ParseGraph(
        ReadOnlySpan<byte> bytes, Dictionary<string, string>? metadata, ExternalDataResolver? resolver)
    {
        var nodes = new List<GraphNode>();
        var initializers = new Dictionary<string, Tensor>();
        var inputNames = new List<string>();
        var outputNames = new List<string>();

        var r = new ProtoReader(bytes);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2:
                    nodes.Add(ParseNode(r.ReadLengthDelimited(), resolver));
                    break;
                case 5 when wire == 2:
                    (string tn, Tensor tt) = ParseTensor(r.ReadLengthDelimited(), resolver);
                    initializers[tn] = tt;
                    break;
                case 11 when wire == 2:
                    inputNames.Add(ParseValueInfoName(r.ReadLengthDelimited()));
                    break;
                case 12 when wire == 2:
                    outputNames.Add(ParseValueInfoName(r.ReadLengthDelimited()));
                    break;
                default:
                    r.SkipField(wire);
                    break;
            }
        }

        // Real feeds = declared graph inputs that are not initializers (weights/constants).
        List<string> feeds = inputNames.Where(n => !initializers.ContainsKey(n)).ToList();

        return new ModelGraph
        {
            Inputs = feeds,
            Outputs = outputNames,
            Nodes = nodes,
            Initializers = initializers,
            MetadataProps = (IReadOnlyDictionary<string, string>?)metadata ?? new Dictionary<string, string>(),
        };
    }

    /// <summary>Parses a StringStringEntryProto (key = field 1, value = field 2).</summary>
    private static (string Key, string Value) ParseMetadataEntry(ReadOnlySpan<byte> bytes)
    {
        string key = "";
        string value = "";

        var r = new ProtoReader(bytes);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2: key = r.ReadString(); break;
                case 2 when wire == 2: value = r.ReadString(); break;
                default: r.SkipField(wire); break;
            }
        }

        return (key, value);
    }

    private static GraphNode ParseNode(ReadOnlySpan<byte> bytes, ExternalDataResolver? resolver)
    {
        var inputs = new List<string>();
        var outputs = new List<string>();
        string name = "";
        string opType = "";
        var attrs = new Dictionary<string, object>();

        var r = new ProtoReader(bytes);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2: inputs.Add(r.ReadString()); break;
                case 2 when wire == 2: outputs.Add(r.ReadString()); break;
                case 3 when wire == 2: name = r.ReadString(); break;
                case 4 when wire == 2: opType = r.ReadString(); break;
                case 5 when wire == 2:
                    (string an, object? av) = ParseAttribute(r.ReadLengthDelimited(), resolver);
                    if (av is not null) attrs[an] = av;
                    break;
                default: r.SkipField(wire); break;
            }
        }

        return new GraphNode(opType, name.Length == 0 ? opType : name, inputs, outputs, attrs);
    }

    private static (string Name, object? Value) ParseAttribute(
        ReadOnlySpan<byte> bytes, ExternalDataResolver? resolver)
    {
        string name = "";
        long type = 0;
        float? f = null;
        long? i = null;
        string? s = null;
        Tensor? t = null;
        ModelGraph? g = null;
        List<long>? ints = null;
        List<float>? floats = null;
        List<ModelGraph>? graphs = null;

        var r = new ProtoReader(bytes);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 2: name = r.ReadString(); break;
                case 20 when wire == 0: type = r.ReadInt64(); break;
                case 2 when wire == 5: f = r.ReadFloat(); break;
                case 3 when wire == 0: i = r.ReadInt64(); break;
                case 4 when wire == 2: s = Encoding.UTF8.GetString(r.ReadLengthDelimited()); break;
                case 5 when wire == 2: t = ParseTensor(r.ReadLengthDelimited(), resolver).Tensor; break;
                // AttributeProto.g (field 6): single nested GraphProto.
                case 6 when wire == 2: g = ParseGraph(r.ReadLengthDelimited(), metadata: null, resolver); break;
                case 7 when wire == 5: (floats ??= new()).Add(r.ReadFloat()); break;
                case 7 when wire == 2: ReadPackedFloats(r.ReadLengthDelimited(), floats ??= new()); break;
                case 8 when wire == 0: (ints ??= new()).Add(r.ReadInt64()); break;
                case 8 when wire == 2: ReadPackedVarints(r.ReadLengthDelimited(), ints ??= new()); break;
                // AttributeProto.graphs (field 10): repeated GraphProto.
                case 10 when wire == 2: (graphs ??= new()).Add(ParseGraph(r.ReadLengthDelimited(), metadata: null, resolver)); break;
                default: r.SkipField(wire); break;
            }
        }

        // AttributeType: FLOAT=1, INT=2, STRING=3, TENSOR=4, GRAPH=5, FLOATS=6, INTS=7, GRAPHS=10.
        object? value =
            type == 1 ? f :
            type == 2 ? i :
            type == 3 ? s :
            type == 4 ? (object?)t :
            type == 5 ? (object?)g :
            type == 6 ? floats?.ToArray() :
            type == 7 ? ints?.ToArray() :
            type == 10 ? (object?)graphs?.ToArray() :
            // Unknown/undefined type: infer from whichever field was present.
            g is not null ? g :
            graphs is not null ? (object?)graphs.ToArray() :
            ints is not null ? ints.ToArray() :
            floats is not null ? (object?)floats.ToArray() :
            i.HasValue ? i.Value :
            f.HasValue ? f.Value :
            s is not null ? s :
            (object?)t;

        return (name, value);
    }

    // ONNX TensorProto.DataType values we materialize. FLOAT16 (10) initializers are kept as compact
    // Tensor&lt;Half&gt; (2 bytes/element) and widened to float32 on demand — and memoized — at the compute
    // boundary (Tensor.ToFloat32 / GraphContext.Get). BFLOAT16 (16) is decoded to float32 on load.
    private const long DtFloat = 1, DtUint8 = 2, DtInt8 = 3, DtInt32 = 6, DtInt64 = 7, DtBool = 9,
        DtFloat16 = 10, DtBFloat16 = 16;

    private static (string Name, Tensor Tensor) ParseTensor(
        ReadOnlySpan<byte> bytes, ExternalDataResolver? resolver)
    {
        var dims = new List<long>();
        long dataType = 0;
        string name = "";
        ReadOnlySpan<byte> rawData = default;
        bool hasRaw = false;
        List<float>? floatData = null;
        List<long>? int64Data = null;
        List<int>? int32Data = null;

        // External-data fields (only meaningful when dataLocation == 1 / EXTERNAL).
        long dataLocation = 0;
        string? extLocation = null;
        long extOffset = 0;
        long extLength = -1; // -1 => "to end of file".
        bool hasExtLength = false;

        var r = new ProtoReader(bytes);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 0: dims.Add(r.ReadInt64()); break;
                case 1 when wire == 2: ReadPackedVarints(r.ReadLengthDelimited(), dims); break;
                case 2 when wire == 0: dataType = r.ReadInt64(); break;
                case 4 when wire == 5: (floatData ??= new()).Add(r.ReadFloat()); break;
                case 4 when wire == 2: ReadPackedFloats(r.ReadLengthDelimited(), floatData ??= new()); break;
                case 5 when wire == 0: (int32Data ??= new()).Add(r.ReadInt32()); break;
                case 5 when wire == 2: ReadPackedVarintsAsInt(r.ReadLengthDelimited(), int32Data ??= new()); break;
                case 7 when wire == 0: (int64Data ??= new()).Add(r.ReadInt64()); break;
                case 7 when wire == 2: ReadPackedVarints(r.ReadLengthDelimited(), int64Data ??= new()); break;
                case 8 when wire == 2: name = r.ReadString(); break;
                case 9 when wire == 2: rawData = r.ReadLengthDelimited(); hasRaw = true; break;
                // external_data (field 13): repeated StringStringEntryProto (key/value).
                case 13 when wire == 2:
                    ApplyExternalDataEntry(
                        r.ReadLengthDelimited(),
                        ref extLocation, ref extOffset, ref extLength, ref hasExtLength);
                    break;
                // data_location (field 14): 0 = DEFAULT (inline), 1 = EXTERNAL.
                case 14 when wire == 0: dataLocation = r.ReadInt64(); break;
                default: r.SkipField(wire); break;
            }
        }

        int[] shapeDims = dims.Select(d => (int)d).ToArray();
        var shape = new TensorShape(shapeDims);
        int count = checked((int)shape.Length);

        // EXTERNAL: pull the tensor's raw bytes from the sibling data file via the resolver, then
        // feed them through the SAME dtype switch as inline raw_data (identical little-endian
        // MemoryMarshal.Cast decode). Only this tensor's [offset, length] slice is copied.
        byte[]? externalRaw = null;
        if (dataLocation == 1 /* EXTERNAL */)
        {
            if (extLocation is null)
                throw new ModelSharpException(
                    $"Initializer '{name}' is EXTERNAL but has no 'location' in external_data.");
            if (resolver is null)
                throw new ModelSharpException(
                    $"Initializer '{name}' references external data file '{extLocation}', but the model " +
                    "was parsed from an in-memory byte buffer with no base directory; load it from a " +
                    "file path (OnnxModelLoader.LoadModel) so external data can be resolved.");

            externalRaw = resolver.Read(extLocation, extOffset, hasExtLength ? extLength : -1, name);
            rawData = externalRaw;
            hasRaw = true;
        }

        // If data_type is unset (0) but typed fields are present, infer it so we
        // still materialize the right dtype.
        if (dataType == 0)
        {
            if (int64Data is not null) dataType = DtInt64;
            else if (int32Data is not null) dataType = DtInt32;
            else dataType = DtFloat;
        }

        // DoS guard: a malformed/adversarial model can declare huge dims with little or no actual data,
        // which would force a giant managed allocation (e.g. dims=[2^31] with a few bytes of data). Every
        // element occupies at least 1 byte of raw data (or one entry of a typed list), so the declared
        // element count cannot legitimately exceed the available data. Reject before allocating.
        long availElems = hasRaw ? rawData.Length : 0;
        if (!hasRaw)
        {
            if (int64Data is not null && int64Data.Count > availElems) availElems = int64Data.Count;
            if (int32Data is not null && int32Data.Count > availElems) availElems = int32Data.Count;
            if (floatData is not null && floatData.Count > availElems) availElems = floatData.Count;
        }
        if (count > availElems)
            throw new ModelSharpException(
                $"Initializer '{name}' declares {count} elements but only {availElems} are available in its " +
                "data (malformed, truncated, or adversarial model).");

        switch (dataType)
        {
            case DtInt64:
            {
                var data = new long[count];
                if (hasRaw && count > 0)
                    MemoryMarshal.Cast<byte, long>(rawData).Slice(0, count).CopyTo(data);
                else if (int64Data is not null)
                    for (int k = 0; k < count && k < int64Data.Count; k++) data[k] = int64Data[k];
                return (name, new Tensor<long>(shape, data));
            }
            case DtInt32:
            {
                var data = new int[count];
                if (hasRaw && count > 0)
                    MemoryMarshal.Cast<byte, int>(rawData).Slice(0, count).CopyTo(data);
                else if (int32Data is not null)
                    for (int k = 0; k < count && k < int32Data.Count; k++) data[k] = int32Data[k];
                return (name, new Tensor<int>(shape, data));
            }
            case DtBool:
            {
                var data = new bool[count];
                if (hasRaw && count > 0)
                {
                    ReadOnlySpan<byte> raw = rawData;
                    for (int k = 0; k < count; k++) data[k] = raw[k] != 0;
                }
                else if (int32Data is not null) // ONNX packs bool into int32_data
                    for (int k = 0; k < count && k < int32Data.Count; k++) data[k] = int32Data[k] != 0;
                return (name, new Tensor<bool>(shape, data));
            }
            case DtFloat:
            {
                var data = new float[count];
                if (hasRaw && count > 0)
                    MemoryMarshal.Cast<byte, float>(rawData).Slice(0, count).CopyTo(data);
                else if (floatData is not null)
                    for (int k = 0; k < count && k < floatData.Count; k++) data[k] = floatData[k];
                return (name, new Tensor<float>(shape, data));
            }
            case DtUint8:
            {
                // uint8 (e.g. quantized weights / zero-points). Raw bytes map directly; otherwise
                // ONNX packs the values into int32_data.
                var data = new byte[count];
                if (hasRaw && count > 0)
                    rawData.Slice(0, count).CopyTo(data);
                else if (int32Data is not null)
                    for (int k = 0; k < count && k < int32Data.Count; k++) data[k] = (byte)int32Data[k];
                return (name, new Tensor<byte>(shape, data));
            }
            case DtInt8:
            {
                var data = new sbyte[count];
                if (hasRaw && count > 0)
                    MemoryMarshal.Cast<byte, sbyte>(rawData).Slice(0, count).CopyTo(data);
                else if (int32Data is not null)
                    for (int k = 0; k < count && k < int32Data.Count; k++) data[k] = (sbyte)int32Data[k];
                return (name, new Tensor<sbyte>(shape, data));
            }
            case DtFloat16:
            {
                // IEEE half-precision initializers are kept as compact System.Half (2 bytes/element —
                // half the memory of float32) and widened to float32 on demand at the compute boundary
                // (Tensor.ToFloat32 / GraphContext.Get). In raw_data each element is a little-endian
                // 16-bit value; otherwise ONNX packs them into int32_data (one value per int32, low 16 bits).
                var data = new Half[count];
                if (hasRaw && count > 0)
                {
                    ReadOnlySpan<Half> bits = MemoryMarshal.Cast<byte, Half>(rawData);
                    for (int k = 0; k < count && k < bits.Length; k++) data[k] = bits[k];
                }
                else if (int32Data is not null)
                    for (int k = 0; k < count && k < int32Data.Count; k++)
                        data[k] = BitConverter.UInt16BitsToHalf((ushort)int32Data[k]);
                return (name, new Tensor<Half>(shape, data));
            }
            case DtBFloat16:
            {
                // bfloat16 initializers are decoded to float32. A bfloat16 value is simply the
                // top 16 bits of a float32, so widening is (bits << 16) reinterpreted as float.
                // raw_data holds little-endian 16-bit values; otherwise ONNX packs them into
                // int32_data (one value per int32, low 16 bits).
                var data = new float[count];
                if (hasRaw && count > 0)
                {
                    ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<byte, ushort>(rawData);
                    for (int k = 0; k < count && k < bits.Length; k++)
                        data[k] = BitConverter.Int32BitsToSingle(bits[k] << 16);
                }
                else if (int32Data is not null)
                    for (int k = 0; k < count && k < int32Data.Count; k++)
                        data[k] = BitConverter.Int32BitsToSingle((ushort)int32Data[k] << 16);
                return (name, new Tensor<float>(shape, data));
            }
            default:
                throw new ModelSharpException(
                    $"Unsupported tensor data_type {dataType} for '{name}'.");
        }
    }

    /// <summary>
    /// Parses one <c>external_data</c> StringStringEntryProto (key = field 1, value = field 2) and
    /// folds the recognized keys (<c>location</c>, <c>offset</c>, <c>length</c>) into the running
    /// external-data descriptor. <c>checksum</c> and any unknown keys are ignored.
    /// </summary>
    private static void ApplyExternalDataEntry(
        ReadOnlySpan<byte> bytes,
        ref string? location, ref long offset, ref long length, ref bool hasLength)
    {
        (string key, string value) = ParseMetadataEntry(bytes);
        switch (key)
        {
            case "location": location = value; break;
            case "offset":
                offset = value.Length == 0 ? 0 : long.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
                break;
            case "length":
                if (value.Length != 0)
                {
                    length = long.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
                    hasLength = true;
                }
                break;
            // "checksum" and any unrecognized keys: ignored.
        }
    }

    /// <summary>
    /// Resolves and reads ONNX external-data weight files relative to the model's directory.
    /// Each referenced file is memory-mapped exactly once and the mapping is reused across every
    /// tensor in a single load, so a multi-GB weights file is never copied into the heap in full —
    /// only each tensor's own [offset, length] slice is copied into its managed array. Mappings are
    /// released when the resolver is disposed at the end of the load.
    /// </summary>
    private sealed class ExternalDataResolver : IDisposable
    {
        private readonly string? _baseDir;
        private readonly Dictionary<string, (MemoryMappedFile File, long Length)> _maps =
            new(StringComparer.Ordinal);

        public ExternalDataResolver(string? baseDir) => _baseDir = baseDir;

        /// <summary>
        /// Reads <paramref name="length"/> bytes (or to end-of-file when <paramref name="length"/>
        /// is negative) starting at <paramref name="offset"/> from the external file named by
        /// <paramref name="location"/> (resolved relative to the model directory) into a fresh array.
        /// </summary>
        public byte[] Read(string location, long offset, long length, string tensorName)
        {
            string resolved = _baseDir is null ? location : Path.Combine(_baseDir, location);

            if (!_maps.TryGetValue(resolved, out var entry))
            {
                if (!File.Exists(resolved))
                    throw new ModelSharpException(
                        $"External data file '{resolved}' for initializer '{tensorName}' was not found.");
                long fileLen = new FileInfo(resolved).Length;
                // A zero-length file cannot be memory-mapped; record it so empty slices still work.
                MemoryMappedFile mmf = fileLen == 0
                    ? null!
                    : MemoryMappedFile.CreateFromFile(
                        resolved, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
                entry = (mmf, fileLen);
                _maps[resolved] = entry;
            }

            if (length < 0) length = entry.Length - offset;
            if (offset < 0 || length < 0 || offset + length > entry.Length)
                throw new ModelSharpException(
                    $"External data slice [{offset}, {offset + length}) for initializer '{tensorName}' " +
                    $"is out of bounds for file '{resolved}' (length {entry.Length}).");

            var buffer = new byte[length];
            if (length == 0) return buffer;

            using MemoryMappedViewAccessor view =
                entry.File.CreateViewAccessor(offset, length, MemoryMappedFileAccess.Read);
            view.ReadArray(0, buffer, 0, checked((int)length));
            return buffer;
        }

        public void Dispose()
        {
            foreach (var entry in _maps.Values) entry.File?.Dispose();
            _maps.Clear();
        }
    }

    private static string ParseValueInfoName(ReadOnlySpan<byte> bytes)
    {
        var r = new ProtoReader(bytes);
        while (r.TryReadTag(out int field, out int wire))
        {
            if (field == 1 && wire == 2) return r.ReadString();
            r.SkipField(wire);
        }
        return "";
    }

    private static void ReadPackedFloats(ReadOnlySpan<byte> span, List<float> outList)
    {
        var rr = new ProtoReader(span);
        while (!rr.Eof) outList.Add(rr.ReadFloat());
    }

    private static void ReadPackedVarints(ReadOnlySpan<byte> span, List<long> outList)
    {
        var rr = new ProtoReader(span);
        while (!rr.Eof) outList.Add(rr.ReadInt64());
    }

    private static void ReadPackedVarintsAsInt(ReadOnlySpan<byte> span, List<int> outList)
    {
        var rr = new ProtoReader(span);
        while (!rr.Eof) outList.Add(rr.ReadInt32());
    }
}
