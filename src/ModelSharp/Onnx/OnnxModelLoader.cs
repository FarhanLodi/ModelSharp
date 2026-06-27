using System;
using System.Collections.Generic;
using System.IO;
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
    public static ModelGraph LoadModel(string path) => ParseModel(File.ReadAllBytes(path));

    /// <summary>Parses an ONNX ModelProto from bytes.</summary>
    public static ModelGraph ParseModel(ReadOnlySpan<byte> modelProto)
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
        return ParseGraph(graphBytes, metadata);
    }

    /// <summary>Loads a standalone TensorProto (e.g. an ONNX test-data <c>.pb</c>) as a float32 tensor.</summary>
    public static Tensor<float> LoadTensor(string path) => ParseTensor(File.ReadAllBytes(path)).Tensor.AsFloat();

    /// <summary>Loads a standalone TensorProto preserving its declared dtype.</summary>
    public static Tensor LoadTensorTyped(string path) => ParseTensor(File.ReadAllBytes(path)).Tensor;

    private static ModelGraph ParseGraph(ReadOnlySpan<byte> bytes, Dictionary<string, string>? metadata)
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
                    nodes.Add(ParseNode(r.ReadLengthDelimited()));
                    break;
                case 5 when wire == 2:
                    (string tn, Tensor tt) = ParseTensor(r.ReadLengthDelimited());
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

    private static GraphNode ParseNode(ReadOnlySpan<byte> bytes)
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
                    (string an, object? av) = ParseAttribute(r.ReadLengthDelimited());
                    if (av is not null) attrs[an] = av;
                    break;
                default: r.SkipField(wire); break;
            }
        }

        return new GraphNode(opType, name.Length == 0 ? opType : name, inputs, outputs, attrs);
    }

    private static (string Name, object? Value) ParseAttribute(ReadOnlySpan<byte> bytes)
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
                case 5 when wire == 2: t = ParseTensor(r.ReadLengthDelimited()).Tensor; break;
                // AttributeProto.g (field 6): single nested GraphProto.
                case 6 when wire == 2: g = ParseGraph(r.ReadLengthDelimited(), metadata: null); break;
                case 7 when wire == 5: (floats ??= new()).Add(r.ReadFloat()); break;
                case 7 when wire == 2: ReadPackedFloats(r.ReadLengthDelimited(), floats ??= new()); break;
                case 8 when wire == 0: (ints ??= new()).Add(r.ReadInt64()); break;
                case 8 when wire == 2: ReadPackedVarints(r.ReadLengthDelimited(), ints ??= new()); break;
                // AttributeProto.graphs (field 10): repeated GraphProto.
                case 10 when wire == 2: (graphs ??= new()).Add(ParseGraph(r.ReadLengthDelimited(), metadata: null)); break;
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

    // ONNX TensorProto.DataType values we materialize.
    private const long DtFloat = 1, DtInt32 = 6, DtInt64 = 7, DtBool = 9;

    private static (string Name, Tensor Tensor) ParseTensor(ReadOnlySpan<byte> bytes)
    {
        var dims = new List<long>();
        long dataType = 0;
        string name = "";
        ReadOnlySpan<byte> rawData = default;
        bool hasRaw = false;
        List<float>? floatData = null;
        List<long>? int64Data = null;
        List<int>? int32Data = null;

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
                default: r.SkipField(wire); break;
            }
        }

        int[] shapeDims = dims.Select(d => (int)d).ToArray();
        var shape = new TensorShape(shapeDims);
        int count = checked((int)shape.Length);

        // If data_type is unset (0) but typed fields are present, infer it so we
        // still materialize the right dtype.
        if (dataType == 0)
        {
            if (int64Data is not null) dataType = DtInt64;
            else if (int32Data is not null) dataType = DtInt32;
            else dataType = DtFloat;
        }

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
            default:
                throw new ModelSharpException(
                    $"Unsupported tensor data_type {dataType} for '{name}'.");
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
