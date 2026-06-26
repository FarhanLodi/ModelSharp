using ModelSharp.Tensors;

namespace ModelSharp.Weights;

/// <summary>
/// Static description of one tensor stored in a GGUF file: its name, dimensions, ggml element
/// type, and the byte offset (relative to the start of the tensor-data blob) where its bytes
/// begin. No data is materialized until <see cref="GgufFile.GetTensor"/> or
/// <see cref="GgufFile.GetRawTensorBytes"/> is called.
/// </summary>
public sealed class GgufTensorInfo
{
    /// <summary>The tensor's name.</summary>
    public string Name { get; }

    /// <summary>The row-major shape. GGUF stores dimensions fastest-varying-first; they are
    /// reversed here so this shape matches the usual row-major (C-order) convention.</summary>
    public TensorShape Shape { get; }

    /// <summary>The ggml element type as declared in the file.</summary>
    public GgmlType Type { get; }

    /// <summary>Byte offset, relative to the start of the tensor-data blob, of this tensor.</summary>
    public long Offset { get; }

    /// <summary>Length, in bytes, of this tensor's data (computed from shape and ggml type).</summary>
    public long ByteLength { get; }

    /// <summary>Total element count (product of <see cref="Shape"/>; 1 for a scalar).</summary>
    public long ElementCount => Shape.Length;

    internal GgufTensorInfo(string name, TensorShape shape, GgmlType type, long offset, long byteLength)
    {
        Name = name;
        Shape = shape;
        Type = type;
        Offset = offset;
        ByteLength = byteLength;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name}: {Type} {Shape} @ {Offset} ({ByteLength} bytes)";
}
