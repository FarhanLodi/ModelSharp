using ModelSharp.Tensors;

namespace ModelSharp.Weights;

/// <summary>
/// Static description of one tensor stored in a safetensors file: its name, on-disk
/// dtype, shape, and the length of its raw byte range. This is the "header" view —
/// no data is materialized until <see cref="SafetensorsFile.GetTensor"/> is called.
/// </summary>
public sealed class SafetensorsTensorInfo
{
    /// <summary>The tensor's name (the header JSON key).</summary>
    public string Name { get; }

    /// <summary>The dtype as declared in the file.</summary>
    public SafetensorsDtype Dtype { get; }

    /// <summary>The tensor's row-major shape.</summary>
    public TensorShape Shape { get; }

    /// <summary>The length, in bytes, of this tensor's slice of the data section.</summary>
    public long ByteLength { get; }

    /// <summary>Total element count (product of <see cref="Shape"/>; 1 for a scalar).</summary>
    public long ElementCount => Shape.Length;

    internal SafetensorsTensorInfo(string name, SafetensorsDtype dtype, TensorShape shape, long byteLength)
    {
        Name = name;
        Dtype = dtype;
        Shape = shape;
        ByteLength = byteLength;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name}: {Dtype} {Shape} ({ByteLength} bytes)";
}
