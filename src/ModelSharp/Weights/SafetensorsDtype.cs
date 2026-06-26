namespace ModelSharp.Weights;

/// <summary>
/// The element data types recognized by the safetensors format. The string forms
/// (<c>F32</c>, <c>BF16</c>, <c>I64</c>, ...) are taken verbatim from the header JSON.
/// </summary>
public enum SafetensorsDtype
{
    /// <summary>Unrecognized / unsupported dtype.</summary>
    Unknown = 0,

    /// <summary>64-bit IEEE-754 binary float (<c>"F64"</c>).</summary>
    Float64,

    /// <summary>32-bit IEEE-754 binary float (<c>"F32"</c>).</summary>
    Float32,

    /// <summary>16-bit IEEE-754 half float (<c>"F16"</c>).</summary>
    Float16,

    /// <summary>16-bit "brain" float — the upper 16 bits of a float32 (<c>"BF16"</c>).</summary>
    BFloat16,

    /// <summary>64-bit signed integer (<c>"I64"</c>).</summary>
    Int64,

    /// <summary>32-bit signed integer (<c>"I32"</c>).</summary>
    Int32,

    /// <summary>16-bit signed integer (<c>"I16"</c>).</summary>
    Int16,

    /// <summary>8-bit signed integer (<c>"I8"</c>).</summary>
    Int8,

    /// <summary>8-bit unsigned integer (<c>"U8"</c>).</summary>
    UInt8,

    /// <summary>1-byte boolean (<c>"BOOL"</c>); each element is a single 0/1 byte.</summary>
    Bool,
}

/// <summary>Parsing and sizing helpers for <see cref="SafetensorsDtype"/>.</summary>
internal static class SafetensorsDtypeInfo
{
    /// <summary>
    /// Maps a safetensors dtype string to a <see cref="SafetensorsDtype"/>.
    /// The 8-bit float variants (<c>F8_*</c>) and any other unknown string are
    /// rejected with a <see cref="ModelSharpException"/>.
    /// </summary>
    public static SafetensorsDtype Parse(string dtype, string tensorName) => dtype switch
    {
        "F64" => SafetensorsDtype.Float64,
        "F32" => SafetensorsDtype.Float32,
        "F16" => SafetensorsDtype.Float16,
        "BF16" => SafetensorsDtype.BFloat16,
        "I64" => SafetensorsDtype.Int64,
        "I32" => SafetensorsDtype.Int32,
        "I16" => SafetensorsDtype.Int16,
        "I8" => SafetensorsDtype.Int8,
        "U8" => SafetensorsDtype.UInt8,
        "BOOL" => SafetensorsDtype.Bool,
        _ => throw new ModelSharpException(
            $"Unsupported safetensors dtype '{dtype}' for tensor '{tensorName}'."),
    };

    /// <summary>The size, in bytes, of a single element of the given dtype.</summary>
    public static int ByteSize(SafetensorsDtype dtype) => dtype switch
    {
        SafetensorsDtype.Float64 or SafetensorsDtype.Int64 => 8,
        SafetensorsDtype.Float32 or SafetensorsDtype.Int32 => 4,
        SafetensorsDtype.Float16 or SafetensorsDtype.BFloat16 or SafetensorsDtype.Int16 => 2,
        SafetensorsDtype.Int8 or SafetensorsDtype.UInt8 or SafetensorsDtype.Bool => 1,
        _ => throw new ModelSharpException($"No byte size is defined for dtype {dtype}."),
    };
}
