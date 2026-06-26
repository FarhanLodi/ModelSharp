using System;

namespace ModelSharp.Weights;

/// <summary>
/// The GGUF metadata value-type tags, as written by llama.cpp. The numeric values are the
/// on-disk <c>uint32</c> tags and must not be reordered.
/// </summary>
public enum GgufValueType : uint
{
    /// <summary>8-bit unsigned integer.</summary>
    UInt8 = 0,

    /// <summary>8-bit signed integer.</summary>
    Int8 = 1,

    /// <summary>16-bit unsigned integer.</summary>
    UInt16 = 2,

    /// <summary>16-bit signed integer.</summary>
    Int16 = 3,

    /// <summary>32-bit unsigned integer.</summary>
    UInt32 = 4,

    /// <summary>32-bit signed integer.</summary>
    Int32 = 5,

    /// <summary>32-bit IEEE-754 float.</summary>
    Float32 = 6,

    /// <summary>1-byte boolean.</summary>
    Bool = 7,

    /// <summary>Length-prefixed UTF-8 string.</summary>
    String = 8,

    /// <summary>Homogeneous array: element type tag + count + elements.</summary>
    Array = 9,

    /// <summary>64-bit unsigned integer.</summary>
    UInt64 = 10,

    /// <summary>64-bit signed integer.</summary>
    Int64 = 11,

    /// <summary>64-bit IEEE-754 float.</summary>
    Float64 = 12,
}

/// <summary>
/// A single parsed GGUF metadata value. Scalars are boxed in <see cref="Value"/> with the CLR
/// type corresponding to <see cref="Type"/> (e.g. <see cref="byte"/>, <see cref="long"/>,
/// <see cref="float"/>, <see cref="bool"/>, <see cref="string"/>); arrays are stored as an
/// <see cref="System.Array"/> of the element CLR type, with <see cref="ArrayElementType"/>
/// giving the GGUF element tag.
/// </summary>
public sealed class GgufMetadataValue
{
    internal GgufMetadataValue(GgufValueType type, object value, GgufValueType? arrayElementType = null)
    {
        Type = type;
        Value = value;
        ArrayElementType = arrayElementType;
    }

    /// <summary>The GGUF value-type tag for this entry.</summary>
    public GgufValueType Type { get; }

    /// <summary>For arrays, the element type tag; otherwise <c>null</c>.</summary>
    public GgufValueType? ArrayElementType { get; }

    /// <summary>
    /// The boxed value: a scalar of the matching CLR type, a <see cref="string"/>, or — when
    /// <see cref="Type"/> is <see cref="GgufValueType.Array"/> — an array of the element CLR type.
    /// </summary>
    public object Value { get; }

    /// <inheritdoc />
    public override string ToString() =>
        Type == GgufValueType.Array
            ? $"{ArrayElementType}[] (count={(Value is Array a ? a.Length : 0)})"
            : $"{Type}: {Value}";
}
