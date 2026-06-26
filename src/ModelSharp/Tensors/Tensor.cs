using System;

namespace ModelSharp.Tensors;

/// <summary>
/// Non-generic, dtype-carrying base for all tensors. Lets the engine plumbing
/// (graph context, named bindings, initializers) move tensors around without
/// knowing their element type up front, then recover the typed view via the
/// <see cref="AsFloat"/> / <see cref="AsInt64"/> / <see cref="AsInt32"/> /
/// <see cref="AsBool"/> accessors.
/// </summary>
public abstract class Tensor
{
    /// <summary>The tensor's shape.</summary>
    public TensorShape Shape { get; }

    /// <summary>The element type carried by this tensor.</summary>
    public abstract ElementType Dtype { get; }

    /// <summary>Total element count.</summary>
    public long Length => Shape.Length;

    private protected Tensor(TensorShape shape) => Shape = shape;

    /// <summary>Recovers the float32 view; throws if the dtype is not Float32.</summary>
    public Tensor<float> AsFloat() => As<float>(ElementType.Float32);

    /// <summary>Recovers the int64 view; throws if the dtype is not Int64.</summary>
    public Tensor<long> AsInt64() => As<long>(ElementType.Int64);

    /// <summary>Recovers the int32 view; throws if the dtype is not Int32.</summary>
    public Tensor<int> AsInt32() => As<int>(ElementType.Int32);

    /// <summary>Recovers the bool view; throws if the dtype is not Boolean.</summary>
    public Tensor<bool> AsBool() => As<bool>(ElementType.Boolean);

    /// <summary>Returns a same-dtype tensor sharing this buffer with a new shape (same element count).
    /// Used by view ops (Reshape / Unsqueeze / Squeeze / Flatten) to stay dtype-agnostic.</summary>
    public abstract Tensor WithShape(TensorShape shape);

    private Tensor<T> As<T>(ElementType expected) where T : unmanaged =>
        this as Tensor<T>
        ?? throw new InvalidOperationException(
            $"Tensor dtype is {Dtype}; expected {expected} (.NET '{typeof(T).Name}').");
}

/// <summary>
/// A dense, row-major tensor backed by a single contiguous buffer of element type <typeparamref name="T"/>.
/// </summary>
public sealed class Tensor<T> : Tensor where T : unmanaged
{
    /// <summary>The backing row-major buffer.</summary>
    public Memory<T> Buffer { get; }

    /// <inheritdoc />
    public override ElementType Dtype { get; }

    /// <summary>Creates a tensor over an existing buffer (no copy).</summary>
    public Tensor(TensorShape shape, Memory<T> buffer)
        : base(shape)
    {
        if (buffer.Length != shape.Length)
            throw new ArgumentException(
                $"Buffer length {buffer.Length} does not match shape {shape} ({shape.Length} elements).",
                nameof(buffer));
        Buffer = buffer;
        Dtype = ElementTypeOf();
    }

    /// <summary>Allocates a zero-filled tensor of the given shape.</summary>
    public Tensor(TensorShape shape)
        : this(shape, new T[checked((int)shape.Length)]) { }

    /// <summary>Mutable view over the buffer.</summary>
    public Span<T> Span => Buffer.Span;

    /// <summary>Wraps an array as a tensor (no copy).</summary>
    public static Tensor<T> FromArray(TensorShape shape, T[] data) => new(shape, data);

    /// <inheritdoc />
    public override Tensor WithShape(TensorShape shape) => new Tensor<T>(shape, Buffer);

    private static ElementType ElementTypeOf()
    {
        Type t = typeof(T);
        if (t == typeof(float)) return ElementType.Float32;
        if (t == typeof(double)) return ElementType.Float64;
        if (t == typeof(int)) return ElementType.Int32;
        if (t == typeof(long)) return ElementType.Int64;
        if (t == typeof(sbyte)) return ElementType.Int8;
        if (t == typeof(byte)) return ElementType.UInt8;
        if (t == typeof(bool)) return ElementType.Boolean;
        return ElementType.Unknown;
    }
}
