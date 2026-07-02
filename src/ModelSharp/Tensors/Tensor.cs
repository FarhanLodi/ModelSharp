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

    /// <summary>Recovers the float16 view; throws if the dtype is not Float16.</summary>
    public Tensor<System.Half> AsFloat16() => As<System.Half>(ElementType.Float16);

    /// <summary>
    /// Returns a float32 tensor: a zero-copy view when the dtype is already Float32, or an upcast when
    /// the dtype is Float16 (fp16 weights are stored compactly and widened to float on demand at the
    /// compute boundary). The Float16 widening is memoized per tensor instance — the first call widens
    /// once and every subsequent call returns the same cached <see cref="Tensor{Single}"/>, so repeated
    /// reads (e.g. re-reading a weight on every inference/token) are free after the first. Throws for
    /// other dtypes.
    /// </summary>
    public Tensor<float> ToFloat32()
    {
        if (this is Tensor<float> f) return f;
        if (this is Tensor<System.Half> h) return h.WidenToFloat32Cached();
        return AsFloat(); // clear error for any other dtype
    }

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

    /// <summary>
    /// Lazily-populated cache of the float32 widening for Float16 tensors, so <see cref="Tensor.ToFloat32"/>
    /// widens Half→float at most once per instance. Populated via <see cref="System.Threading.LazyInitializer"/>
    /// (safe under concurrent kernel reads from <c>Parallel.For</c>); stays null for all other dtypes.
    /// </summary>
    private Tensor<float>? _float32Cache;

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

    /// <summary>
    /// Allocates a tensor of the given shape whose buffer is left uninitialized
    /// (<c>GC.AllocateUninitializedArray</c>) rather than zero-filled. For write-only outputs where every
    /// element is overwritten before it is read; skips the runtime's zeroing pass. Do not read an element
    /// before writing it.
    /// </summary>
    public static Tensor<T> AllocateUninitialized(TensorShape shape) =>
        new(shape, System.GC.AllocateUninitializedArray<T>(checked((int)shape.Length)));

    /// <summary>Mutable view over the buffer.</summary>
    public Span<T> Span => Buffer.Span;

    /// <summary>Wraps an array as a tensor (no copy).</summary>
    public static Tensor<T> FromArray(TensorShape shape, T[] data) => new(shape, data);

    /// <summary>
    /// Widens this Float16 tensor to float32 exactly once, caching and returning the result on every
    /// subsequent call. Only meaningful for <c>Tensor&lt;Half&gt;</c>; called by <see cref="Tensor.ToFloat32"/>.
    /// Thread-safe: concurrent callers may race to compute, but all observe the single cached instance.
    /// </summary>
    internal Tensor<float> WidenToFloat32Cached() =>
        System.Threading.LazyInitializer.EnsureInitialized(ref _float32Cache, WidenToFloat32);

    private Tensor<float> WidenToFloat32()
    {
        // Only Tensor<Half> reaches here (guarded by ToFloat32). The destination is fully overwritten
        // below, so it can be allocated uninitialized. The scalar widening loop is bounds-check-free
        // and auto-vectorized by the JIT; Half→float is a lossless, exact conversion.
        var half = (Tensor<System.Half>)(object)this;
        int len = checked((int)Length);
        var data = System.GC.AllocateUninitializedArray<float>(len);
        System.ReadOnlySpan<System.Half> src = half.Buffer.Span;
        Span<float> dst = data;
        for (int i = 0; i < len; i++) dst[i] = (float)src[i];
        return new Tensor<float>(Shape, data);
    }

    /// <inheritdoc />
    public override Tensor WithShape(TensorShape shape) => new Tensor<T>(shape, Buffer);

    private static ElementType ElementTypeOf()
    {
        Type t = typeof(T);
        if (t == typeof(float)) return ElementType.Float32;
        if (t == typeof(double)) return ElementType.Float64;
        if (t == typeof(System.Half)) return ElementType.Float16;
        if (t == typeof(int)) return ElementType.Int32;
        if (t == typeof(long)) return ElementType.Int64;
        if (t == typeof(sbyte)) return ElementType.Int8;
        if (t == typeof(byte)) return ElementType.UInt8;
        if (t == typeof(bool)) return ElementType.Boolean;
        return ElementType.Unknown;
    }
}
