using System;

namespace ModelSharp.Tensors;

/// <summary>An immutable, row-major tensor shape.</summary>
public readonly struct TensorShape : IEquatable<TensorShape>
{
    private readonly int[]? _dims;

    /// <summary>Creates a shape from its dimensions. Dimensions must be non-negative.</summary>
    public TensorShape(params int[] dims)
    {
        if (dims is null) throw new ArgumentNullException(nameof(dims));
        foreach (int d in dims)
            if (d < 0)
                throw new ArgumentOutOfRangeException(nameof(dims),
                    "Dimensions must be non-negative; resolve dynamic axes via the manifest first.");
        _dims = dims;
    }

    /// <summary>The dimensions, row-major.</summary>
    public ReadOnlySpan<int> Dimensions => _dims ?? Array.Empty<int>();

    /// <summary>Number of axes.</summary>
    public int Rank => _dims?.Length ?? 0;

    /// <summary>The size of the given axis.</summary>
    public int this[int axis] => Dimensions[axis];

    /// <summary>Total number of elements (product of dimensions; 1 for a scalar).</summary>
    public long Length
    {
        get
        {
            long n = 1;
            if (_dims is not null)
                foreach (int d in _dims) n *= d;
            return n;
        }
    }

    /// <inheritdoc />
    public bool Equals(TensorShape other) => Dimensions.SequenceEqual(other.Dimensions);

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hc = new HashCode();
        foreach (int d in Dimensions) hc.Add(d);
        return hc.ToHashCode();
    }

    /// <inheritdoc />
    public override string ToString() => "[" + string.Join(", ", Dimensions.ToArray()) + "]";
}
