using System;

namespace ModelSharp.Weights;

/// <summary>
/// An <see cref="IDataSection"/> backed by an in-memory <see cref="ReadOnlyMemory{T}"/>.
/// Used by the <c>FromBytes</c>/<c>FromStream</c> paths; offsets are still exposed as
/// <see cref="long"/> for a uniform interface, but the underlying buffer is necessarily
/// &lt;= 2 GB. <see cref="Dispose"/> is a no-op — the buffer is owned by the caller / GC.
/// </summary>
internal sealed class MemoryDataSection : IDataSection
{
    private readonly ReadOnlyMemory<byte> _data;

    public MemoryDataSection(ReadOnlyMemory<byte> data) => _data = data;

    /// <inheritdoc />
    public long Length => _data.Length;

    /// <inheritdoc />
    public byte[] ReadBytes(long byteOffset, int length)
    {
        if (length < 0 || byteOffset < 0 || byteOffset + length > _data.Length)
            throw new ModelSharpException(
                $"Data-section read [{byteOffset}, {byteOffset + length}) is out of " +
                $"range (section is {_data.Length} bytes).");

        var result = new byte[length];
        if (length > 0)
            _data.Span.Slice((int)byteOffset, length).CopyTo(result);
        return result;
    }

    /// <inheritdoc />
    public void Dispose() { /* in-memory section owns nothing */ }
}
