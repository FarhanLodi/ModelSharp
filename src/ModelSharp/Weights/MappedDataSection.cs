using System;
using System.IO.MemoryMappedFiles;

namespace ModelSharp.Weights;

/// <summary>
/// An <see cref="IDataSection"/> backed by a memory-mapped view of a file. The view is created
/// over the whole file (offset 0); reads address bytes through the accessor's 64-bit
/// <c>position</c> parameter, so a shard larger than 2 GB is served without ever allocating the
/// whole file as a managed array — each call copies only the requested range into a fresh array.
/// <para>
/// The data section logically begins at <c>headerOffset</c> bytes into the view (the 8-byte
/// length prefix and the header JSON, for safetensors, or the aligned header end, for GGUF). The
/// owning reader keeps the parent <see cref="MemoryMappedFile"/> alive; this type disposes the
/// view it created.
/// </para>
/// </summary>
internal sealed class MappedDataSection : IDataSection
{
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly long _headerOffset;
    private readonly long _length;
    private bool _disposed;

    /// <param name="accessor">A view accessor covering the whole file (offset 0).</param>
    /// <param name="headerOffset">Byte offset, within the view, where the data blob begins.</param>
    /// <param name="length">Length, in bytes, of the data blob.</param>
    public MappedDataSection(MemoryMappedViewAccessor accessor, long headerOffset, long length)
    {
        _accessor = accessor;
        _headerOffset = headerOffset;
        _length = length;
    }

    /// <inheritdoc />
    public long Length => _length;

    /// <inheritdoc />
    public byte[] ReadBytes(long byteOffset, int length)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MappedDataSection));
        if (length < 0 || byteOffset < 0 || byteOffset + length > _length)
            throw new ModelSharpException(
                $"Data-section read [{byteOffset}, {byteOffset + length}) is out of " +
                $"range (section is {_length} bytes).");

        var result = new byte[length];
        if (length > 0)
        {
            // ReadArray takes a long position, so the absolute byte offset may exceed int range
            // even though the per-tensor read length does not.
            int read = _accessor.ReadArray(_headerOffset + byteOffset, result, 0, length);
            if (read != length)
                throw new ModelSharpException(
                    $"Short read from memory-mapped data section: expected {length} bytes, got {read}.");
        }
        return result;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _accessor.Dispose();
    }
}
