using System;

namespace ModelSharp.Weights;

/// <summary>
/// An abstraction over the packed tensor-data blob that trails a weight file's header.
/// Unlike a plain <see cref="ReadOnlyMemory{T}"/> (whose offsets are 32-bit), a data
/// section addresses bytes with a <see cref="long"/> offset so that a single shard
/// larger than 2 GB can be served — the byte range of any individual tensor still fits
/// in a managed array, but its <em>position</em> within the blob may not fit in an
/// <see cref="int"/>.
/// <para>
/// Implementations are: an in-memory section backed by a <see cref="ReadOnlyMemory{T}"/>
/// (used by <c>FromBytes</c>/<c>FromStream</c>) and a memory-mapped section backed by an
/// OS file mapping (used by <c>FromFile</c>). The latter owns native resources, hence
/// <see cref="IDisposable"/>.
/// </para>
/// </summary>
internal interface IDataSection : IDisposable
{
    /// <summary>Total length, in bytes, of the data section.</summary>
    long Length { get; }

    /// <summary>
    /// Returns a freshly-allocated array of <paramref name="length"/> bytes starting at
    /// <paramref name="byteOffset"/> within the section (which may be beyond
    /// <see cref="int.MaxValue"/>). The requested range must lie fully within the section.
    /// </summary>
    byte[] ReadBytes(long byteOffset, int length);
}
