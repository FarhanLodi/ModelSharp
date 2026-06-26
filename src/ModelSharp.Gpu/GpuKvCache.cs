using System;
using ILGPU;
using ILGPU.Runtime;

namespace ModelSharp.Gpu;

/// <summary>
/// A device-resident key/value cache for autoregressive transformer decoding (B5). The K and V tensors are
/// laid out <c>[numHeads, maxSeq, headDim]</c> in a single device buffer each and persist <em>across</em>
/// decode steps: <see cref="IlgpuEngine.DecodeStepAttention"/> appends each step's keys/values into the
/// region at the current <see cref="SeqLen"/> offset (a device→device copy, no realloc, no host round-trip)
/// and then attends over the whole cached prefix. Create one via <see cref="IlgpuEngine.CreateKvCache"/>;
/// it borrows the engine's accelerator, so dispose it before (or together with) the engine.
/// </summary>
public sealed class GpuKvCache : IDisposable
{
    internal MemoryBuffer1D<float, Stride1D.Dense> KBuffer { get; }
    internal MemoryBuffer1D<float, Stride1D.Dense> VBuffer { get; }

    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; }

    /// <summary>Maximum number of past tokens the cache can hold.</summary>
    public int MaxSeq { get; }

    /// <summary>Per-head key/value dimension.</summary>
    public int HeadDim { get; }

    /// <summary>The number of tokens currently cached (the append offset for the next step).</summary>
    public int SeqLen { get; internal set; }

    internal GpuKvCache(
        MemoryBuffer1D<float, Stride1D.Dense> kBuffer,
        MemoryBuffer1D<float, Stride1D.Dense> vBuffer,
        int numHeads, int maxSeq, int headDim)
    {
        KBuffer = kBuffer;
        VBuffer = vBuffer;
        NumHeads = numHeads;
        MaxSeq = maxSeq;
        HeadDim = headDim;
        SeqLen = 0;
    }

    /// <summary>Resets the cache to empty (the device buffers are retained for reuse on a new sequence).</summary>
    public void Reset() => SeqLen = 0;

    /// <summary>Releases the device key/value buffers.</summary>
    public void Dispose()
    {
        KBuffer.Dispose();
        VBuffer.Dispose();
    }
}
