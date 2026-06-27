using System;
using System.Numerics;
using System.Threading.Tasks;

namespace ModelSharp.Cpu.Kernels.Linear;

/// <summary>
/// Shared multithreading + SIMD primitives for the CPU matmul family (MatMul, Gemm,
/// MatMulNBits, MatMulInteger, QLinearMatMul). All routines here are pure helpers over flat
/// row-major buffers; the parallel axis is always an <b>independent output dimension</b> so no
/// two threads ever write the same output element and no locking is required.
/// </summary>
/// <remarks>
/// <para><b>SIMD:</b> the inner contraction (the K loop) is vectorized with
/// <see cref="Vector{T}"/> over contiguous lanes. For float dot products this changes the
/// accumulation order (lane-partial sums then a horizontal reduce), which can shift results by a
/// few ULP — that is within the existing float test tolerances. For integer reductions the
/// accumulation stays a plain per-output-element sum so the result is bit-exact.</para>
/// <para><b>Threshold:</b> going parallel only pays off once the work
/// (<c>rows · innerCost</c>) is large enough to amortise task-scheduling overhead; below the
/// threshold callers run the serial body. See <see cref="ParallelThreshold"/>.</para>
/// </remarks>
internal static class MatMulParallel
{
    /// <summary>
    /// Minimum estimated FLOP-ish work (product of the parallelised row count and the per-row
    /// inner cost) before we dispatch to <c>Parallel.For</c>. Tuned so that tiny matmuls
    /// (e.g. the per-token decode GEMV of a small model, or attention score blocks) stay serial
    /// and avoid thread-pool churn, while the big LLM projection/MLP matmuls parallelise.
    /// </summary>
    public const long ParallelThreshold = 1L << 16; // 65,536 multiply-adds

    /// <summary>Number of worker partitions to request (caps oversubscription on big machines).</summary>
    public static int DegreeOfParallelism => Math.Max(1, Environment.ProcessorCount);

    /// <summary>True when the work is big enough that parallelism is expected to win.</summary>
    public static bool ShouldParallelize(long rows, long innerCost)
        => rows > 1 && DegreeOfParallelism > 1 && checked(rows * innerCost) >= ParallelThreshold;

    /// <summary>
    /// Runs <paramref name="body"/> for each index in <c>[0, count)</c>, in parallel when the work
    /// is large enough and serially otherwise. The body must only write output elements uniquely
    /// owned by its index.
    /// </summary>
    public static void For(int count, long innerCost, Action<int> body)
    {
        if (ShouldParallelize(count, innerCost))
        {
            var opts = new ParallelOptions { MaxDegreeOfParallelism = DegreeOfParallelism };
            Parallel.For(0, count, opts, body);
        }
        else
        {
            for (int i = 0; i < count; i++) body(i);
        }
    }

    /// <summary>
    /// Vectorized dot product of two contiguous float spans of equal length <paramref name="k"/>:
    /// <c>Σ a[aBase+i] · w[i]</c>. <paramref name="a"/> is offset by <paramref name="aBase"/>;
    /// <paramref name="w"/> is taken from its start. Lane-partial accumulation then horizontal sum.
    /// </summary>
    public static float Dot(ReadOnlySpan<float> a, int aBase, ReadOnlySpan<float> w, int k)
    {
        int width = Vector<float>.Count;
        // Four independent accumulators break the single-dependency chain so the CPU can keep
        // multiple multiply-adds in flight (a lone accumulator stalls on FMA latency).
        Vector<float> a0 = Vector<float>.Zero, a1 = Vector<float>.Zero,
                      a2 = Vector<float>.Zero, a3 = Vector<float>.Zero;
        int i = 0;
        int limit4 = k - 4 * width;
        for (; i <= limit4; i += 4 * width)
        {
            a0 += new Vector<float>(a.Slice(aBase + i, width)) * new Vector<float>(w.Slice(i, width));
            a1 += new Vector<float>(a.Slice(aBase + i + width, width)) * new Vector<float>(w.Slice(i + width, width));
            a2 += new Vector<float>(a.Slice(aBase + i + 2 * width, width)) * new Vector<float>(w.Slice(i + 2 * width, width));
            a3 += new Vector<float>(a.Slice(aBase + i + 3 * width, width)) * new Vector<float>(w.Slice(i + 3 * width, width));
        }
        var acc = (a0 + a1) + (a2 + a3);
        int limit = k - width;
        for (; i <= limit; i += width)
            acc += new Vector<float>(a.Slice(aBase + i, width)) * new Vector<float>(w.Slice(i, width));
        float sum = Vector.Dot(acc, Vector<float>.One);
        for (; i < k; i++) sum += a[aBase + i] * w[i];
        return sum;
    }
}
