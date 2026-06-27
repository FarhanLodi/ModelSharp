using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Llm;

/// <summary>
/// Small SIMD primitives shared by the attention / normalization / convolution
/// CPU kernels. All operate on flat <see cref="float"/> arrays at explicit
/// offsets so they can be called from inside <c>Parallel.For</c> bodies (which
/// cannot capture <c>Span&lt;T&gt;</c>). They use <see cref="Vector{T}"/> for the
/// inner reductions/accumulations and fall back to scalar tails, so results are
/// bit-stable for a given vector width but may differ from a pure left-to-right
/// scalar sum within float tolerance (accumulation order).
/// </summary>
internal static class KernelSimd
{
    private static readonly int W = Vector<float>.Count;

    /// <summary>
    /// Returns the contiguous backing array of a freshly-allocated tensor so it can be
    /// indexed from inside a <c>Parallel.For</c> body. Tensors created via
    /// <c>new Tensor&lt;float&gt;(shape)</c> wrap a whole zero-offset array; if that ever
    /// fails to hold (offset != 0 or non-array memory) we copy to a dense array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] Array(Tensor<float> t)
    {
        if (MemoryMarshal.TryGetArray<float>(t.Buffer, out var seg) && seg.Offset == 0 &&
            seg.Array is { } arr && arr.Length == t.Buffer.Length)
            return arr;
        return t.Span.ToArray();
    }

    /// <summary>Dot product of <c>a[aOff..]</c> and <c>b[bOff..]</c> over <paramref name="n"/> lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(float[] a, int aOff, float[] b, int bOff, int n)
    {
        int i = 0;
        var acc = Vector<float>.Zero;
        int last = n - W;
        for (; i <= last; i += W)
            acc += new Vector<float>(a, aOff + i) * new Vector<float>(b, bOff + i);
        float dot = Vector.Dot(acc, Vector<float>.One);
        for (; i < n; i++) dot += a[aOff + i] * b[bOff + i];
        return dot;
    }

    /// <summary>Sum of squares of <c>a[off..off+n]</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SumSquares(float[] a, int off, int n)
    {
        int i = 0;
        var acc = Vector<float>.Zero;
        int last = n - W;
        for (; i <= last; i += W)
        {
            var v = new Vector<float>(a, off + i);
            acc += v * v;
        }
        float s = Vector.Dot(acc, Vector<float>.One);
        for (; i < n; i++) { float v = a[off + i]; s += v * v; }
        return s;
    }

    /// <summary>Sum of (a[off+i] - mean)² over i in [0,n).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SumSquaresCentered(float[] a, int off, float mean, int n)
    {
        int i = 0;
        var mv = new Vector<float>(mean);
        var acc = Vector<float>.Zero;
        int last = n - W;
        for (; i <= last; i += W)
        {
            var d = new Vector<float>(a, off + i) - mv;
            acc += d * d;
        }
        float s = Vector.Dot(acc, Vector<float>.One);
        for (; i < n; i++) { float d = a[off + i] - mean; s += d * d; }
        return s;
    }

    /// <summary>Sum of <c>a[off..off+n]</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(float[] a, int off, int n)
    {
        int i = 0;
        var acc = Vector<float>.Zero;
        int last = n - W;
        for (; i <= last; i += W)
            acc += new Vector<float>(a, off + i);
        float s = Vector.Dot(acc, Vector<float>.One);
        for (; i < n; i++) s += a[off + i];
        return s;
    }

    /// <summary>dst[dOff+i] = x[xOff+i] * factor * w[i] (+ bias[i] if provided) for i in [0,n).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NormApply(
        float[] dst, int dOff, float[] x, int xOff, float factor,
        float[] w, float[]? bias, int n)
    {
        int i = 0;
        var fv = new Vector<float>(factor);
        int last = n - W;
        if (bias is null)
        {
            for (; i <= last; i += W)
            {
                var v = new Vector<float>(x, xOff + i) * fv * new Vector<float>(w, i);
                v.CopyTo(dst, dOff + i);
            }
            for (; i < n; i++) dst[dOff + i] = x[xOff + i] * factor * w[i];
        }
        else
        {
            for (; i <= last; i += W)
            {
                var v = new Vector<float>(x, xOff + i) * fv * new Vector<float>(w, i) + new Vector<float>(bias, i);
                v.CopyTo(dst, dOff + i);
            }
            for (; i < n; i++) dst[dOff + i] = x[xOff + i] * factor * w[i] + bias[i];
        }
    }

    /// <summary>dst[dOff+i] = (x[xOff+i] - mean) * factor * w[i] (+ bias[i]) for i in [0,n).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NormApplyCentered(
        float[] dst, int dOff, float[] x, int xOff, float mean, float factor,
        float[] w, float[]? bias, int n)
    {
        int i = 0;
        var mv = new Vector<float>(mean);
        var fv = new Vector<float>(factor);
        int last = n - W;
        for (; i <= last; i += W)
        {
            var v = (new Vector<float>(x, xOff + i) - mv) * fv * new Vector<float>(w, i);
            if (bias is not null) v += new Vector<float>(bias, i);
            v.CopyTo(dst, dOff + i);
        }
        for (; i < n; i++)
        {
            float val = (x[xOff + i] - mean) * factor * w[i];
            dst[dOff + i] = bias is null ? val : val + bias[i];
        }
    }

    /// <summary>dst[i] = a[i] + b[i] (+ c[i] if provided), all over [0,n).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add3(float[] dst, int dOff, float[] a, int aOff, float[] b, int bOff, float[]? c, int n)
    {
        int i = 0;
        int last = n - W;
        if (c is null)
        {
            for (; i <= last; i += W)
                (new Vector<float>(a, aOff + i) + new Vector<float>(b, bOff + i)).CopyTo(dst, dOff + i);
            for (; i < n; i++) dst[dOff + i] = a[aOff + i] + b[bOff + i];
        }
        else
        {
            for (; i <= last; i += W)
                (new Vector<float>(a, aOff + i) + new Vector<float>(b, bOff + i) + new Vector<float>(c, i)).CopyTo(dst, dOff + i);
            for (; i < n; i++) dst[dOff + i] = a[aOff + i] + b[bOff + i] + c[i];
        }
    }

    /// <summary>dst[dOff+i] += w * src[sOff+i] for i in [0,n).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AxpyInto(float[] dst, int dOff, float[] src, int sOff, float w, int n)
    {
        int i = 0;
        var wv = new Vector<float>(w);
        int last = n - W;
        for (; i <= last; i += W)
        {
            var d = new Vector<float>(dst, dOff + i);
            var s = new Vector<float>(src, sOff + i);
            (d + wv * s).CopyTo(dst, dOff + i);
        }
        for (; i < n; i++) dst[dOff + i] += w * src[sOff + i];
    }
}
