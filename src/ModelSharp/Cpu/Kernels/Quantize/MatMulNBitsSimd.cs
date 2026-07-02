using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace ModelSharp.Cpu.Kernels.Quantize;

/// <summary>
/// SIMD primitives for the block-wise INT4 quantized matmul (<c>MatMulNBits</c>). Two families:
/// <list type="number">
/// <item><description><b>Exact-preserving default</b> (<see cref="DequantBlock4"/> +
/// <see cref="FmaAccumulate"/>): vectorized nibble unpack and fused FMA dot that keeps the exact
/// fp32 dequant math <c>(q − zp)·scale</c> so the tight parity tests stay green. The only numeric
/// difference from a left-to-right scalar dot is accumulation order (a few ULP), identical in spirit
/// to the existing <see cref="Linear.MatMulParallel.Dot"/>.</description></item>
/// <item><description><b>Opt-in W4A8</b> (<see cref="DotU8xU8"/>): a dynamically-int8-quantized
/// activation dotted against the packed int4 weights via VNNI (<c>vpdpbusd</c>) or maddubs
/// (<c>vpmaddubsw</c>), gated by an env flag and OFF by default. Lower precision, much less traffic.
/// </description></item>
/// </list>
/// All routines take flat arrays at explicit offsets so they can run inside <c>Parallel.For</c>
/// bodies (which cannot capture <c>Span&lt;T&gt;</c>).
/// </summary>
internal static class MatMulNBitsSimd
{
    private static readonly int W = Vector<float>.Count;

    // ----------------------------------------------------------------------------------------------
    // Path 1 — exact-preserving default
    // ----------------------------------------------------------------------------------------------

    /// <summary>
    /// Dequantizes <paramref name="count"/> packed 4-bit codes starting at the low nibble of
    /// <paramref name="src"/>[<paramref name="blobBase"/>] into <paramref name="dst"/> as
    /// <c>(q − zp)·scale</c>. Codes are laid out least-significant-first (even index → low nibble,
    /// odd index → high nibble). Uses an AVX2 mask/shift unpack when available; otherwise a tight
    /// scalar unpack. Produces bit-identical dequant values to the scalar reference (same fp32 ops).
    /// </summary>
    public static void DequantBlock4(
        ReadOnlySpan<byte> src, int blobBase, int count, float zp, float scale, Span<float> dst)
    {
        int i = 0;

        // SIMD mask/shift unpack: 16 packed bytes -> 32 codes per iteration. Low nibbles are the
        // even codes, high nibbles the odd codes; Sse2.UnpackLow/High interleave them back into
        // natural code order [lo0,hi0,lo1,hi1,…]. We then widen the 8-bit codes to int32, convert to
        // float, and apply (q - zp) * scale in vectors of 8. All work is 128-bit so no cross-lane
        // garbage can leak in (the packed row is loaded 16 bytes at a time).
        if (Avx2.IsSupported && Sse2.IsSupported && count >= 32)
        {
            var lowMask = Vector128.Create((byte)0x0F);
            var zpv = Vector256.Create(zp);
            var scalev = Vector256.Create(scale);
            for (; i + 32 <= count; i += 32)
            {
                Vector128<byte> packed = Vector128.LoadUnsafe(
                    ref Unsafe.AsRef(in src[blobBase + (i >> 1)]));                 // 16 packed bytes
                Vector128<byte> lo = Sse2.And(packed, lowMask);                     // even codes 0,2,4,…
                Vector128<byte> hi = Sse2.And(
                    Sse2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), lowMask); // odd codes 1,3,5,…
                Vector128<byte> first16 = Sse2.UnpackLow(lo, hi);   // codes 0..15
                Vector128<byte> last16 = Sse2.UnpackHigh(lo, hi);   // codes 16..31
                DequantStore8(first16, 0, zpv, scalev, dst, i + 0);
                DequantStore8(first16, 8, zpv, scalev, dst, i + 8);
                DequantStore8(last16, 0, zpv, scalev, dst, i + 16);
                DequantStore8(last16, 8, zpv, scalev, dst, i + 24);
            }
        }

        // Scalar-unpack + vector-arithmetic remainder (also the whole path when AVX2 is absent).
        {
            int width = W;
            var zpv = new Vector<float>(zp);
            var scalev = new Vector<float>(scale);
            Span<float> tmp = stackalloc float[width];
            for (; i + width <= count; i += width)
            {
                for (int j = 0; j < width; j++)
                {
                    int idx = i + j;
                    int b = src[blobBase + (idx >> 1)];
                    tmp[j] = ((idx & 1) == 0) ? (b & 0x0F) : (b >> 4) & 0x0F;
                }
                var qv = new Vector<float>(tmp);
                ((qv - zpv) * scalev).CopyTo(dst.Slice(i, width));
            }
        }

        for (; i < count; i++)
        {
            int b = src[blobBase + (i >> 1)];
            int q = ((i & 1) == 0) ? (b & 0x0F) : (b >> 4) & 0x0F;
            dst[i] = (q - zp) * scale;
        }
    }

    /// <summary>Widens 8 code bytes at <paramref name="byteOff"/> of <paramref name="v"/> to float,
    /// applies <c>(q − zp)·scale</c>, and stores them at <paramref name="dstOff"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DequantStore8(
        Vector128<byte> v, int byteOff, Vector256<float> zpv, Vector256<float> scalev,
        Span<float> dst, int dstOff)
    {
        // Move the 8 target bytes into the low 8 lanes, widen u8 -> i32 (256-bit, 8 ints).
        Vector128<byte> sel = byteOff == 0 ? v : Sse2.ShiftRightLogical128BitLane(v, 8);
        Vector256<int> wide = Avx2.ConvertToVector256Int32(sel); // widens low 8 bytes to 8 int32
        Vector256<float> q = Avx.ConvertToVector256Single(wide);
        Vector256<float> res = Avx.Multiply(Avx.Subtract(q, zpv), scalev);
        res.StoreUnsafe(ref dst[dstOff]);
    }

    /// <summary>
    /// Dequantizes <paramref name="count"/> packed 8-bit codes into <paramref name="dst"/> as
    /// <c>(q − zp)·scale</c>. One code per byte.
    /// </summary>
    public static void DequantBlock8(
        ReadOnlySpan<byte> src, int blobBase, int count, float zp, float scale, Span<float> dst)
    {
        int i = 0;
        int width = W;
        var zpv = new Vector<float>(zp);
        var scalev = new Vector<float>(scale);
        Span<float> tmp = stackalloc float[width];
        for (; i + width <= count; i += width)
        {
            for (int j = 0; j < width; j++) tmp[j] = src[blobBase + i + j];
            var qv = new Vector<float>(tmp);
            ((qv - zpv) * scalev).CopyTo(dst.Slice(i, width));
        }
        for (; i < count; i++) dst[i] = (src[blobBase + i] - zp) * scale;
    }

    /// <summary>
    /// Fused multiply-add of a dequantized weight block against an activation slice:
    /// <c>Σ a[aOff+i] · w[i]</c> over <paramref name="n"/> lanes, added to an existing scalar
    /// <paramref name="acc"/>. Uses hardware FMA (<c>Fma</c> / <c>Avx512F</c>) when available,
    /// otherwise the portable <see cref="Vector{T}"/> path. Four independent accumulators break the
    /// FMA-latency dependency chain.
    /// </summary>
    public static float FmaAccumulate(float acc, float[] a, int aOff, float[] w, int wOff, int n)
    {
        int i = 0;

        if (Avx512F.IsSupported && n >= 16)
        {
            Vector512<float> s0 = Vector512<float>.Zero, s1 = Vector512<float>.Zero,
                             s2 = Vector512<float>.Zero, s3 = Vector512<float>.Zero;
            int last4 = n - 64;
            for (; i <= last4; i += 64)
            {
                s0 = Avx512F.FusedMultiplyAdd(V512(a, aOff + i), V512(w, wOff + i), s0);
                s1 = Avx512F.FusedMultiplyAdd(V512(a, aOff + i + 16), V512(w, wOff + i + 16), s1);
                s2 = Avx512F.FusedMultiplyAdd(V512(a, aOff + i + 32), V512(w, wOff + i + 32), s2);
                s3 = Avx512F.FusedMultiplyAdd(V512(a, aOff + i + 48), V512(w, wOff + i + 48), s3);
            }
            var s = Avx512F.Add(Avx512F.Add(s0, s1), Avx512F.Add(s2, s3));
            int last = n - 16;
            for (; i <= last; i += 16)
                s = Avx512F.FusedMultiplyAdd(V512(a, aOff + i), V512(w, wOff + i), s);
            acc += Vector512.Sum(s);
        }
        else if (Fma.IsSupported && n >= 8)
        {
            Vector256<float> s0 = Vector256<float>.Zero, s1 = Vector256<float>.Zero,
                             s2 = Vector256<float>.Zero, s3 = Vector256<float>.Zero;
            int last4 = n - 32;
            for (; i <= last4; i += 32)
            {
                s0 = Fma.MultiplyAdd(V256(a, aOff + i), V256(w, wOff + i), s0);
                s1 = Fma.MultiplyAdd(V256(a, aOff + i + 8), V256(w, wOff + i + 8), s1);
                s2 = Fma.MultiplyAdd(V256(a, aOff + i + 16), V256(w, wOff + i + 16), s2);
                s3 = Fma.MultiplyAdd(V256(a, aOff + i + 24), V256(w, wOff + i + 24), s3);
            }
            var s = Avx.Add(Avx.Add(s0, s1), Avx.Add(s2, s3));
            int last = n - 8;
            for (; i <= last; i += 8)
                s = Fma.MultiplyAdd(V256(a, aOff + i), V256(w, wOff + i), s);
            acc += Vector256.Sum(s);
        }
        else
        {
            int width = W;
            Vector<float> v0 = Vector<float>.Zero, v1 = Vector<float>.Zero,
                          v2 = Vector<float>.Zero, v3 = Vector<float>.Zero;
            int last4 = n - 4 * width;
            for (; i <= last4; i += 4 * width)
            {
                v0 += new Vector<float>(a, aOff + i) * new Vector<float>(w, wOff + i);
                v1 += new Vector<float>(a, aOff + i + width) * new Vector<float>(w, wOff + i + width);
                v2 += new Vector<float>(a, aOff + i + 2 * width) * new Vector<float>(w, wOff + i + 2 * width);
                v3 += new Vector<float>(a, aOff + i + 3 * width) * new Vector<float>(w, wOff + i + 3 * width);
            }
            var vs = (v0 + v1) + (v2 + v3);
            int last = n - width;
            for (; i <= last; i += width)
                vs += new Vector<float>(a, aOff + i) * new Vector<float>(w, wOff + i);
            acc += Vector.Dot(vs, Vector<float>.One);
        }

        for (; i < n; i++) acc += a[aOff + i] * w[wOff + i];
        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> V256(float[] a, int off)
        => Vector256.LoadUnsafe(ref Unsafe.AsRef(in a[off]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> V512(float[] a, int off)
        => Vector512.LoadUnsafe(ref Unsafe.AsRef(in a[off]));

    // ----------------------------------------------------------------------------------------------
    // Path 2 — opt-in W4A8 (int8 × int4 -> int32), OFF by default
    // ----------------------------------------------------------------------------------------------

    /// <summary>True when the CPU offers an int8 dot-product path for the W4A8 kernel.</summary>
    public static bool W4A8Available => AvxVnni.IsSupported || Avx2.IsSupported;

    /// <summary>
    /// Quantizes <paramref name="count"/> activations <c>a[aOff..]</c> to signed int8 with a single
    /// per-block symmetric scale (absmax / 127) and returns that scale. Writes the int8 codes to
    /// <paramref name="dst"/> and their running sum to <paramref name="sumOut"/> (needed for the
    /// zero-point compensation term Σ_k a_q · zp_w).
    /// </summary>
    public static float QuantizeActivationInt8(
        float[] a, int aOff, int count, Span<sbyte> dst, out int sumOut)
    {
        float absmax = 0f;
        for (int i = 0; i < count; i++)
        {
            float v = MathF.Abs(a[aOff + i]);
            if (v > absmax) absmax = v;
        }
        if (absmax == 0f)
        {
            dst.Slice(0, count).Clear();
            sumOut = 0;
            return 0f;
        }
        float scale = absmax / 127f;
        float inv = 127f / absmax;
        int sum = 0;
        for (int i = 0; i < count; i++)
        {
            int q = (int)MathF.Round(a[aOff + i] * inv);
            if (q > 127) q = 127; else if (q < -128) q = -128;
            dst[i] = (sbyte)q;
            sum += q;
        }
        sumOut = sum;
        return scale;
    }

    /// <summary>
    /// Unsigned int8 (activation, offset by +128) × int4 weight code (0..15) dot product over
    /// <paramref name="count"/> lanes, returning the raw int32 accumulation <c>Σ aq · q</c>.
    /// The caller removes the +128 bias afterward. Uses AVX-VNNI <c>vpdpbusd</c>
    /// (<see cref="AvxVnni.MultiplyWideningAndAdd(Vector256{int}, Vector256{byte}, Vector256{sbyte})"/>)
    /// when available, else AVX2 <c>vpmaddubsw</c>+<c>vpmaddwd</c>, else scalar.
    /// <paramref name="wCodes"/> holds the already-unpacked 0..15 weight codes as bytes (fit sbyte).
    /// </summary>
    public static int DotU8xU8(ReadOnlySpan<byte> aOffset, ReadOnlySpan<byte> wCodes, int count)
    {
        int i = 0;
        int acc = 0;

        if (AvxVnni.IsSupported && count >= 32)
        {
            Vector256<int> s = Vector256<int>.Zero;
            for (; i + 32 <= count; i += 32)
            {
                Vector256<byte> av = Vector256.LoadUnsafe(ref Unsafe.AsRef(in aOffset[i]));
                Vector256<sbyte> wv = Vector256.LoadUnsafe(ref Unsafe.AsRef(in wCodes[i])).AsSByte();
                // vpdpbusd: u8 * s8 -> s32 fused. Weights 0..15 are valid sbyte, so exact.
                s = AvxVnni.MultiplyWideningAndAdd(s, av, wv);
            }
            acc += Vector256.Sum(s);
        }
        else if (Avx2.IsSupported && count >= 32)
        {
            var one16 = Vector256.Create((short)1);
            Vector256<int> s = Vector256<int>.Zero;
            for (; i + 32 <= count; i += 32)
            {
                Vector256<byte> av = Vector256.LoadUnsafe(ref Unsafe.AsRef(in aOffset[i]));
                Vector256<sbyte> wv = Vector256.LoadUnsafe(ref Unsafe.AsRef(in wCodes[i])).AsSByte();
                // maddubs: u8 * s8 -> s16 with adjacent-pair add; weights (0..15) are non-negative
                // so treating them as signed is exact. Then widen s16 pairs to s32 via madd·1.
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(av, wv);
                s = Avx2.Add(s, Avx2.MultiplyAddAdjacent(prod, one16));
            }
            acc += Vector256.Sum(s);
        }

        for (; i < count; i++) acc += aOffset[i] * wCodes[i];
        return acc;
    }
}
