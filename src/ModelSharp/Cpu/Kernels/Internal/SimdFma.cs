using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace ModelSharp.Cpu.Kernels.Internal;

/// <summary>
/// Single-source fused-multiply-add for the hot float32 SIMD accumulation loops
/// (dot / GEMM / axpy). The managed <see cref="Vector{T}"/> abstraction writes accumulation as a
/// separate multiply then add (<c>c + a * b</c>) which the JIT does NOT contract into a hardware
/// FMA, so every contraction loop loses ~1.5–2× of achievable FLOPs.
///
/// <para>This helper keeps the <see cref="Vector{T}"/> data type used throughout the kernels but,
/// when the running CPU exposes a real FMA unit, reinterprets the operands to the fixed-width
/// <see cref="Vector256{T}"/> / <see cref="Vector512{T}"/> (x86 FMA / AVX-512F) or
/// <see cref="Vector64{T}"/>-family (ARM <see cref="AdvSimd"/>) and issues a genuine
/// <c>FusedMultiplyAdd</c>. <see cref="Vector{T}.Count"/> equals 8 under AVX2 and 16 under AVX-512,
/// so the reinterpret width always matches. When no FMA path applies it falls back to the exact
/// same <c>a * b + c</c> the kernels used before, so behaviour is byte-for-byte unchanged on
/// non-FMA hardware and the whole thing stays AOT-safe (the intrinsic classes are guarded by their
/// <c>IsSupported</c> checks, which the JIT/AOT folds to constants).</para>
///
/// <para>FMA computes <c>a*b+c</c> with a single rounding instead of two, so results can shift by a
/// few ULP versus the split form. That is well inside the engine's ~1e-2 ORT tolerance, and the
/// kernels already document that SIMD reassociation moves results by a few ULP.</para>
/// </summary>
internal static class SimdFma
{
    /// <summary>
    /// Returns <c>a * b + c</c> using a hardware fused-multiply-add when the running CPU supports one
    /// (single rounding), otherwise the plain <c>a * b + c</c> fallback (double rounding, bit-identical
    /// to the pre-FMA kernels). The <see cref="Vector{T}"/> width must be 256 or 512 bits on x86 for the
    /// fused path to engage; any other width takes the fallback.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> MulAdd(Vector<float> a, Vector<float> b, Vector<float> c)
    {
        if (Vector<float>.Count == Vector256<float>.Count)
        {
            if (Fma.IsSupported)
                return Fma.MultiplyAdd(a.AsVector256(), b.AsVector256(), c.AsVector256()).AsVector();
        }
        else if (Vector<float>.Count == Vector512<float>.Count)
        {
            if (Avx512F.IsSupported)
                return Avx512F.FusedMultiplyAdd(a.AsVector512(), b.AsVector512(), c.AsVector512()).AsVector();
        }
        else if (Vector<float>.Count == Vector128<float>.Count)
        {
            // ARM NEON's narrowest SIMD width for the managed Vector<T> is 128-bit; AdvSimd.FusedMultiplyAdd
            // computes addend + left * right, so the accumulator c is the first argument.
            if (AdvSimd.IsSupported)
                return AdvSimd.FusedMultiplyAdd(c.AsVector128(), a.AsVector128(), b.AsVector128()).AsVector();
        }
        return a * b + c;
    }
}
