using System;
using ModelSharp.Cpu.Kernels;

namespace ModelSharp.Cpu.Kernels.MathOps;

/// <summary>Elementwise sine.</summary>
public sealed class SinKernel : UnaryKernel { public override string OpType => "Sin"; protected override float Apply(float x) => MathF.Sin(x); }

/// <summary>Elementwise cosine.</summary>
public sealed class CosKernel : UnaryKernel { public override string OpType => "Cos"; protected override float Apply(float x) => MathF.Cos(x); }

/// <summary>Elementwise tangent.</summary>
public sealed class TanKernel : UnaryKernel { public override string OpType => "Tan"; protected override float Apply(float x) => MathF.Tan(x); }

/// <summary>Elementwise arcsine.</summary>
public sealed class AsinKernel : UnaryKernel { public override string OpType => "Asin"; protected override float Apply(float x) => MathF.Asin(x); }

/// <summary>Elementwise arccosine.</summary>
public sealed class AcosKernel : UnaryKernel { public override string OpType => "Acos"; protected override float Apply(float x) => MathF.Acos(x); }

/// <summary>Elementwise arctangent.</summary>
public sealed class AtanKernel : UnaryKernel { public override string OpType => "Atan"; protected override float Apply(float x) => MathF.Atan(x); }

/// <summary>Elementwise hyperbolic sine.</summary>
public sealed class SinhKernel : UnaryKernel { public override string OpType => "Sinh"; protected override float Apply(float x) => MathF.Sinh(x); }

/// <summary>Elementwise hyperbolic cosine.</summary>
public sealed class CoshKernel : UnaryKernel { public override string OpType => "Cosh"; protected override float Apply(float x) => MathF.Cosh(x); }

/// <summary>Elementwise inverse hyperbolic sine.</summary>
public sealed class AsinhKernel : UnaryKernel { public override string OpType => "Asinh"; protected override float Apply(float x) => MathF.Asinh(x); }

/// <summary>Elementwise inverse hyperbolic cosine.</summary>
public sealed class AcoshKernel : UnaryKernel { public override string OpType => "Acosh"; protected override float Apply(float x) => MathF.Acosh(x); }

/// <summary>Elementwise inverse hyperbolic tangent.</summary>
public sealed class AtanhKernel : UnaryKernel { public override string OpType => "Atanh"; protected override float Apply(float x) => MathF.Atanh(x); }

/// <summary>Elementwise reciprocal (1/x).</summary>
public sealed class ReciprocalKernel : UnaryKernel { public override string OpType => "Reciprocal"; protected override float Apply(float x) => 1f / x; }

/// <summary>Elementwise floor (round toward negative infinity).</summary>
public sealed class FloorKernel : UnaryKernel { public override string OpType => "Floor"; protected override float Apply(float x) => MathF.Floor(x); }

/// <summary>Elementwise ceiling (round toward positive infinity).</summary>
public sealed class CeilKernel : UnaryKernel { public override string OpType => "Ceil"; protected override float Apply(float x) => MathF.Ceiling(x); }

/// <summary>Elementwise round-half-to-even (banker's rounding), per ONNX <c>Round</c>.</summary>
public sealed class RoundKernel : UnaryKernel { public override string OpType => "Round"; protected override float Apply(float x) => MathF.Round(x, MidpointRounding.ToEven); }

/// <summary>Elementwise sign: -1, 0, or +1 (NaN is preserved).</summary>
public sealed class SignKernel : UnaryKernel { public override string OpType => "Sign"; protected override float Apply(float x) => x > 0f ? 1f : (x < 0f ? -1f : x); }
