using System;
using ModelSharp.Cpu.Kernels;

namespace ModelSharp.Cpu.Kernels.Arithmetic;

public sealed class SubKernel : BroadcastBinaryKernel { public override string OpType => "Sub"; protected override float Apply(float a, float b) => a - b; }
public sealed class DivKernel : BroadcastBinaryKernel { public override string OpType => "Div"; protected override float Apply(float a, float b) => a / b; }
public sealed class PowKernel : BroadcastBinaryKernel { public override string OpType => "Pow"; protected override float Apply(float a, float b) => MathF.Pow(a, b); }
