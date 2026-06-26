using ModelSharp.Cpu.Kernels;

namespace ModelSharp.Cpu.Kernels.Arithmetic;

/// <summary>Elementwise multiply with NumPy-style broadcasting.</summary>
public sealed class MulKernel : BroadcastBinaryKernel
{
    public override string OpType => "Mul";
    protected override float Apply(float a, float b) => a * b;
}
