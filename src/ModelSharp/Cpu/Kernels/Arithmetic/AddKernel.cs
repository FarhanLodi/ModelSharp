using ModelSharp.Cpu.Kernels;

namespace ModelSharp.Cpu.Kernels.Arithmetic;

/// <summary>Elementwise add with NumPy-style broadcasting.</summary>
public sealed class AddKernel : BroadcastBinaryKernel
{
    public override string OpType => "Add";
    protected override float Apply(float a, float b) => a + b;
}
