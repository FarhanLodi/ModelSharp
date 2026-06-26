namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registration helper for the miscellaneous math / generator ops (Mod, BitShift, and the
/// Random* family). Kept out of <see cref="KernelRegistry.CreateDefault"/> so the op set can be
/// composed à la carte: <c>KernelRegistry.CreateDefault().AddMiscMathOps()</c>.
/// </summary>
public static class MiscMathRegistrations
{
    /// <summary>Registers Mod, BitShift, and the RandomNormal/RandomUniform (+ *Like) generators.</summary>
    public static KernelRegistry AddMiscMathOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.MathOps.ModKernel())
        .Register(new ModelSharp.Cpu.Kernels.MathOps.BitShiftKernel())
        .Register(new ModelSharp.Cpu.Kernels.Generators.RandomNormalKernel())
        .Register(new ModelSharp.Cpu.Kernels.Generators.RandomUniformKernel())
        .Register(new ModelSharp.Cpu.Kernels.Generators.RandomNormalLikeKernel())
        .Register(new ModelSharp.Cpu.Kernels.Generators.RandomUniformLikeKernel());
}
