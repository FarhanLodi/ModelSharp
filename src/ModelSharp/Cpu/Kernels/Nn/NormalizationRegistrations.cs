namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the normalization-family ONNX kernels (InstanceNormalization,
/// GroupNormalization, MeanVarianceNormalization, LpNormalization, LRN).
/// </summary>
public static class NormalizationRegistrations
{
    /// <summary>Adds all normalization kernels to the registry.</summary>
    public static KernelRegistry AddNormalizationOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.Nn.InstanceNormalizationKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.GroupNormalizationKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.MeanVarianceNormalizationKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.LpNormalizationKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.LRNKernel());
}
