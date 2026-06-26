namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the extra pooling / argmax operators (<c>GlobalMaxPool</c>, <c>LpPool</c>,
/// <c>GlobalLpPool</c>, <c>Hardmax</c>, <c>MaxUnpool</c>) onto a <see cref="KernelRegistry"/>.
/// </summary>
public static class PoolingExtraRegistrations
{
    /// <summary>Adds the extra pooling op kernels to <paramref name="r"/> and returns it for chaining.</summary>
    public static KernelRegistry AddPoolingExtraOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.Nn.GlobalMaxPoolKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.LpPoolKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.GlobalLpPoolKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.HardmaxKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.MaxUnpoolKernel());
}
