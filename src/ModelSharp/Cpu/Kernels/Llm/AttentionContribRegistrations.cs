namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the ONNXRuntime contrib attention kernels: the unpacked-Q/K/V
/// <c>MultiHeadAttention</c> and the causal grouped-query <c>GroupQueryAttention</c>.
/// </summary>
public static class AttentionContribRegistrations
{
    /// <summary>Adds the contrib attention kernels to the registry.</summary>
    public static KernelRegistry AddAttentionContribOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.Llm.MultiHeadAttentionKernel())
        .Register(new ModelSharp.Cpu.Kernels.Llm.GroupQueryAttentionKernel());
}
