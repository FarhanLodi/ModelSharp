namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the fused LLM normalization / positional ops that appear in
/// HuggingFace and ONNXRuntime exports: <c>SimplifiedLayerNormalization</c> (RMSNorm),
/// <c>SkipSimplifiedLayerNormalization</c>, and <c>RotaryEmbedding</c>.
/// </summary>
public static class LlmNormRegistrations
{
    /// <summary>Adds the fused LLM norm / RoPE kernels to the registry.</summary>
    public static KernelRegistry AddLlmNormOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.Llm.SimplifiedLayerNormalizationKernel())
        .Register(new ModelSharp.Cpu.Kernels.Llm.SkipSimplifiedLayerNormalizationKernel())
        .Register(new ModelSharp.Cpu.Kernels.Llm.RotaryEmbeddingKernel());
}
