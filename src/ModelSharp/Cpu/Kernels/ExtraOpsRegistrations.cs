namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the "extra ops" batch that raises standard ONNX coverage: bitwise logic, the cosine
/// window generators, Einsum / Det, Unique / CenterCropPad / Col2Im, ConvTranspose / GridSample /
/// MaxRoiPool / Upsample, NonMaxSuppression, and the Bernoulli / Multinomial samplers. Each owns its
/// own kernel file; this method keeps <see cref="KernelRegistry.CreateDefault"/> tidy.
/// </summary>
public static class ExtraOpsRegistrations
{
    /// <summary>Adds the extra-ops batch to a registry.</summary>
    public static KernelRegistry AddExtraOps(this KernelRegistry r) => r
        // Bitwise integer ops
        .Register(new ModelSharp.Cpu.Kernels.MathOps.BitwiseAndKernel())
        .Register(new ModelSharp.Cpu.Kernels.MathOps.BitwiseOrKernel())
        .Register(new ModelSharp.Cpu.Kernels.MathOps.BitwiseXorKernel())
        .Register(new ModelSharp.Cpu.Kernels.MathOps.BitwiseNotKernel())
        // Cosine windows
        .Register(new ModelSharp.Cpu.Kernels.MathOps.HannWindowKernel())
        .Register(new ModelSharp.Cpu.Kernels.MathOps.HammingWindowKernel())
        .Register(new ModelSharp.Cpu.Kernels.MathOps.BlackmanWindowKernel())
        // Linear algebra
        .Register(new ModelSharp.Cpu.Kernels.Linear.EinsumKernel())
        .Register(new ModelSharp.Cpu.Kernels.Linear.DetKernel())
        // Shape / data movement
        .Register(new ModelSharp.Cpu.Kernels.Shape.UniqueKernel())
        .Register(new ModelSharp.Cpu.Kernels.Shape.CenterCropPadKernel())
        // NN layers
        .Register(new ModelSharp.Cpu.Kernels.Nn.DropoutKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.ConvTransposeKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.GridSampleKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.MaxRoiPoolKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.Col2ImKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.UpsampleKernel())
        .Register(new ModelSharp.Cpu.Kernels.Nn.NonMaxSuppressionKernel())
        // Samplers
        .Register(new ModelSharp.Cpu.Kernels.Generators.BernoulliKernel())
        .Register(new ModelSharp.Cpu.Kernels.Generators.MultinomialKernel());
}
