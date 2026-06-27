namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the "extra ops" batch that raises standard ONNX coverage: bitwise logic, the cosine
/// window generators, Einsum / Det, Unique / CenterCropPad / Col2Im, ConvTranspose / GridSample /
/// MaxRoiPool / Upsample, NonMaxSuppression, the Bernoulli / Multinomial samplers, and the
/// signal-processing family (DFT / STFT / MelWeightMatrix). Each owns its own kernel file; this
/// method keeps <see cref="KernelRegistry.CreateDefault"/> tidy.
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
        .Register(new ModelSharp.Cpu.Kernels.Generators.MultinomialKernel())
        // Signal processing (opset 17)
        .Register(new ModelSharp.Cpu.Kernels.Signal.DftKernel())
        .Register(new ModelSharp.Cpu.Kernels.Signal.StftKernel())
        .Register(new ModelSharp.Cpu.Kernels.Signal.MelWeightMatrixKernel())
        // Contrib / legacy elementwise activations real exports hit
        .Register(new ModelSharp.Cpu.Kernels.Activations.FastGeluKernel())
        .Register(new ModelSharp.Cpu.Kernels.Activations.BiasGeluKernel())
        .Register(new ModelSharp.Cpu.Kernels.Activations.QuickGeluKernel())
        .Register(new ModelSharp.Cpu.Kernels.Activations.AffineKernel())
        .Register(new ModelSharp.Cpu.Kernels.Activations.ImageScalerKernel());

    /// <summary>
    /// Adds the ONNX control-flow ops — <c>If</c>, <c>Loop</c>, and <c>Scan</c> — which execute
    /// nested subgraphs (GRAPH attributes) via the <see cref="GraphContext"/> subgraph-runner hook.
    /// </summary>
    public static KernelRegistry AddControlFlowOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.ControlFlow.IfKernel())
        .Register(new ModelSharp.Cpu.Kernels.ControlFlow.LoopKernel())
        .Register(new ModelSharp.Cpu.Kernels.ControlFlow.ScanKernel());

    /// <summary>
    /// Adds the ONNX <c>Sequence*</c> and <c>Optional*</c> op families. These produce/consume the
    /// non-tensor runtime values (<see cref="ModelSharp.Cpu.Kernels.Sequence.SeqValue"/>) carried in
    /// the <see cref="GraphContext"/>'s parallel sequence/optional value map — graph inputs/outputs
    /// stay tensors, so the public <c>Run</c> contract is unchanged.
    /// </summary>
    public static KernelRegistry AddSequenceOps(this KernelRegistry r) => r
        // Sequence value type
        .Register(new ModelSharp.Cpu.Kernels.Sequence.SequenceEmptyKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.SequenceConstructKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.SequenceInsertKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.SequenceEraseKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.SequenceAtKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.SequenceLengthKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.SplitToSequenceKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.ConcatFromSequenceKernel())
        // Optional value type
        .Register(new ModelSharp.Cpu.Kernels.Sequence.OptionalKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.OptionalGetElementKernel())
        .Register(new ModelSharp.Cpu.Kernels.Sequence.OptionalHasElementKernel());

    /// <summary>
    /// Adds the <c>QLinear*</c> quantized op family (common in quantized ONNX exports):
    /// <c>QLinearConv</c>, <c>QLinearMatMul</c>, <c>QLinearAdd</c>, <c>QLinearMul</c>,
    /// <c>QLinearGlobalAveragePool</c>, plus <c>ConvInteger</c>. Each dequantizes its integer
    /// inputs, runs the existing float kernel math, and requantizes — reusing the shared
    /// <see cref="ModelSharp.Cpu.Kernels.Quantize.QuantizeOps"/> helpers.
    /// </summary>
    public static KernelRegistry AddQLinearOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.Quantize.QLinearMatMulKernel())
        .Register(new ModelSharp.Cpu.Kernels.Quantize.QLinearAddKernel())
        .Register(new ModelSharp.Cpu.Kernels.Quantize.QLinearMulKernel())
        .Register(new ModelSharp.Cpu.Kernels.Quantize.QLinearConvKernel())
        .Register(new ModelSharp.Cpu.Kernels.Quantize.ConvIntegerKernel())
        .Register(new ModelSharp.Cpu.Kernels.Quantize.QLinearGlobalAveragePoolKernel());
}
