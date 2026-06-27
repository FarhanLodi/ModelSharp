using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Activations;
using ModelSharp.Cpu.Kernels.Arithmetic;
using ModelSharp.Cpu.Kernels.Linear;
using ModelSharp.Cpu.Kernels.Logical;
using ModelSharp.Cpu.Kernels.MathOps;
using ModelSharp.Cpu.Kernels.Nn;
using ModelSharp.Cpu.Kernels.Reduction;
using ModelSharp.Cpu.Kernels.Rnn;
using ModelSharp.Cpu.Kernels.Shape;

namespace ModelSharp.Cpu.Kernels;

/// <summary>Maps an ONNX op type to the kernel that runs it.</summary>
public sealed class KernelRegistry
{
    private readonly Dictionary<string, IKernel> _kernels = new();

    /// <summary>Registers (or replaces) a kernel by its op type.</summary>
    public KernelRegistry Register(IKernel kernel)
    {
        _kernels[kernel.OpType] = kernel;
        return this;
    }

    /// <summary>Looks up the kernel for an op type.</summary>
    public bool TryGet(string opType, out IKernel? kernel) => _kernels.TryGetValue(opType, out kernel);

    /// <summary>Number of registered operators.</summary>
    public int Count => _kernels.Count;

    /// <summary>All kernels implemented so far (Phase 1 CNN core + transformer/sequence building blocks).</summary>
    public static KernelRegistry CreateDefault() => new KernelRegistry()
        // Arithmetic (broadcasting)
        .Register(new AddKernel())
        .Register(new SubKernel())
        .Register(new MulKernel())
        .Register(new DivKernel())
        .Register(new PowKernel())
        // Activations / elementwise
        .Register(new ReluKernel())
        .Register(new SigmoidKernel())
        .Register(new TanhKernel())
        .Register(new ExpKernel())
        .Register(new LogKernel())
        .Register(new SqrtKernel())
        .Register(new AbsKernel())
        .Register(new NegKernel())
        .Register(new ErfKernel())
        .Register(new GeluKernel())
        .Register(new IdentityKernel())
        .Register(new LeakyReluKernel())
        .Register(new ClipKernel())
        .Register(new SoftmaxKernel())
        .Register(new LogSoftmaxKernel())
        // Elementwise unary math
        .Register(new SinKernel())
        .Register(new CosKernel())
        .Register(new TanKernel())
        .Register(new AsinKernel())
        .Register(new AcosKernel())
        .Register(new AtanKernel())
        .Register(new SinhKernel())
        .Register(new CoshKernel())
        .Register(new AsinhKernel())
        .Register(new AcoshKernel())
        .Register(new AtanhKernel())
        .Register(new ReciprocalKernel())
        .Register(new FloorKernel())
        .Register(new CeilKernel())
        .Register(new RoundKernel())
        .Register(new SignKernel())
        // Extra activations
        .Register(new EluKernel())
        .Register(new SeluKernel())
        .Register(new HardSigmoidKernel())
        .Register(new SoftplusKernel())
        .Register(new SoftsignKernel())
        .Register(new MishKernel())
        .Register(new PReluKernel())
        .Register(new HardSwishKernel())
        .Register(new ThresholdedReluKernel())
        .Register(new CeluKernel())
        .Register(new ShrinkKernel())
        // Variadic elementwise
        .Register(new MinKernel())
        .Register(new MaxKernel())
        .Register(new SumKernel())
        .Register(new MeanKernel())
        // Neural-net layers
        .Register(new ConvKernel())
        .Register(new MaxPoolKernel())
        .Register(new AveragePoolKernel())
        .Register(new GlobalAveragePoolKernel())
        .Register(new BatchNormalizationKernel())
        .Register(new LayerNormalizationKernel())
        .Register(new ResizeKernel())
        // Recurrent
        .Register(new LstmKernel())
        .Register(new GruKernel())
        // Linear algebra
        .Register(new MatMulKernel())
        .Register(new GemmKernel())
        // Reduction
        .Register(new ReduceMeanKernel())
        .Register(new ReduceSumKernel())
        .Register(new ReduceMaxKernel())
        .Register(new ReduceMinKernel())
        .Register(new ReduceProdKernel())
        .Register(new ReduceL1Kernel())
        .Register(new ReduceL2Kernel())
        .Register(new ReduceSumSquareKernel())
        .Register(new ReduceLogSumKernel())
        .Register(new ReduceLogSumExpKernel())
        .Register(new ArgMaxKernel())
        .Register(new ArgMinKernel())
        .Register(new TopKKernel())
        .Register(new CumSumKernel())
        // Shape / data movement
        .Register(new ReshapeKernel())
        .Register(new FlattenKernel())
        .Register(new ConcatKernel())
        .Register(new TransposeKernel())
        .Register(new GatherKernel())
        .Register(new UnsqueezeKernel())
        .Register(new SqueezeKernel())
        .Register(new CastKernel())
        .Register(new ShapeKernel())
        .Register(new SizeKernel())
        .Register(new NonZeroKernel())
        .Register(new ConstantKernel())
        .Register(new ConstantOfShapeKernel())
        .Register(new SliceKernel())
        .Register(new ExpandKernel())
        .Register(new SplitKernel())
        .Register(new PadKernel())
        .Register(new TileKernel())
        .Register(new TriluKernel())
        .Register(new RangeKernel())
        .Register(new GatherElementsKernel())
        .Register(new ScatterElementsKernel())
        .Register(new GatherNDKernel())
        .Register(new ScatterNDKernel())
        // Logical / comparison
        .Register(new WhereKernel())
        .Register(new EqualKernel())
        .Register(new LessKernel())
        .Register(new GreaterKernel())
        .Register(new GreaterOrEqualKernel())
        .Register(new LessOrEqualKernel())
        .Register(new NotKernel())
        .Register(new AndKernel())
        .Register(new OrKernel())
        .Register(new XorKernel())
        .Register(new IsNaNKernel())
        .Register(new IsInfKernel())
        // Extended op families (registered via per-family extension methods so each owns its own file)
        .AddQuantize()              // C3: DequantizeLinear/QuantizeLinear/DynamicQuantizeLinear/MatMulInteger
        .AddLlmNormOps()            // C6: SimplifiedLayerNormalization (RMSNorm) / Skip variant / RotaryEmbedding
        .AddAttentionContribOps()   // C6: MultiHeadAttention / GroupQueryAttention
        .AddNormalizationOps()      // InstanceNormalization/GroupNormalization/MVN/LpNormalization/LRN
        .AddDataMovementOps()       // OneHot/EyeLike/Compress/DepthToSpace/SpaceToDepth/ReverseSequence
        .AddMiscMathOps()           // Mod/BitShift/RandomNormal/RandomUniform(+Like)
        .AddPoolingExtraOps()       // GlobalMaxPool/LpPool/GlobalLpPool/Hardmax/MaxUnpool
        .AddExtraOps()              // Bitwise{And,Or,Xor,Not}/{Hann,Hamming,Blackman}Window/Einsum/Det/
                                    // Unique/CenterCropPad/Dropout/ConvTranspose/GridSample/MaxRoiPool/
                                    // Col2Im/Upsample/NonMaxSuppression/Bernoulli/Multinomial
        .AddControlFlowOps();       // If/Loop/Scan (subgraph execution via the GraphContext runner hook)
}
