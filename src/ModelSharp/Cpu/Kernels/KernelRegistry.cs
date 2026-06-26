using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Activations;
using ModelSharp.Cpu.Kernels.Arithmetic;
using ModelSharp.Cpu.Kernels.Linear;
using ModelSharp.Cpu.Kernels.Logical;
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
        // Neural-net layers
        .Register(new ConvKernel())
        .Register(new MaxPoolKernel())
        .Register(new GlobalAveragePoolKernel())
        .Register(new BatchNormalizationKernel())
        .Register(new LayerNormalizationKernel())
        // Recurrent
        .Register(new LstmKernel())
        .Register(new GruKernel())
        // Linear algebra
        .Register(new MatMulKernel())
        .Register(new GemmKernel())
        // Reduction
        .Register(new ReduceMeanKernel())
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
        .Register(new ConstantKernel())
        .Register(new ConstantOfShapeKernel())
        .Register(new SliceKernel())
        .Register(new ExpandKernel())
        // Logical / comparison
        .Register(new WhereKernel())
        .Register(new EqualKernel())
        .Register(new LessKernel())
        .Register(new GreaterKernel());
}
