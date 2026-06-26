namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the shape / data-movement kernels: <c>OneHot</c>, <c>EyeLike</c>, <c>Compress</c>,
/// <c>DepthToSpace</c>, <c>SpaceToDepth</c>, and <c>ReverseSequence</c>.
/// </summary>
public static class DataMovementRegistrations
{
    /// <summary>Adds the shape / data-movement kernels to a registry.</summary>
    public static KernelRegistry AddDataMovementOps(this KernelRegistry r) => r
        .Register(new ModelSharp.Cpu.Kernels.Shape.OneHotKernel())
        .Register(new ModelSharp.Cpu.Kernels.Shape.EyeLikeKernel())
        .Register(new ModelSharp.Cpu.Kernels.Shape.CompressKernel())
        .Register(new ModelSharp.Cpu.Kernels.Shape.DepthToSpaceKernel())
        .Register(new ModelSharp.Cpu.Kernels.Shape.SpaceToDepthKernel())
        .Register(new ModelSharp.Cpu.Kernels.Shape.ReverseSequenceKernel());
}
