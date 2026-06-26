using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Test collection that serializes all hardware-CUDA tests. Each <see cref="ModelSharp.Gpu.IlgpuEngine"/>
/// instance allocates its own ILGPU CUDA context + accelerator, and the default xUnit behaviour runs
/// different test classes in parallel — multiple live CUDA contexts exhaust device memory ("out of
/// memory" on accelerator creation). Placing every CUDA test class in one non-parallel collection means
/// at most one CUDA context exists at a time.
/// </summary>
[CollectionDefinition("CudaGpu", DisableParallelization = true)]
public sealed class CudaGpuCollection
{
}
