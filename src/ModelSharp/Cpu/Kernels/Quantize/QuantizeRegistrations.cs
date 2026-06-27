using ModelSharp.Cpu.Kernels.Quantize;

namespace ModelSharp.Cpu.Kernels;

/// <summary>
/// Registers the quantization kernels (Roadmap C3): <c>DequantizeLinear</c>,
/// <c>QuantizeLinear</c>, <c>DynamicQuantizeLinear</c>, and <c>MatMulInteger</c>, plus the
/// block-wise n-bit weight matmul contrib op <c>com.microsoft.MatMulNBits</c> used by the
/// GenAI INT4 LLM exports.
/// </summary>
public static class QuantizeRegistrations
{
    /// <summary>Adds the quantization kernels to a registry.</summary>
    public static KernelRegistry AddQuantize(this KernelRegistry r) => r
        .Register(new DequantizeLinearKernel())
        .Register(new QuantizeLinearKernel())
        .Register(new DynamicQuantizeLinearKernel())
        .Register(new MatMulIntegerKernel())
        .Register(new MatMulNBitsKernel());
}
