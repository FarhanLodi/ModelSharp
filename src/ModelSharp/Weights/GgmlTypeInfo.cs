namespace ModelSharp.Weights;

/// <summary>
/// Sizing traits for <see cref="GgmlType"/>. ggml stores tensors in blocks of
/// <see cref="BlockSize"/> elements occupying <see cref="TypeSize"/> bytes; for the
/// unquantized scalar types a "block" is a single element. The on-disk byte length of a
/// tensor is therefore <c>(elementCount / blockSize) * typeSize</c>.
/// </summary>
internal static class GgmlTypeInfo
{
    /// <summary>Returns <c>true</c> for quantized block types that ModelSharp does not dequantize.</summary>
    public static bool IsQuantized(GgmlType type) => type switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.F64 or GgmlType.BF16 or
        GgmlType.I8 or GgmlType.I16 or GgmlType.I32 or GgmlType.I64 => false,
        _ => true,
    };

    /// <summary>Number of elements per ggml block (1 for unquantized scalar types).</summary>
    public static int BlockSize(GgmlType type) => type switch
    {
        GgmlType.F32 or GgmlType.F16 or GgmlType.F64 or GgmlType.BF16 or
        GgmlType.I8 or GgmlType.I16 or GgmlType.I32 or GgmlType.I64 => 1,

        // QK = 32 family.
        GgmlType.Q4_0 or GgmlType.Q4_1 or GgmlType.Q5_0 or GgmlType.Q5_1 or
        GgmlType.Q8_0 or GgmlType.Q8_1 or GgmlType.IQ4_NL => 32,

        // k-quant / super-block family: QK_K = 256.
        GgmlType.Q2_K or GgmlType.Q3_K or GgmlType.Q4_K or GgmlType.Q5_K or
        GgmlType.Q6_K or GgmlType.Q8_K or GgmlType.IQ2_XXS or GgmlType.IQ2_XS or
        GgmlType.IQ3_XXS or GgmlType.IQ1_S or GgmlType.IQ3_S or GgmlType.IQ2_S or
        GgmlType.IQ4_XS or GgmlType.IQ1_M => 256,

        _ => 0,
    };

    /// <summary>Bytes occupied by one ggml block of the given type.</summary>
    public static int TypeSize(GgmlType type) => type switch
    {
        GgmlType.F32 or GgmlType.I32 => 4,
        GgmlType.F16 or GgmlType.I16 or GgmlType.BF16 => 2,
        GgmlType.I8 => 1,
        GgmlType.F64 or GgmlType.I64 => 8,

        GgmlType.Q4_0 => 18,   // 2 (scale fp16) + 16 (32 * 4 bits)
        GgmlType.Q4_1 => 20,   // 2 + 2 + 16
        GgmlType.Q5_0 => 22,   // 2 + 4 + 16
        GgmlType.Q5_1 => 24,   // 2 + 2 + 4 + 16
        GgmlType.Q8_0 => 34,   // 2 + 32
        GgmlType.Q8_1 => 36,   // 4 + 32

        GgmlType.Q2_K => 84,
        GgmlType.Q3_K => 110,
        GgmlType.Q4_K => 144,
        GgmlType.Q5_K => 176,
        GgmlType.Q6_K => 210,
        GgmlType.Q8_K => 292,
        GgmlType.IQ2_XXS => 66,
        GgmlType.IQ2_XS => 74,
        GgmlType.IQ3_XXS => 98,
        GgmlType.IQ1_S => 50,
        GgmlType.IQ4_NL => 18,
        GgmlType.IQ3_S => 110,
        GgmlType.IQ2_S => 82,
        GgmlType.IQ4_XS => 136,
        GgmlType.IQ1_M => 56,

        _ => 0,
    };

    /// <summary>
    /// Computes the on-disk byte length of a tensor of <paramref name="elementCount"/> elements.
    /// Returns <c>false</c> if the type is unknown or the element count is not a multiple of the
    /// block size (a malformed file).
    /// </summary>
    public static bool TryByteLength(GgmlType type, long elementCount, out long byteLength)
    {
        byteLength = 0;
        int block = BlockSize(type);
        int size = TypeSize(type);
        if (block == 0 || size == 0) return false;
        if (elementCount % block != 0) return false;
        byteLength = (elementCount / block) * size;
        return true;
    }
}
