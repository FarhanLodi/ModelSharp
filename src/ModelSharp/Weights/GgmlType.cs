namespace ModelSharp.Weights;

/// <summary>
/// The ggml tensor element types used in GGUF files (llama.cpp). The numeric values are the
/// on-disk <c>uint32</c> type tags. ModelSharp natively decodes only the unquantized types
/// (<see cref="F32"/>, <see cref="F16"/>); the quantized block types are surfaced as raw bytes
/// for a future dequantization kernel and must not be reordered.
/// </summary>
public enum GgmlType : uint
{
    /// <summary>32-bit IEEE-754 float.</summary>
    F32 = 0,

    /// <summary>16-bit IEEE-754 half float.</summary>
    F16 = 1,

    /// <summary>Q4_0 4-bit quantized block.</summary>
    Q4_0 = 2,

    /// <summary>Q4_1 4-bit quantized block.</summary>
    Q4_1 = 3,

    /// <summary>Q5_0 5-bit quantized block.</summary>
    Q5_0 = 6,

    /// <summary>Q5_1 5-bit quantized block.</summary>
    Q5_1 = 7,

    /// <summary>Q8_0 8-bit quantized block.</summary>
    Q8_0 = 8,

    /// <summary>Q8_1 8-bit quantized block.</summary>
    Q8_1 = 9,

    /// <summary>Q2_K k-quant block.</summary>
    Q2_K = 10,

    /// <summary>Q3_K k-quant block.</summary>
    Q3_K = 11,

    /// <summary>Q4_K k-quant block.</summary>
    Q4_K = 12,

    /// <summary>Q5_K k-quant block.</summary>
    Q5_K = 13,

    /// <summary>Q6_K k-quant block.</summary>
    Q6_K = 14,

    /// <summary>Q8_K k-quant block.</summary>
    Q8_K = 15,

    /// <summary>IQ2_XXS k-quant block.</summary>
    IQ2_XXS = 16,

    /// <summary>IQ2_XS k-quant block.</summary>
    IQ2_XS = 17,

    /// <summary>IQ3_XXS k-quant block.</summary>
    IQ3_XXS = 18,

    /// <summary>IQ1_S k-quant block.</summary>
    IQ1_S = 19,

    /// <summary>IQ4_NL k-quant block.</summary>
    IQ4_NL = 20,

    /// <summary>IQ3_S k-quant block.</summary>
    IQ3_S = 21,

    /// <summary>IQ2_S k-quant block.</summary>
    IQ2_S = 22,

    /// <summary>IQ4_XS k-quant block.</summary>
    IQ4_XS = 23,

    /// <summary>8-bit signed integer.</summary>
    I8 = 24,

    /// <summary>16-bit signed integer.</summary>
    I16 = 25,

    /// <summary>32-bit signed integer.</summary>
    I32 = 26,

    /// <summary>64-bit signed integer.</summary>
    I64 = 27,

    /// <summary>64-bit IEEE-754 float.</summary>
    F64 = 28,

    /// <summary>IQ1_M k-quant block.</summary>
    IQ1_M = 29,

    /// <summary>BF16 brain float.</summary>
    BF16 = 30,
}
