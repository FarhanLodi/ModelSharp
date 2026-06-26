using System;
using ModelSharp.Tensors;

namespace ModelSharp.Weights;

/// <summary>
/// Reconstructs full-precision <see cref="Tensor{Single}"/> weight matrices from the bit-packed
/// quantized Linear layers produced by the two dominant HuggingFace post-training-quantization
/// schemes, <b>GPTQ</b> and <b>AWQ</b>. Both store a Linear layer as a small family of tensors
/// sharing a common <c>prefix</c> (e.g. <c>model.layers.0.mlp.down_proj</c>):
/// <list type="bullet">
///   <item><c>{prefix}.qweight</c> — <see cref="SafetensorsDtype.Int32"/>, the 4-bit (or 8-bit)
///         weight codes packed eight-per-int32.</item>
///   <item><c>{prefix}.qzeros</c> — <see cref="SafetensorsDtype.Int32"/>, the per-group zero
///         points, packed the same way along the output dimension.</item>
///   <item><c>{prefix}.scales</c> — <see cref="SafetensorsDtype.Float16"/> (or F32/BF16), one
///         scale per <c>(group, out_channel)</c>.</item>
/// </list>
/// The two schemes differ in which axis is packed and in the nibble ordering; the exact layouts
/// this class assumes are documented on <see cref="DequantizeGptq"/> and <see cref="DequantizeAwq"/>.
/// Output is always a row-major <c>[in_features, out_features]</c> float matrix (the natural
/// weight orientation for <c>y = x · W</c>); this is intentionally CPU-/float-only and is not
/// wired into the engine.
/// </summary>
public static class QuantizedWeights
{
    /// <summary>The AWQ 4-bit nibble interleave: logical output index <c>i</c> is stored at
    /// packed position <c>AwqOrder4Bit[i % 8]</c> within each int32. AWQ reverses this when packing,
    /// so reading back applies the same permutation to recover logical order.</summary>
    private static readonly int[] AwqOrder4Bit = { 0, 4, 1, 5, 2, 6, 3, 7 };

    // -------------------------------------------------------------------------------------
    // GPTQ
    // -------------------------------------------------------------------------------------

    /// <summary>
    /// Dequantizes a GPTQ-packed Linear layer to a <c>[in_features, out_features]</c> float matrix.
    /// <para><b>Assumed layout</b> (the AutoGPTQ / GPTQ-for-LLaMa convention):</para>
    /// <list type="bullet">
    ///   <item><c>qweight</c> has shape <c>[in_features / pack, out_features]</c> and is packed along
    ///         the <b>input/row</b> axis: the weight code at logical <c>(in, out)</c> is field
    ///         <c>in % pack</c> of <c>qweight[in / pack, out]</c>, where <c>pack = 32 / bits</c>.
    ///         Fields are taken from the low bits upward, in natural order (no interleave).</item>
    ///   <item><c>scales</c> has shape <c>[num_groups, out_features]</c>, one scale per group/column,
    ///         where <c>num_groups = in_features / groupSize</c> and the group of row <c>in</c> is
    ///         <c>in / groupSize</c>.</item>
    ///   <item><c>qzeros</c> has shape <c>[num_groups, out_features / pack]</c> and is packed along
    ///         the <b>output</b> axis: the zero code for <c>(group, out)</c> is field <c>out % pack</c>
    ///         of <c>qzeros[group, out / pack]</c>.</item>
    /// </list>
    /// <para>Dequantization uses <c>w(in, out) = (q − zero) · scale</c> with the raw stored zero
    /// point (no <c>+1</c> adjustment); callers whose checkpoints follow the alternate
    /// <c>zero + 1</c> convention should pre-bias their zeros accordingly.</para>
    /// </summary>
    /// <param name="file">The loaded safetensors file.</param>
    /// <param name="prefix">The Linear layer prefix, without a trailing dot.</param>
    /// <param name="bits">Bit width of each code; 4 and 8 are supported.</param>
    /// <param name="groupSize">Rows sharing one scale/zero. Must divide <c>in_features</c>.</param>
    /// <exception cref="ModelSharpException">A tensor is missing, has the wrong dtype, or the shapes
    /// are mutually inconsistent.</exception>
    public static Tensor<float> DequantizeGptq(
        SafetensorsFile file, string prefix, int bits = 4, int groupSize = 128)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(prefix);
        ValidateBits(bits);
        if (groupSize <= 0)
            throw new ModelSharpException($"GPTQ group size must be positive (got {groupSize}).");

        int pack = 32 / bits;

        Tensor<int> qweight = LoadInt32(file, prefix, "qweight");
        Tensor<int> qzeros = LoadInt32(file, prefix, "qzeros");
        Tensor<float> scales = LoadScales(file, prefix);

        Require2D(qweight, prefix, "qweight");
        Require2D(qzeros, prefix, "qzeros");
        Require2D(scales, prefix, "scales");

        int packedRows = qweight.Shape[0];
        int outFeatures = qweight.Shape[1];
        int inFeatures = checked(packedRows * pack);

        int numGroups = scales.Shape[0];
        if (scales.Shape[1] != outFeatures)
            throw new ModelSharpException(
                $"GPTQ layer '{prefix}': scales out dim {scales.Shape[1]} does not match " +
                $"qweight out dim {outFeatures}.");

        if (inFeatures % groupSize != 0)
            throw new ModelSharpException(
                $"GPTQ layer '{prefix}': in_features {inFeatures} is not divisible by group size {groupSize}.");
        if (inFeatures / groupSize != numGroups)
            throw new ModelSharpException(
                $"GPTQ layer '{prefix}': scales declare {numGroups} groups but in_features {inFeatures} " +
                $"with group size {groupSize} implies {inFeatures / groupSize}.");

        int packedOut = CeilDiv(outFeatures, pack);
        if (qzeros.Shape[0] != numGroups || qzeros.Shape[1] != packedOut)
            throw new ModelSharpException(
                $"GPTQ layer '{prefix}': qzeros shape [{qzeros.Shape[0]}, {qzeros.Shape[1]}] does not " +
                $"match expected [{numGroups}, {packedOut}].");

        int mask = (1 << bits) - 1;
        ReadOnlySpan<int> qw = qweight.Span;
        ReadOnlySpan<int> qz = qzeros.Span;
        ReadOnlySpan<float> sc = scales.Span;

        var result = new float[checked(inFeatures * outFeatures)];
        for (int inRow = 0; inRow < inFeatures; inRow++)
        {
            int group = inRow / groupSize;
            int packRow = inRow / pack;
            int field = inRow % pack;
            int shift = field * bits;
            int qwBase = packRow * outFeatures;
            int scBase = group * outFeatures;
            int qzBase = group * packedOut;
            int outBase = inRow * outFeatures;

            for (int outCol = 0; outCol < outFeatures; outCol++)
            {
                int q = (qw[qwBase + outCol] >> shift) & mask;
                int zero = (qz[qzBase + (outCol / pack)] >> ((outCol % pack) * bits)) & mask;
                float scale = sc[scBase + outCol];
                result[outBase + outCol] = (q - zero) * scale;
            }
        }

        return new Tensor<float>(new TensorShape(inFeatures, outFeatures), result);
    }

    // -------------------------------------------------------------------------------------
    // AWQ
    // -------------------------------------------------------------------------------------

    /// <summary>
    /// Dequantizes an AWQ-packed Linear layer to a <c>[in_features, out_features]</c> float matrix.
    /// <para><b>Assumed layout</b> (the AutoAWQ GEMM convention):</para>
    /// <list type="bullet">
    ///   <item><c>qweight</c> has shape <c>[in_features, out_features / pack]</c> and is packed along
    ///         the <b>output</b> axis, where <c>pack = 32 / bits</c>. The eight 4-bit codes inside one
    ///         int32 are not stored in natural order: the logical output channel
    ///         <c>out = base + j</c> (with <c>base = (out / pack) · pack</c>, <c>j ∈ [0, pack)</c>)
    ///         lives at bit field <see cref="AwqOrder4Bit"/><c>[j]</c>, i.e. the interleave
    ///         <c>[0, 4, 1, 5, 2, 6, 3, 7]</c> for 4-bit. (For 8-bit the order is the identity.)</item>
    ///   <item><c>scales</c> has shape <c>[num_groups, out_features]</c> and is <b>not</b> packed —
    ///         one float per group/column, in natural output order.</item>
    ///   <item><c>qzeros</c> has shape <c>[num_groups, out_features / pack]</c>, packed along the
    ///         output axis with the <b>same interleave</b> as <c>qweight</c>.</item>
    /// </list>
    /// <para>Group of an input row <c>in</c> is <c>in / groupSize</c>; dequantization is
    /// <c>w(in, out) = (q − zero) · scale</c>.</para>
    /// </summary>
    /// <param name="file">The loaded safetensors file.</param>
    /// <param name="prefix">The Linear layer prefix, without a trailing dot.</param>
    /// <param name="bits">Bit width of each code; 4 and 8 are supported.</param>
    /// <param name="groupSize">Rows sharing one scale/zero. Must divide <c>in_features</c>.</param>
    /// <exception cref="ModelSharpException">A tensor is missing, has the wrong dtype, or the shapes
    /// are mutually inconsistent.</exception>
    public static Tensor<float> DequantizeAwq(
        SafetensorsFile file, string prefix, int bits = 4, int groupSize = 128)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(prefix);
        ValidateBits(bits);
        if (groupSize <= 0)
            throw new ModelSharpException($"AWQ group size must be positive (got {groupSize}).");

        int pack = 32 / bits;

        Tensor<int> qweight = LoadInt32(file, prefix, "qweight");
        Tensor<int> qzeros = LoadInt32(file, prefix, "qzeros");
        Tensor<float> scales = LoadScales(file, prefix);

        Require2D(qweight, prefix, "qweight");
        Require2D(qzeros, prefix, "qzeros");
        Require2D(scales, prefix, "scales");

        int inFeatures = qweight.Shape[0];
        int packedOut = qweight.Shape[1];
        int outFeatures = checked(packedOut * pack);

        int numGroups = scales.Shape[0];
        if (scales.Shape[1] != outFeatures)
            throw new ModelSharpException(
                $"AWQ layer '{prefix}': scales out dim {scales.Shape[1]} does not match " +
                $"qweight out dim {outFeatures} (= {packedOut} × {pack}).");

        if (inFeatures % groupSize != 0)
            throw new ModelSharpException(
                $"AWQ layer '{prefix}': in_features {inFeatures} is not divisible by group size {groupSize}.");
        if (inFeatures / groupSize != numGroups)
            throw new ModelSharpException(
                $"AWQ layer '{prefix}': scales declare {numGroups} groups but in_features {inFeatures} " +
                $"with group size {groupSize} implies {inFeatures / groupSize}.");

        if (qzeros.Shape[0] != numGroups || qzeros.Shape[1] != packedOut)
            throw new ModelSharpException(
                $"AWQ layer '{prefix}': qzeros shape [{qzeros.Shape[0]}, {qzeros.Shape[1]}] does not " +
                $"match expected [{numGroups}, {packedOut}].");

        int[] order = InterleaveOrder(bits, pack);
        int mask = (1 << bits) - 1;
        ReadOnlySpan<int> qw = qweight.Span;
        ReadOnlySpan<int> qz = qzeros.Span;
        ReadOnlySpan<float> sc = scales.Span;

        var result = new float[checked(inFeatures * outFeatures)];
        for (int inRow = 0; inRow < inFeatures; inRow++)
        {
            int group = inRow / groupSize;
            int qwBase = inRow * packedOut;
            int scBase = group * outFeatures;
            int qzBase = group * packedOut;
            int outBase = inRow * outFeatures;

            for (int outCol = 0; outCol < outFeatures; outCol++)
            {
                int block = outCol / pack;
                int j = outCol % pack;
                int shift = order[j] * bits;

                int q = (qw[qwBase + block] >> shift) & mask;
                int zero = (qz[qzBase + block] >> shift) & mask;
                float scale = sc[scBase + outCol];
                result[outBase + outCol] = (q - zero) * scale;
            }
        }

        return new Tensor<float>(new TensorShape(inFeatures, outFeatures), result);
    }

    // -------------------------------------------------------------------------------------
    // Bit-unpacking helpers
    // -------------------------------------------------------------------------------------

    /// <summary>
    /// Extracts the eight 4-bit nibbles packed into a single int32, from the low bits upward.
    /// Result <c>[0]</c> is bits 0–3, <c>[1]</c> is bits 4–7, …, <c>[7]</c> is bits 28–31. This is
    /// the natural (non-interleaved) GPTQ ordering; AWQ callers must additionally apply the
    /// <c>[0, 4, 1, 5, 2, 6, 3, 7]</c> permutation.
    /// </summary>
    public static int[] UnpackNibbles(int packed)
    {
        var values = new int[8];
        for (int i = 0; i < 8; i++)
            values[i] = (packed >> (i * 4)) & 0xF;
        return values;
    }

    /// <summary>
    /// Unpacks <paramref name="count"/> consecutive <paramref name="bits"/>-wide unsigned fields
    /// from a span of packed int32 words, reading each word from its low bits upward. <c>bits</c>
    /// must divide 32 (so the fields never straddle a word boundary); supported widths are 2, 4,
    /// 8 and 16. The first <c>32 / bits</c> values come from <c>packed[0]</c>, the next from
    /// <c>packed[1]</c>, and so on.
    /// </summary>
    /// <exception cref="ModelSharpException"><paramref name="bits"/> is invalid, <paramref name="count"/>
    /// is negative, or the span is too short to supply <paramref name="count"/> values.</exception>
    public static int[] UnpackBits(ReadOnlySpan<int> packed, int bits, int count)
    {
        if (bits != 2 && bits != 4 && bits != 8 && bits != 16)
            throw new ModelSharpException(
                $"UnpackBits supports 2/4/8/16-bit fields that divide 32; got {bits}.");
        if (count < 0)
            throw new ModelSharpException($"UnpackBits count must be non-negative (got {count}).");

        int perWord = 32 / bits;
        int neededWords = CeilDiv(count, perWord);
        if (packed.Length < neededWords)
            throw new ModelSharpException(
                $"UnpackBits needs {neededWords} packed word(s) to produce {count} value(s) " +
                $"at {bits} bits, but only {packed.Length} were supplied.");

        int mask = bits == 32 ? -1 : (1 << bits) - 1;
        var values = new int[count];
        for (int i = 0; i < count; i++)
        {
            int word = packed[i / perWord];
            int shift = (i % perWord) * bits;
            values[i] = (word >> shift) & mask;
        }
        return values;
    }

    // -------------------------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------------------------

    /// <summary>The packed-field permutation for a given bit width: AWQ's interleave for 4-bit,
    /// the identity otherwise.</summary>
    private static int[] InterleaveOrder(int bits, int pack)
    {
        if (bits == 4) return AwqOrder4Bit;
        var identity = new int[pack];
        for (int i = 0; i < pack; i++) identity[i] = i;
        return identity;
    }

    private static void ValidateBits(int bits)
    {
        if (bits != 4 && bits != 8)
            throw new ModelSharpException($"Only 4-bit and 8-bit quantization are supported (got {bits}).");
    }

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;

    private static Tensor<int> LoadInt32(SafetensorsFile file, string prefix, string suffix)
    {
        string name = prefix + "." + suffix;
        Tensor t = file.GetTensorRaw(name);
        if (t is not Tensor<int> ti)
            throw new ModelSharpException(
                $"Quantized tensor '{name}' must be Int32 but was {t.Dtype}.");
        return ti;
    }

    /// <summary>Loads <c>{prefix}.scales</c> as float, accepting F16/BF16/F32 (all decoded to float
    /// by <see cref="SafetensorsFile.GetTensor"/>).</summary>
    private static Tensor<float> LoadScales(SafetensorsFile file, string prefix)
    {
        string name = prefix + ".scales";
        Tensor t = file.GetTensor(name);
        if (t is not Tensor<float> tf)
            throw new ModelSharpException(
                $"Quantized tensor '{name}' must decode to Float32 (from F16/BF16/F32) but was {t.Dtype}.");
        return tf;
    }

    private static void Require2D(Tensor t, string prefix, string suffix)
    {
        if (t.Shape.Rank != 2)
            throw new ModelSharpException(
                $"Quantized tensor '{prefix}.{suffix}' must be 2-D but has rank {t.Shape.Rank} ({t.Shape}).");
    }
}
