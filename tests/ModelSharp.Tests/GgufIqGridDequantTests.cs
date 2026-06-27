using System;
using ModelSharp;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Tests for the grid-codebook "IQ" quant families (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S,
/// IQ1_M). These families reconstruct each group of weights by indexing a large, fixed lattice
/// "grid" constant table (256–2048 entries each). ModelSharp does not vendor a bit-exact,
/// independently verified copy of those tables, and a single wrong entry would silently corrupt
/// model weights; rather than emit unverified (silently wrong) output it throws. These tests pin
/// that contract: every grid-codebook family is reported unsupported and throws a descriptive
/// <see cref="ModelSharpException"/> from both <see cref="GgufDequant.Dequantize"/> entry points.
///
/// <para>
/// When a verified grid (plus the upstream attribution in NOTICE) is later vendored for a family,
/// the corresponding entry here should be moved out of <see cref="GridFamilies"/> and replaced with
/// a hand-computed synthetic-block round-trip test in the style of
/// <see cref="GgufIqDequantTests.Iq4Xs_DequantizesExactly"/>.
/// </para>
/// </summary>
public class GgufIqGridDequantTests
{
    /// <summary>
    /// The grid-codebook families together with their on-disk per-super-block byte size
    /// (<see cref="GgmlTypeInfo.TypeSize"/>). All are QK_K = 256 element super-blocks.
    /// </summary>
    public static TheoryData<GgmlType, int> GridFamilies => new()
    {
        { GgmlType.IQ2_XXS, 66 },
        { GgmlType.IQ2_XS, 74 },
        { GgmlType.IQ2_S, 82 },
        { GgmlType.IQ3_XXS, 98 },
        { GgmlType.IQ3_S, 110 },
        { GgmlType.IQ1_S, 50 },
        { GgmlType.IQ1_M, 56 },
    };

    [Theory]
    [MemberData(nameof(GridFamilies))]
    public void GridFamily_IsNotSupported(GgmlType type, int typeSize)
    {
        _ = typeSize;
        Assert.False(GgufDequant.IsSupported(type), $"{type} must remain unsupported until a verified grid is vendored");
    }

    [Theory]
    [MemberData(nameof(GridFamilies))]
    public void GridFamily_DequantizeThrowsDescriptively(GgmlType type, int typeSize)
    {
        // One full super-block worth of zero bytes — the type is rejected before any bytes are read,
        // so the buffer contents are irrelevant.
        var raw = new byte[typeSize];
        var ex = Assert.Throws<ModelSharpException>(() => GgufDequant.Dequantize(raw, type, 256));

        // The message should explain *why* (grid/lattice not vendored) rather than just naming the type,
        // and must promise not to approximate.
        Assert.Contains("grid", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not approximated", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [MemberData(nameof(GridFamilies))]
    public void GridFamily_BlockSizeIsSuperBlock(GgmlType type, int typeSize)
    {
        _ = typeSize;
        // Sanity: the sizing traits are still wired up even though dequant is gated, so a future
        // implementation drops in without touching GgmlTypeInfo.
        Assert.Equal(256, GgmlTypeInfo.BlockSize(type));
        Assert.True(GgmlTypeInfo.IsQuantized(type));
    }

    [Theory]
    [MemberData(nameof(GridFamilies))]
    public void GridFamily_TypeSizeMatchesReference(GgmlType type, int typeSize)
    {
        Assert.Equal(typeSize, GgmlTypeInfo.TypeSize(type));
    }

    [Fact]
    public void Iq4Family_StaysSupported()
    {
        // Guards against accidentally over-broadening the gate: the already-implemented codebook
        // quants must remain supported.
        Assert.True(GgufDequant.IsSupported(GgmlType.IQ4_NL));
        Assert.True(GgufDequant.IsSupported(GgmlType.IQ4_XS));
    }
}
