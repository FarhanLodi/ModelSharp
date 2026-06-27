using System;
using ModelSharp;
using ModelSharp.Weights;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Tests for the grid-codebook "IQ" quant families (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S,
/// IQ1_M). These families reconstruct each group of weights by indexing a large, fixed lattice
/// "grid" constant table (256–2048 entries each). ModelSharp now vendors those tables verbatim from
/// llama.cpp's <c>ggml-common.h</c> (MIT, pinned commit — see <see cref="GgufIqGrids"/> and NOTICE)
/// and ports each <c>dequantize_row_iq*</c> routine, so every family dequantizes.
///
/// <para>
/// This file pins the structural contract (support flags + sizing). The bit-exact end-to-end
/// reconstruction (synthetic blocks with known grid indices/signs/scales compared against values
/// computed by hand from the same vendored grid) lives in <see cref="GgufIq123DequantTests"/>.
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
    public void GridFamily_IsSupported(GgmlType type, int typeSize)
    {
        _ = typeSize;
        Assert.True(GgufDequant.IsSupported(type), $"{type} is backed by a vendored grid and must dequantize");
    }

    [Theory]
    [MemberData(nameof(GridFamilies))]
    public void GridFamily_DequantizesToSuperBlock(GgmlType type, int typeSize)
    {
        // One full super-block worth of zero bytes is a valid encoding (grid index 0, scale 0, etc.)
        // and must reconstruct exactly QK_K = 256 finite floats.
        var raw = new byte[typeSize];
        float[] got = GgufDequant.Dequantize(raw, type, 256);
        Assert.Equal(256, got.Length);
        foreach (float f in got)
            Assert.True(float.IsFinite(f), $"{type}: produced non-finite value");
    }

    [Theory]
    [MemberData(nameof(GridFamilies))]
    public void GridFamily_BlockSizeIsSuperBlock(GgmlType type, int typeSize)
    {
        _ = typeSize;
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
        // Guards against accidentally narrowing the gate: the codebook quants remain supported.
        Assert.True(GgufDequant.IsSupported(GgmlType.IQ4_NL));
        Assert.True(GgufDequant.IsSupported(GgmlType.IQ4_XS));
    }

    [Fact]
    public void VendoredGrids_HaveExpectedSizes()
    {
        // Sanity: the vendored tables have the exact element counts of the upstream GGML_TABLE_BEGIN
        // declarations (256/512/1024 for IQ2, 256/512 for IQ3, 2048 for IQ1, 8/128 for the masks).
        Assert.Equal(256, GgufIqGrids.Iq2xxsGrid.Length);
        Assert.Equal(512, GgufIqGrids.Iq2xsGrid.Length);
        Assert.Equal(1024, GgufIqGrids.Iq2sGrid.Length);
        Assert.Equal(256, GgufIqGrids.Iq3xxsGrid.Length);
        Assert.Equal(512, GgufIqGrids.Iq3sGrid.Length);
        Assert.Equal(2048, GgufIqGrids.Iq1sGrid.Length);
        Assert.Equal(8, GgufIqGrids.KmaskIq2xs.Length);
        Assert.Equal(128, GgufIqGrids.KsignsIq2xs.Length);
    }
}
