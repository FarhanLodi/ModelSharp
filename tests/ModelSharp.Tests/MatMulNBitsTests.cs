using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Quantize;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for <c>com.microsoft.MatMulNBits</c> — block-wise n-bit (INT4/INT8)
/// quantized matmul. Each test hand-packs the n-bit weights into the
/// <c>[N, n_blocks_per_row, blob_size]</c> uint8 layout, computes the dequantized weight
/// matrix and the expected <c>Y = A · Wᵀ</c> by hand, and asserts the kernel matches.
///
/// Packing reminder (bits=4): two weights per byte, low nibble = even index, high nibble =
/// odd index. Scales/zero-points are indexed <c>[n · n_blocks_per_row + b]</c>. Absent
/// zero_points ⇒ default symmetric zp = 2^(bits-1) (8 for 4-bit, 128 for 8-bit).
/// </summary>
public class MatMulNBitsTests
{
    private const float Tol = 1e-4f;

    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static GraphNode Node(string[] ins, string[] outs, Dictionary<string, object> attrs)
        => new("MatMulNBits", "n", ins, outs, attrs);

    private static Tensor<float> F(int[] dims, params float[] data)
        => Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<byte> U8(int[] dims, params byte[] data)
        => Tensor<byte>.FromArray(new TensorShape(dims), data);

    /// <summary>Packs two 4-bit values into one byte (low nibble = lo, high nibble = hi).</summary>
    private static byte Pack4(int lo, int hi) => (byte)((lo & 0xF) | ((hi & 0xF) << 4));

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= Tol,
                $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    private static Dictionary<string, object> Attrs(long n, long k, long bits, long blockSize)
        => new() { ["N"] = n, ["K"] = k, ["bits"] = bits, ["block_size"] = blockSize };

    // ---- 4-bit, default (symmetric) zero point --------------------------------------------------

    [Fact]
    public void Bits4_DefaultZeroPoint_SingleBlock()
    {
        // K=4, N=2, bits=4, block_size=4 → n_blocks_per_row=1, blob_size=ceil(4*4/8)=2.
        // Row 0 codes q = [8, 9, 7, 10]; Row 1 codes q = [0, 15, 8, 4].
        // Default zp = 8. scale row0 = 2.0, row1 = 0.5.
        //   W0 = (q-8)*2.0 = [ 0,  2, -2,  4]
        //   W1 = (q-8)*0.5 = [-4, 3.5, 0, -2]
        // A = [1,2,3,4]:
        //   Y0 = 1*0 + 2*2 + 3*(-2) + 4*4 = 14
        //   Y1 = 1*(-4) + 2*3.5 + 3*0 + 4*(-2) = -5
        var b = U8(new[] { 2, 1, 2 },
            Pack4(8, 9), Pack4(7, 10),    // row 0
            Pack4(0, 15), Pack4(8, 4));   // row 1
        var ctx = Ctx(
            ("A", F(new[] { 1, 4 }, 1, 2, 3, 4)),
            ("B", b),
            ("scales", F(new[] { 2 }, 2.0f, 0.5f)));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales" }, new[] { "Y" }, Attrs(2, 4, 4, 4)), ctx);

        Tensor<float> y = ctx.Get("Y");
        Assert.Equal(new[] { 1, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 14f, -5f }, y.Span);
    }

    // ---- 4-bit, multiple blocks (K = 8, two blocks of 4) ----------------------------------------

    [Fact]
    public void Bits4_DefaultZeroPoint_TwoBlocks_PerBlockScale()
    {
        // K=8, N=2, bits=4, block_size=4 → n_blocks_per_row=2, blob_size=2 (per block).
        // Row 0: block0 q=[8,10,8,8] scale=1 ; block1 q=[8,8,12,8] scale=2.
        //   W0 = [ (q-8)*scale ] = [0,2,0,0, 0,0,8,0]
        // Row 1: block0 q=[9,8,8,7] scale=0.5 ; block1 q=[8,8,8,16? no -> 15] scale=1.
        //   block1 q=[8,8,8,15] → (q-8)*1 = [0,0,0,7]
        //   W1 = [0.5,0,0,-0.5, 0,0,0,7]
        // A = [1,2,3,4, 5,6,7,8]:
        //   Y0 = 1*0+2*2+3*0+4*0 + 5*0+6*0+7*8+8*0 = 4 + 56 = 60
        //   Y1 = 1*0.5+2*0+3*0+4*(-0.5) + 5*0+6*0+7*0+8*7 = (0.5-2) + 56 = 54.5
        var b = U8(new[] { 2, 2, 2 },
            // row 0
            Pack4(8, 10), Pack4(8, 8),    // block 0
            Pack4(8, 8), Pack4(12, 8),    // block 1
            // row 1
            Pack4(9, 8), Pack4(8, 7),     // block 0
            Pack4(8, 8), Pack4(8, 15));   // block 1
        var ctx = Ctx(
            ("A", F(new[] { 1, 8 }, 1, 2, 3, 4, 5, 6, 7, 8)),
            ("B", b),
            // scales[n*nblocks + b]: row0 -> [1,2], row1 -> [0.5,1]
            ("scales", F(new[] { 4 }, 1f, 2f, 0.5f, 1f)));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales" }, new[] { "Y" }, Attrs(2, 8, 4, 4)), ctx);

        AssertClose(new[] { 60f, 54.5f }, ctx.Get("Y").Span);
    }

    // ---- 4-bit, explicit packed (n-bit) zero points ---------------------------------------------

    [Fact]
    public void Bits4_PackedZeroPoints()
    {
        // K=4, N=2, bits=4, block_size=4 → n_blocks_per_row=1.
        // Packed zp: zp_row_bytes = ceil(1*4/8) = 1 byte/row. Only the low nibble (block 0) is used.
        // Row 0 zp = 6, Row 1 zp = 10.
        // Row 0 q=[8,9,7,10] scale=2 → W0=(q-6)*2 = [4,6,2,8]
        // Row 1 q=[0,15,8,4] scale=0.5 → W1=(q-10)*0.5 = [-5,2.5,-1,-3]
        // A = [1,2,3,4]:
        //   Y0 = 1*4+2*6+3*2+4*8 = 4+12+6+32 = 54
        //   Y1 = 1*(-5)+2*2.5+3*(-1)+4*(-3) = -5+5-3-12 = -15
        var b = U8(new[] { 2, 1, 2 },
            Pack4(8, 9), Pack4(7, 10),
            Pack4(0, 15), Pack4(8, 4));
        var zp = U8(new[] { 2 }, Pack4(6, 0), Pack4(10, 0));  // 1 byte/row, low nibble = block0 zp
        var ctx = Ctx(
            ("A", F(new[] { 1, 4 }, 1, 2, 3, 4)),
            ("B", b),
            ("scales", F(new[] { 2 }, 2.0f, 0.5f)),
            ("zp", zp));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales", "zp" }, new[] { "Y" }, Attrs(2, 4, 4, 4)), ctx);

        AssertClose(new[] { 54f, -15f }, ctx.Get("Y").Span);
    }

    // ---- 4-bit, explicit float zero points ------------------------------------------------------

    [Fact]
    public void Bits4_FloatZeroPoints()
    {
        // Same weights as the packed-zp case but zero_points supplied as float [N*nblocks].
        var b = U8(new[] { 2, 1, 2 },
            Pack4(8, 9), Pack4(7, 10),
            Pack4(0, 15), Pack4(8, 4));
        var ctx = Ctx(
            ("A", F(new[] { 1, 4 }, 1, 2, 3, 4)),
            ("B", b),
            ("scales", F(new[] { 2 }, 2.0f, 0.5f)),
            ("zp", F(new[] { 2 }, 6f, 10f)));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales", "zp" }, new[] { "Y" }, Attrs(2, 4, 4, 4)), ctx);

        AssertClose(new[] { 54f, -15f }, ctx.Get("Y").Span);
    }

    // ---- 4-bit, batched A (2 rows) --------------------------------------------------------------

    [Fact]
    public void Bits4_BatchedA()
    {
        // Reuse the default-zp weights (W0=[0,2,-2,4], W1=[-4,3.5,0,-2]).
        // A row0 = [1,2,3,4] → Y=[14,-5] (as case 1).
        // A row1 = [2,0,1,0]:
        //   Y0 = 2*0+0+1*(-2)+0 = -2
        //   Y1 = 2*(-4)+0+1*0+0 = -8
        var b = U8(new[] { 2, 1, 2 },
            Pack4(8, 9), Pack4(7, 10),
            Pack4(0, 15), Pack4(8, 4));
        var ctx = Ctx(
            ("A", F(new[] { 2, 4 }, 1, 2, 3, 4, 2, 0, 1, 0)),
            ("B", b),
            ("scales", F(new[] { 2 }, 2.0f, 0.5f)));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales" }, new[] { "Y" }, Attrs(2, 4, 4, 4)), ctx);

        Tensor<float> y = ctx.Get("Y");
        Assert.Equal(new[] { 2, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 14f, -5f, -2f, -8f }, y.Span);
    }

    // ---- 3-D A (leading batch dims preserved) ---------------------------------------------------

    [Fact]
    public void Bits4_ThreeDimA_PreservesLeadingDims()
    {
        // A shape [2,1,4]; output shape [2,1,2]. Same rows as the batched case.
        var b = U8(new[] { 2, 1, 2 },
            Pack4(8, 9), Pack4(7, 10),
            Pack4(0, 15), Pack4(8, 4));
        var ctx = Ctx(
            ("A", F(new[] { 2, 1, 4 }, 1, 2, 3, 4, 2, 0, 1, 0)),
            ("B", b),
            ("scales", F(new[] { 2 }, 2.0f, 0.5f)));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales" }, new[] { "Y" }, Attrs(2, 4, 4, 4)), ctx);

        Tensor<float> y = ctx.Get("Y");
        Assert.Equal(new[] { 2, 1, 2 }, y.Shape.Dimensions.ToArray());
        AssertClose(new[] { 14f, -5f, -2f, -8f }, y.Span);
    }

    // ---- K not a multiple of block_size (tail block partially filled) ---------------------------

    [Fact]
    public void Bits4_KNotMultipleOfBlockSize()
    {
        // ONNX Runtime allows K not divisible by block_size: the last block is partially used,
        // only the first (K - (nblocks-1)*block_size) codes of its blob contribute.
        // K=6, N=1, bits=4, block_size=4 → n_blocks_per_row=ceil(6/4)=2, blob_size=2.
        // block0 covers k=0..3, block1 covers k=4..5 (only 2 of its 4 slots used).
        // Row 0: block0 q=[8,9,8,7] scale=1 → W=[0,1,0,-1]; block1 q=[10,8,(6),(6)] scale=2,
        //   only k=4,5 used → W=[(10-8)*2,(8-8)*2] = [4,0]. (codes at packed idx 2,3 ignored.)
        // W0 = [0,1,0,-1, 4,0]
        // A = [1,2,3,4,5,6]:
        //   Y = 1*0+2*1+3*0+4*(-1) + 5*4+6*0 = (2-4) + 20 = 18
        var b = U8(new[] { 1, 2, 2 },
            Pack4(8, 9), Pack4(8, 7),     // block 0 (k=0..3)
            Pack4(10, 8), Pack4(6, 6));   // block 1 (k=4..5 used; last two ignored)
        var ctx = Ctx(
            ("A", F(new[] { 1, 6 }, 1, 2, 3, 4, 5, 6)),
            ("B", b),
            ("scales", F(new[] { 2 }, 1f, 2f)));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales" }, new[] { "Y" }, Attrs(1, 6, 4, 4)), ctx);

        AssertClose(new[] { 18f }, ctx.Get("Y").Span);
    }

    // ---- 8-bit path -----------------------------------------------------------------------------

    [Fact]
    public void Bits8_DefaultZeroPoint()
    {
        // K=4, N=2, bits=8, block_size=4 → n_blocks_per_row=1, blob_size=ceil(4*8/8)=4 (1 byte/code).
        // Default zp = 128.
        // Row 0 q=[128,130,126,132] scale=2 → W0=(q-128)*2 = [0,4,-4,8]
        // Row 1 q=[120,128,140,128] scale=0.5 → W1=(q-128)*0.5 = [-4,0,6,0]
        // A=[1,2,3,4]:
        //   Y0 = 1*0+2*4+3*(-4)+4*8 = 0+8-12+32 = 28
        //   Y1 = 1*(-4)+2*0+3*6+4*0 = -4+18 = 14
        var b = U8(new[] { 2, 1, 4 },
            128, 130, 126, 132,
            120, 128, 140, 128);
        var ctx = Ctx(
            ("A", F(new[] { 1, 4 }, 1, 2, 3, 4)),
            ("B", b),
            ("scales", F(new[] { 2 }, 2.0f, 0.5f)));

        new MatMulNBitsKernel().Execute(
            Node(new[] { "A", "B", "scales" }, new[] { "Y" }, Attrs(2, 4, 8, 4)), ctx);

        AssertClose(new[] { 28f, 14f }, ctx.Get("Y").Span);
    }

    // ---- registration smoke ---------------------------------------------------------------------

    [Fact]
    public void Kernel_IsRegistered()
    {
        KernelRegistry r = KernelRegistry.CreateDefault();
        Assert.True(r.TryGet("MatMulNBits", out IKernel? k));
        Assert.IsType<MatMulNBitsKernel>(k);
    }
}
