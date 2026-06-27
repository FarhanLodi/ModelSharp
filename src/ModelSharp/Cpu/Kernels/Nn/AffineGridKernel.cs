using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>AffineGrid</c> (opset 20): generates a flow field (sampling grid) from a batch of affine
/// matrices <c>theta</c> and an output <c>size</c>, matching <c>GridSample</c>'s grid convention.
/// <para>
/// 2-D: <c>theta</c> is <c>[N,2,3]</c>, <c>size</c> is <c>[N,C,H,W]</c>, output grid is
/// <c>[N,H,W,2]</c>. 3-D: <c>theta</c> is <c>[N,3,4]</c>, <c>size</c> is <c>[N,C,D,H,W]</c>, output
/// is <c>[N,D,H,W,3]</c>. Base coordinates span a normalized <c>[-1,1]</c> grid (endpoint behavior
/// controlled by <c>align_corners</c>, default 0) and are mapped by the affine transform
/// <c>grid = base · thetaᵀ</c> with the homogeneous 1 appended. Float32.
/// </para>
/// </summary>
public sealed class AffineGridKernel : IKernel
{
    public string OpType => "AffineGrid";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> theta = ctx.Get(node.Inputs[0]);
        long[] size = TensorInts.Read(ctx.GetTensor(node.Inputs[1]));
        bool align = Attr.Int(node, "align_corners", 0) != 0;

        int N = (int)size[0];
        Span<float> th = theta.Span;

        if (size.Length == 4)
        {
            int H = (int)size[2], W = (int)size[3];
            var grid = new float[N * H * W * 2];
            for (int n = 0; n < N; n++)
            {
                int tb = n * 6; // 2x3
                for (int h = 0; h < H; h++)
                {
                    float by = Linspace(h, H, align);
                    for (int w = 0; w < W; w++)
                    {
                        float bx = Linspace(w, W, align);
                        int o = ((n * H + h) * W + w) * 2;
                        grid[o + 0] = th[tb + 0] * bx + th[tb + 1] * by + th[tb + 2];
                        grid[o + 1] = th[tb + 3] * bx + th[tb + 4] * by + th[tb + 5];
                    }
                }
            }
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(N, H, W, 2), grid));
        }
        else if (size.Length == 5)
        {
            int Dp = (int)size[2], H = (int)size[3], W = (int)size[4];
            var grid = new float[N * Dp * H * W * 3];
            for (int n = 0; n < N; n++)
            {
                int tb = n * 12; // 3x4
                for (int z = 0; z < Dp; z++)
                {
                    float bz = Linspace(z, Dp, align);
                    for (int h = 0; h < H; h++)
                    {
                        float by = Linspace(h, H, align);
                        for (int w = 0; w < W; w++)
                        {
                            float bx = Linspace(w, W, align);
                            int o = (((n * Dp + z) * H + h) * W + w) * 3;
                            grid[o + 0] = th[tb + 0] * bx + th[tb + 1] * by + th[tb + 2] * bz + th[tb + 3];
                            grid[o + 1] = th[tb + 4] * bx + th[tb + 5] * by + th[tb + 6] * bz + th[tb + 7];
                            grid[o + 2] = th[tb + 8] * bx + th[tb + 9] * by + th[tb + 10] * bz + th[tb + 11];
                        }
                    }
                }
            }
            ctx.Set(node.Outputs[0], new Tensor<float>(new TensorShape(N, Dp, H, W, 3), grid));
        }
        else
        {
            throw new ModelSharpException("AffineGrid: size must describe a 2-D ([N,C,H,W]) or 3-D ([N,C,D,H,W]) grid.");
        }
    }

    /// <summary>The normalized coordinate of index <paramref name="i"/> along an axis of length
    /// <paramref name="size"/>, spanning [-1,1]. align_corners places samples on the endpoints.</summary>
    private static float Linspace(int i, int size, bool align)
    {
        if (size <= 1) return 0f;
        return align
            ? -1f + 2f * i / (size - 1)
            : -1f + (2f * i + 1f) / size;
    }
}
