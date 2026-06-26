using System;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Linear;

/// <summary>
/// ONNX <c>Det</c>: determinant of square matrices stacked in the last two axes. Input shape
/// <c>[*, M, M]</c> produces output shape <c>[*]</c> (the batch dims; a scalar for a single
/// matrix). Computed by Gaussian elimination with partial pivoting (Float32).
/// </summary>
public sealed class DetKernel : IKernel
{
    public string OpType => "Det";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        if (rank < 2 || dims[rank - 1] != dims[rank - 2])
            throw new ModelSharpException($"Det: input must be [*,M,M]; got {x.Shape}.");

        int m = dims[rank - 1];
        int matSize = m * m;
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= dims[i];

        var outDims = new int[rank - 2];
        for (int i = 0; i < rank - 2; i++) outDims[i] = dims[i];
        var y = new Tensor<float>(new TensorShape(outDims));

        Span<float> xs = x.Span, ys = y.Span;
        var a = new double[matSize];
        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            int baseOff = bIdx * matSize;
            for (int i = 0; i < matSize; i++) a[i] = xs[baseOff + i];
            ys[bIdx] = (float)Determinant(a, m);
        }
        ctx.Set(node.Outputs[0], y);
    }

    private static double Determinant(double[] a, int n)
    {
        double det = 1.0;
        for (int col = 0; col < n; col++)
        {
            // Partial pivot: find the largest-magnitude entry in this column.
            int pivot = col;
            double best = Math.Abs(a[col * n + col]);
            for (int r = col + 1; r < n; r++)
            {
                double v = Math.Abs(a[r * n + col]);
                if (v > best) { best = v; pivot = r; }
            }
            if (best == 0.0) return 0.0;

            if (pivot != col)
            {
                for (int k = 0; k < n; k++)
                    (a[col * n + k], a[pivot * n + k]) = (a[pivot * n + k], a[col * n + k]);
                det = -det;
            }

            double diag = a[col * n + col];
            det *= diag;
            for (int r = col + 1; r < n; r++)
            {
                double factor = a[r * n + col] / diag;
                if (factor == 0.0) continue;
                for (int k = col; k < n; k++) a[r * n + k] -= factor * a[col * n + k];
            }
        }
        return det;
    }
}
