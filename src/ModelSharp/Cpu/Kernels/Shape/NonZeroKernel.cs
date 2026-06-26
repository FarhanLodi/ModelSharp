using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>NonZero</c>: returns the row-major coordinates of the input's nonzero
/// elements as an Int64 tensor with ONNX layout <c>[rank, nnz]</c> — i.e. one row per
/// axis, each column the multi-index of one nonzero element, in row-major scan order.
/// Works for Float32, Int64, Int32 and Boolean inputs.
/// </summary>
public sealed class NonZeroKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "NonZero";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        ReadOnlySpan<int> dims = input.Shape.Dimensions;
        int rank = dims.Length;
        int n = (int)input.Shape.Length;

        // Collect flat indices of nonzero elements in row-major order.
        var nz = new List<int>();
        switch (input.Dtype)
        {
            case ElementType.Float32:
            {
                Span<float> s = input.AsFloat().Span;
                for (int i = 0; i < n; i++) if (s[i] != 0f) nz.Add(i);
                break;
            }
            case ElementType.Int64:
            {
                Span<long> s = input.AsInt64().Span;
                for (int i = 0; i < n; i++) if (s[i] != 0L) nz.Add(i);
                break;
            }
            case ElementType.Int32:
            {
                Span<int> s = input.AsInt32().Span;
                for (int i = 0; i < n; i++) if (s[i] != 0) nz.Add(i);
                break;
            }
            case ElementType.Boolean:
            {
                Span<bool> s = input.AsBool().Span;
                for (int i = 0; i < n; i++) if (s[i]) nz.Add(i);
                break;
            }
            default:
                throw new ModelSharpException($"NonZero does not support dtype {input.Dtype}.");
        }

        int nnz = nz.Count;
        // A scalar input (rank 0) yields a [0, nnz] tensor per ONNX (no coordinates).
        int[] strides = Nd.Strides(dims);
        var buf = new long[rank * nnz];
        for (int col = 0; col < nnz; col++)
        {
            int flat = nz[col];
            for (int ax = 0; ax < rank; ax++)
            {
                int coord = (flat / strides[ax]) % dims[ax];
                buf[ax * nnz + col] = coord;
            }
        }

        ctx.Set(node.Outputs[0], new Tensor<long>(new TensorShape(rank, nnz), buf));
    }
}
