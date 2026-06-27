using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Signal;

/// <summary>
/// ONNX <c>DFT</c> (opset 17 and opset 20). Computes the discrete Fourier transform along
/// <c>axis</c>. The input's last dimension is the complex dimension: size 1 means a real signal,
/// size 2 means complex (real, imag). The output's last dimension is always 2.
///
/// Inputs: <c>input</c>; optional <c>dft_length</c> (input 1, scalar; truncates/zero-pads the axis);
/// optional <c>axis</c> (input 2, int scalar — opset 20). In opset 17 <c>axis</c> was an attribute
/// (default 1) instead; both forms are supported. The optional <c>dft_length</c> and <c>axis</c>
/// inputs are disambiguated purely by position (index 1 vs 2), with an empty-string input name
/// denoting an omitted optional.
/// Attributes: <c>axis</c> (default 1, opset-17 form), <c>inverse</c> (default 0), <c>onesided</c>
/// (default 0). When <c>onesided</c> is set (forward only), only floor(N/2)+1 bins are returned
/// along the axis. Forward exponent is <c>-2*pi*i*k*n/N</c>; inverse is <c>+2*pi*i*k*n/N</c> scaled
/// by 1/N. The transform uses an FFT fast path (see <see cref="DftCore"/>).
/// </summary>
public sealed class DftKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "DFT";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        if (rank < 2)
            throw new ModelSharpException("DFT: input must have rank >= 2 (last dim is the complex dim).");

        int complexDim = dims[rank - 1];
        if (complexDim != 1 && complexDim != 2)
            throw new ModelSharpException($"DFT: last dimension must be 1 (real) or 2 (complex); got {complexDim}.");

        bool inverse = Attr.Int(node, "inverse", 0) != 0;
        bool onesided = Attr.Int(node, "onesided", 0) != 0;

        // axis: opset 20 moves it to optional input index 2 (an int scalar); opset 17 has it as an
        // attribute (default 1). dft_length is input index 1. The two optional inputs are
        // distinguished purely by position; an empty-string input name means the optional is
        // omitted. If input 2 is present and non-empty, it wins over the attribute.
        int axis;
        if (node.Inputs.Count > 2 && !string.IsNullOrEmpty(node.Inputs[2]))
            axis = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[2]))[0];
        else
            axis = (int)Attr.Int(node, "axis", 1);
        if (axis < 0) axis += rank;
        // The complex (last) axis cannot be the transform axis.
        if (axis < 0 || axis >= rank - 1)
            throw new ModelSharpException($"DFT: axis {axis} out of range for rank {rank} input.");

        int signalLen = dims[axis];
        int n = signalLen;
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
            n = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[1]))[0];
        if (n <= 0) throw new ModelSharpException($"DFT: dft_length must be positive; got {n}.");

        int outBins = (onesided && !inverse) ? DftCore.OutputBins(n, true) : n;

        // Output shape: same as input but axis -> outBins and last dim -> 2.
        var outDims = dims.ToArray();
        outDims[axis] = outBins;
        outDims[rank - 1] = 2;
        var outShape = new TensorShape(outDims);
        var outBuf = new float[checked((int)outShape.Length)];

        // Row-major strides for input and output.
        var inStrides = Strides(dims);
        var outStrides = Strides(outDims);

        // Iterate over every "lane": all axes except the transform axis and the complex axis.
        // We hold fixed indices for the non-transform, non-complex axes and run a DFT along `axis`.
        Span<int> idx = new int[rank];
        var reIn = new double[n];
        var imIn = new double[n];
        var reOut = new double[outBins];
        var imOut = new double[outBins];

        // Count of independent lanes = product of all dims except axis and last.
        int lanes = 1;
        for (int d = 0; d < rank; d++)
            if (d != axis && d != rank - 1) lanes *= dims[d];

        for (int lane = 0; lane < lanes; lane++)
        {
            // Decode `lane` into idx[] for the non-transform/non-complex axes.
            int rem = lane;
            for (int d = rank - 1; d >= 0; d--)
            {
                if (d == axis || d == rank - 1) { idx[d] = 0; continue; }
                idx[d] = rem % dims[d];
                rem /= dims[d];
            }

            // Gather the frame along `axis` (zero-padded / truncated to length n).
            for (int t = 0; t < n; t++)
            {
                double re = 0.0, im = 0.0;
                if (t < signalLen)
                {
                    idx[axis] = t;
                    idx[rank - 1] = 0;
                    long off = Offset(idx, inStrides);
                    re = x.Span[(int)off];
                    if (complexDim == 2)
                    {
                        idx[rank - 1] = 1;
                        im = x.Span[(int)Offset(idx, inStrides)];
                    }
                }
                reIn[t] = re;
                imIn[t] = im;
            }

            DftCore.Dft1d(reIn, imIn, reOut, imOut, n, outBins, inverse);

            // Scatter the result.
            for (int k = 0; k < outBins; k++)
            {
                idx[axis] = k;
                idx[rank - 1] = 0;
                outBuf[(int)Offset(idx, outStrides)] = (float)reOut[k];
                idx[rank - 1] = 1;
                outBuf[(int)Offset(idx, outStrides)] = (float)imOut[k];
            }
        }

        ctx.Set(node.Outputs[0], new Tensor<float>(outShape, outBuf));
    }

    private static long[] Strides(ReadOnlySpan<int> dims)
    {
        var s = new long[dims.Length];
        long acc = 1;
        for (int d = dims.Length - 1; d >= 0; d--)
        {
            s[d] = acc;
            acc *= dims[d];
        }
        return s;
    }

    private static long Offset(ReadOnlySpan<int> idx, long[] strides)
    {
        long off = 0;
        for (int d = 0; d < idx.Length; d++) off += idx[d] * strides[d];
        return off;
    }
}
