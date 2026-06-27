using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Signal;

/// <summary>
/// ONNX <c>STFT</c> (opset 17). Slides a window of <c>frame_length</c> over the signal with hop
/// <c>frame_step</c>, optionally multiplies each frame by <c>window</c>, and DFTs every frame.
///
/// Inputs: <c>signal</c> (<c>[batch, length]</c> or <c>[batch, length, 1|2]</c>), <c>frame_step</c>
/// (scalar), optional <c>window</c> (1-D, length = frame_length), optional <c>frame_length</c>
/// (scalar). If neither <c>window</c> nor <c>frame_length</c> is given the frame length defaults to
/// the signal length. Attribute: <c>onesided</c> (default 1). Output shape is
/// <c>[batch, n_frames, n_dft, 2]</c> where <c>n_dft = floor(frame_length/2)+1</c> when onesided,
/// else <c>frame_length</c>.
/// </summary>
public sealed class StftKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "STFT";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> signal = ctx.Get(node.Inputs[0]);
        ReadOnlySpan<int> sdims = signal.Shape.Dimensions;
        if (sdims.Length < 2)
            throw new ModelSharpException("STFT: signal must be [batch, length] or [batch, length, 1|2].");

        int batch = sdims[0];
        int signalLen = sdims[1];
        int complexDim = sdims.Length >= 3 ? sdims[2] : 1;
        if (complexDim != 1 && complexDim != 2)
            throw new ModelSharpException($"STFT: signal complex dim must be 1 or 2; got {complexDim}.");

        int frameStep = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[1]))[0];
        if (frameStep <= 0) throw new ModelSharpException($"STFT: frame_step must be positive; got {frameStep}.");

        bool hasWindow = node.Inputs.Count > 2 && !string.IsNullOrEmpty(node.Inputs[2]);
        Tensor<float>? window = hasWindow ? ctx.Get(node.Inputs[2]) : null;

        int frameLength = -1;
        if (node.Inputs.Count > 3 && !string.IsNullOrEmpty(node.Inputs[3]))
            frameLength = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[3]))[0];
        if (frameLength < 0)
            frameLength = window is not null ? window.Shape[0] : signalLen;
        if (frameLength <= 0) throw new ModelSharpException($"STFT: frame_length must be positive; got {frameLength}.");

        if (window is not null && window.Shape[0] != frameLength)
            throw new ModelSharpException(
                $"STFT: window length {window.Shape[0]} does not match frame_length {frameLength}.");

        bool onesided = Attr.Int(node, "onesided", 1) != 0;
        int nDft = onesided ? DftCore.OutputBins(frameLength, true) : frameLength;

        // Number of frames that fully fit.
        int nFrames = signalLen >= frameLength
            ? (signalLen - frameLength) / frameStep + 1
            : 0;

        var outShape = new TensorShape(batch, nFrames, nDft, 2);
        var outBuf = new float[checked((int)outShape.Length)];

        // Strides over the input signal buffer.
        int sStrideBatch = signalLen * complexDim;
        int sStrideTime = complexDim;

        var reIn = new double[frameLength];
        var imIn = new double[frameLength];
        var reOut = new double[nDft];
        var imOut = new double[nDft];

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < nFrames; f++)
            {
                int start = f * frameStep;
                for (int t = 0; t < frameLength; t++)
                {
                    int baseOff = b * sStrideBatch + (start + t) * sStrideTime;
                    double re = signal.Span[baseOff];
                    double im = complexDim == 2 ? signal.Span[baseOff + 1] : 0.0;
                    if (window is not null)
                    {
                        double w = window.Span[t];
                        re *= w;
                        im *= w;
                    }
                    reIn[t] = re;
                    imIn[t] = im;
                }

                DftCore.Dft1d(reIn, imIn, reOut, imOut, frameLength, nDft, inverse: false);

                int outBase = ((b * nFrames + f) * nDft) * 2;
                for (int k = 0; k < nDft; k++)
                {
                    outBuf[outBase + k * 2] = (float)reOut[k];
                    outBuf[outBase + k * 2 + 1] = (float)imOut[k];
                }
            }
        }

        ctx.Set(node.Outputs[0], new Tensor<float>(outShape, outBuf));
    }
}
