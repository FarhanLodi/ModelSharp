using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// ONNX <c>Dropout</c> (inference semantics). With <c>training_mode</c> absent or false the op is
/// the identity: the data passes through unchanged and the optional boolean <c>mask</c> output is
/// all-true. This matches how exported inference graphs use Dropout (ratio is ignored at eval time).
/// Dtype-preserving for the data path; the mask is always Boolean.
/// </summary>
public sealed class DropoutKernel : IKernel
{
    public string OpType => "Dropout";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);

        bool training = node.Inputs.Count > 2
            && !string.IsNullOrEmpty(node.Inputs[2])
            && TensorInts.Read(ctx.GetTensor(node.Inputs[2]))[0] != 0;
        if (training)
            throw new ModelSharpException("Dropout: training_mode=true is not supported (inference only).");

        // Output 0: pass the data through unchanged (preserve dtype/shape).
        ctx.Set(node.Outputs[0], data);

        // Optional output 1: mask (all true at inference time).
        if (node.Outputs.Count > 1 && !string.IsNullOrEmpty(node.Outputs[1]))
        {
            var mask = new Tensor<bool>(data.Shape);
            Span<bool> m = mask.Span;
            for (int i = 0; i < m.Length; i++) m[i] = true;
            ctx.Set(node.Outputs[1], mask);
        }
    }
}
