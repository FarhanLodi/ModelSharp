using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Sequence;

/// <summary>
/// ONNX <c>Optional</c>: wraps its (optional) input into an optional value. With an input present
/// the result is a present optional carrying that tensor (or sequence); with the input omitted the
/// result is an absent optional ("none") whose declared element type comes from the <c>type</c>
/// attribute (not modeled at runtime — an absent optional carries no payload).
/// </summary>
public sealed class OptionalKernel : IKernel
{
    public string OpType => "Optional";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        bool hasInput = node.Inputs.Count > 0 && node.Inputs[0].Length > 0;
        if (!hasInput)
        {
            ctx.SetSeq(node.Outputs[0], SeqValue.None);
            return;
        }

        string inName = node.Inputs[0];
        if (ctx.HasSeq(inName))
        {
            // Wrapping a sequence (or re-wrapping an existing optional's payload).
            SeqValue inner = ctx.GetSeq(inName);
            ctx.SetSeq(node.Outputs[0], inner.IsOptional
                ? inner                                  // already optional: pass through
                : SeqValue.SomeSequence(inner));         // a plain sequence: wrap it
        }
        else
        {
            ctx.SetSeq(node.Outputs[0], SeqValue.SomeTensor(ctx.GetTensor(inName)));
        }
    }
}

/// <summary>
/// ONNX <c>OptionalGetElement</c>: unwraps an optional, yielding the contained tensor or sequence.
/// Throws if the optional is absent. Also accepts a plain tensor/sequence input (identity) per the
/// opset-18 relaxation where the input need not be an optional.
/// </summary>
public sealed class OptionalGetElementKernel : IKernel
{
    public string OpType => "OptionalGetElement";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        string inName = node.Inputs[0];

        if (ctx.HasSeq(inName))
        {
            SeqValue v = ctx.GetSeq(inName);
            switch (v.Kind)
            {
                case SeqValue.ValueKind.OptionalNone:
                    throw new ModelSharpException(
                        $"OptionalGetElement node '{node.Name}': optional is empty (none).");
                case SeqValue.ValueKind.OptionalTensor:
                    ctx.Set(node.Outputs[0], v.Tensors[0]);
                    break;
                case SeqValue.ValueKind.OptionalSequence:
                    ctx.SetSeq(node.Outputs[0], SeqValue.Sequence(v.Tensors));
                    break;
                default: // a plain sequence: identity passthrough
                    ctx.SetSeq(node.Outputs[0], v);
                    break;
            }
            return;
        }

        // Plain tensor input: identity.
        ctx.Set(node.Outputs[0], ctx.GetTensor(inName));
    }
}

/// <summary>
/// ONNX <c>OptionalHasElement</c>: returns a Boolean scalar — true when the optional is present
/// (or, per opset-18, when a plain tensor/sequence is supplied), false when it is absent or the
/// input is omitted entirely.
/// </summary>
public sealed class OptionalHasElementKernel : IKernel
{
    public string OpType => "OptionalHasElement";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        bool present;
        if (node.Inputs.Count == 0 || node.Inputs[0].Length == 0)
        {
            present = false;   // omitted input -> false (opset 18)
        }
        else
        {
            string inName = node.Inputs[0];
            if (ctx.HasSeq(inName))
            {
                SeqValue v = ctx.GetSeq(inName);
                // Optional-none -> false; a present optional or a plain sequence -> true.
                present = v.Kind != SeqValue.ValueKind.OptionalNone;
            }
            else
            {
                present = ctx.Has(inName);   // a plain tensor is "present".
            }
        }

        ctx.Set(node.Outputs[0], new Tensor<bool>(new TensorShape(), new[] { present }));
    }
}
