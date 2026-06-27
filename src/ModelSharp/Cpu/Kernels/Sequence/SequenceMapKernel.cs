using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.ControlFlow;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Sequence;

/// <summary>
/// ONNX <c>SequenceMap</c>: applies the <c>body</c> subgraph to each element of the input sequence
/// (input 0), optionally threading additional per-iteration inputs that may be sequences (sliced
/// elementwise, same length) or plain tensors (broadcast to every iteration). The body's first
/// formal input receives the current sequence element; the remaining formals receive the extra
/// inputs. Each body output is accumulated across iterations into one output sequence (so output
/// <c>k</c> is the sequence of the body's k-th output over all elements). Subgraph execution goes
/// through the <see cref="GraphContext"/> runner, matching <c>Loop</c>/<c>Scan</c>.
/// </summary>
public sealed class SequenceMapKernel : IKernel
{
    public string OpType => "SequenceMap";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        ModelGraph body = CfAttr.Graph(node, "body");
        SeqValue input = ctx.GetSeq(node.Inputs[0]);
        int n = input.Count;

        // Additional inputs (node inputs [1..]): either sequences (per-element) or tensors (broadcast).
        int extra = node.Inputs.Count - 1;
        var extraSeqs = new IReadOnlyList<Tensor>?[extra];
        var extraTensors = new Tensor?[extra];
        for (int e = 0; e < extra; e++)
        {
            string name = node.Inputs[1 + e];
            if (ctx.HasSeq(name)) extraSeqs[e] = ctx.GetSeq(name).Tensors;
            else extraTensors[e] = ctx.GetTensor(name);
        }

        int outCount = body.Outputs.Count;
        var accum = new List<Tensor>[outCount];
        for (int k = 0; k < outCount; k++) accum[k] = new List<Tensor>(n);

        for (int i = 0; i < n; i++)
        {
            var feeds = new Dictionary<string, Tensor>
            {
                [body.Inputs[0]] = input.Tensors[i],
            };
            for (int e = 0; e < extra; e++)
            {
                Tensor feed = extraSeqs[e] is not null ? extraSeqs[e]![i] : extraTensors[e]!;
                feeds[body.Inputs[1 + e]] = feed;
            }

            IReadOnlyDictionary<string, Tensor> outs = ctx.RunSubgraph(body, feeds);
            for (int k = 0; k < outCount; k++)
                accum[k].Add(outs[body.Outputs[k]]);
        }

        for (int k = 0; k < outCount && k < node.Outputs.Count; k++)
            ctx.SetSeq(node.Outputs[k], SeqValue.Sequence(accum[k]));
    }
}
