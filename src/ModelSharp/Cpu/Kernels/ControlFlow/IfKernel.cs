using System.Collections.Generic;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.ControlFlow;

/// <summary>
/// ONNX <c>If</c>: selects between two subgraphs by a boolean scalar condition.
/// The chosen branch (<c>then_branch</c> when <c>cond</c> is true, else <c>else_branch</c>)
/// has NO formal inputs — it captures everything it needs from the outer scope — and
/// produces exactly as many outputs as the <c>If</c> node has, mapped positionally.
/// </summary>
public sealed class IfKernel : IKernel
{
    /// <inheritdoc />
    public string OpType => "If";

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor condT = ctx.GetTensor(node.Inputs[0]);
        bool cond = ReadBoolScalar(condT, node);

        ModelGraph branch = cond
            ? CfAttr.Graph(node, "then_branch")
            : CfAttr.Graph(node, "else_branch");

        // Branches take no inputs; outer scope is captured by the runner.
        IReadOnlyDictionary<string, Tensor> outs =
            ctx.RunSubgraph(branch, new Dictionary<string, Tensor>());

        // Map branch outputs positionally to the If node's outputs.
        for (int i = 0; i < node.Outputs.Count && i < branch.Outputs.Count; i++)
            ctx.Set(node.Outputs[i], outs[branch.Outputs[i]]);
    }

    /// <summary>Reads a (possibly 0-D / 1-element) boolean condition scalar.</summary>
    internal static bool ReadBoolScalar(Tensor t, GraphNode node)
    {
        if (t.Dtype != ElementType.Boolean)
            throw new ModelSharpException(
                $"{node.OpType} node '{node.Name}': condition must be Boolean, got {t.Dtype}.");
        var span = t.AsBool().Span;
        if (span.Length == 0)
            throw new ModelSharpException($"{node.OpType} node '{node.Name}': empty condition tensor.");
        return span[0];
    }
}
