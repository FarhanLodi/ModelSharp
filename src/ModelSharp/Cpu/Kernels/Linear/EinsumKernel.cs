using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Linear;

/// <summary>
/// ONNX <c>Einsum</c> over Float32 tensors. Supports the common cases used by exported models:
/// one or two operands, an equation with comma-separated subscripts and an optional explicit
/// <c>-&gt;</c> output spec, repeated indices (diagonal/trace), free indices, and summed (contracted)
/// indices. Implicit output mode follows NumPy: indices appearing exactly once, in alphabetical
/// order. Ellipsis (<c>...</c>) is not supported.
/// </summary>
public sealed class EinsumKernel : IKernel
{
    public string OpType => "Einsum";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        string equation = Attr.Str(node, "equation", string.Empty).Replace(" ", string.Empty);
        if (equation.Length == 0) throw new ModelSharpException("Einsum: missing 'equation' attribute.");
        if (equation.Contains("...")) throw new ModelSharpException("Einsum: ellipsis is not supported.");

        string lhs, rhs;
        bool explicitOut = equation.Contains("->");
        if (explicitOut)
        {
            string[] parts = equation.Split("->");
            lhs = parts[0];
            rhs = parts.Length > 1 ? parts[1] : string.Empty;
        }
        else { lhs = equation; rhs = string.Empty; }

        string[] terms = lhs.Split(',');
        if (terms.Length != node.Inputs.Count)
            throw new ModelSharpException(
                $"Einsum: equation has {terms.Length} operands but {node.Inputs.Count} inputs were provided.");
        if (terms.Length is < 1 or > 2)
            throw new ModelSharpException("Einsum: only one or two operands are supported.");

        var operands = new Tensor<float>[terms.Length];
        for (int i = 0; i < terms.Length; i++) operands[i] = ctx.Get(node.Inputs[i]);

        // Map each label to its dimension size (consistency-checked across operands).
        var dimOf = new Dictionary<char, int>();
        for (int t = 0; t < terms.Length; t++)
        {
            string sub = terms[t];
            ReadOnlySpan<int> shape = operands[t].Shape.Dimensions;
            if (sub.Length != shape.Length)
                throw new ModelSharpException(
                    $"Einsum: subscript '{sub}' rank {sub.Length} != operand rank {shape.Length}.");
            for (int a = 0; a < sub.Length; a++)
            {
                char c = sub[a];
                if (dimOf.TryGetValue(c, out int existing))
                {
                    if (existing != shape[a])
                        throw new ModelSharpException($"Einsum: inconsistent size for index '{c}'.");
                }
                else dimOf[c] = shape[a];
            }
        }

        // Determine output labels.
        string outLabels;
        if (explicitOut) outLabels = rhs;
        else
        {
            var counts = new Dictionary<char, int>();
            foreach (string sub in terms)
                foreach (char c in sub)
                    counts[c] = counts.GetValueOrDefault(c) + 1;
            outLabels = new string(counts.Where(kv => kv.Value == 1).Select(kv => kv.Key)
                .OrderBy(c => c).ToArray());
        }

        // All distinct labels: output labels first, then the summed ones.
        var allLabels = new List<char>(outLabels);
        foreach (char c in dimOf.Keys)
            if (!allLabels.Contains(c)) allLabels.Add(c);

        // Build output.
        var outDims = new int[outLabels.Length];
        for (int i = 0; i < outLabels.Length; i++) outDims[i] = dimOf[outLabels[i]];
        var outShape = new TensorShape(outDims);
        var y = new Tensor<float>(outShape);

        // Strides for each operand keyed by label position.
        var strides = new int[terms.Length][];
        for (int t = 0; t < terms.Length; t++) strides[t] = Nd.Strides(operands[t].Shape.Dimensions);

        // Iterate over the full cartesian product of all labels, accumulating into output.
        int totalLabels = allLabels.Count;
        var sizes = new int[totalLabels];
        var labelIndex = new Dictionary<char, int>();
        for (int i = 0; i < totalLabels; i++) { sizes[i] = dimOf[allLabels[i]]; labelIndex[allLabels[i]] = i; }

        // Precompute per-operand and output stride contribution per global label slot.
        var opLabelStride = new int[terms.Length][];
        for (int t = 0; t < terms.Length; t++)
        {
            opLabelStride[t] = new int[totalLabels];
            for (int a = 0; a < terms[t].Length; a++)
                opLabelStride[t][labelIndex[terms[t][a]]] += strides[t][a]; // += handles repeated labels (diagonal)
        }
        int[] outStridesArr = Nd.Strides(outDims);
        var outLabelStride = new int[totalLabels];
        for (int a = 0; a < outLabels.Length; a++) outLabelStride[labelIndex[outLabels[a]]] += outStridesArr[a];

        Span<float> ys = y.Span;
        var spans = new float[terms.Length][];
        for (int t = 0; t < terms.Length; t++) spans[t] = operands[t].Span.ToArray();

        long total = 1;
        foreach (int s in sizes) total *= s;

        var coord = new int[totalLabels];
        var opOff = new int[terms.Length];
        int outOff = 0;
        for (long idx = 0; idx < total; idx++)
        {
            float prod = 1f;
            for (int t = 0; t < terms.Length; t++) prod *= spans[t][opOff[t]];
            ys[outOff] += prod;

            for (int ax = totalLabels - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                for (int t = 0; t < terms.Length; t++) opOff[t] += opLabelStride[t][ax];
                outOff += outLabelStride[ax];
                if (coord[ax] < sizes[ax]) break;
                coord[ax] = 0;
                for (int t = 0; t < terms.Length; t++) opOff[t] -= opLabelStride[t][ax] * sizes[ax];
                outOff -= outLabelStride[ax] * sizes[ax];
            }
        }

        ctx.Set(node.Outputs[0], y);
    }
}
