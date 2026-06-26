using System;
using System.Collections.Generic;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Unique</c> in flattened mode (no <c>axis</c> attribute). Returns the unique elements of
/// the input as a 1-D tensor <c>Y</c>, plus the three optional int64 outputs: <c>indices</c> (first
/// occurrence position of each unique value in the flattened input), <c>inverse_indices</c> (for
/// each input element, the index of its value within <c>Y</c>), and <c>counts</c>. The <c>sorted</c>
/// attribute (default 1) sorts <c>Y</c> ascending; otherwise values appear in first-seen order.
/// Dtype-preserving for <c>Y</c> (Float32 / Int64 / Int32).
/// </summary>
public sealed class UniqueKernel : IKernel
{
    public string OpType => "Unique";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        if (node.Attributes.ContainsKey("axis"))
            throw new ModelSharpException("Unique: axis mode is not supported (flattened mode only).");
        bool sorted = Attr.Int(node, "sorted", 1) != 0;

        Tensor input = ctx.GetTensor(node.Inputs[0]);
        int n = checked((int)input.Length);

        // Read flattened values as double for grouping (preserves exact integer/float identity).
        var vals = ReadDoubles(input);

        // First-seen unique values with their first index, in encounter order.
        var firstIndex = new Dictionary<double, int>();
        var order = new List<double>();
        for (int i = 0; i < n; i++)
        {
            if (!firstIndex.ContainsKey(vals[i])) { firstIndex[vals[i]] = i; order.Add(vals[i]); }
        }

        // Final ordering of unique values.
        var uniques = new List<double>(order);
        if (sorted) uniques.Sort();

        // Map value -> position in Y.
        var posInY = new Dictionary<double, int>();
        for (int u = 0; u < uniques.Count; u++) posInY[uniques[u]] = u;

        int k = uniques.Count;
        var indices = new long[k];
        var counts = new long[k];
        for (int u = 0; u < k; u++) { indices[u] = firstIndex[uniques[u]]; counts[u] = 0; }
        var inverse = new long[n];
        for (int i = 0; i < n; i++)
        {
            int p = posInY[vals[i]];
            inverse[i] = p;
            counts[p]++;
        }

        ctx.Set(node.Outputs[0], BuildY(input.Dtype, uniques));
        if (node.Outputs.Count > 1 && !string.IsNullOrEmpty(node.Outputs[1]))
            ctx.Set(node.Outputs[1], Tensor<long>.FromArray(new TensorShape(k), indices));
        if (node.Outputs.Count > 2 && !string.IsNullOrEmpty(node.Outputs[2]))
            ctx.Set(node.Outputs[2], Tensor<long>.FromArray(new TensorShape(n), inverse));
        if (node.Outputs.Count > 3 && !string.IsNullOrEmpty(node.Outputs[3]))
            ctx.Set(node.Outputs[3], Tensor<long>.FromArray(new TensorShape(k), counts));
    }

    private static double[] ReadDoubles(Tensor t)
    {
        int n = checked((int)t.Length);
        var r = new double[n];
        switch (t.Dtype)
        {
            case ElementType.Float32: { var s = t.AsFloat().Span; for (int i = 0; i < n; i++) r[i] = s[i]; break; }
            case ElementType.Int64: { var s = t.AsInt64().Span; for (int i = 0; i < n; i++) r[i] = s[i]; break; }
            case ElementType.Int32: { var s = t.AsInt32().Span; for (int i = 0; i < n; i++) r[i] = s[i]; break; }
            default: throw new ModelSharpException($"Unique: unsupported dtype {t.Dtype}.");
        }
        return r;
    }

    private static Tensor BuildY(ElementType dt, List<double> u)
    {
        var shape = new TensorShape(u.Count);
        switch (dt)
        {
            case ElementType.Float32: { var b = new float[u.Count]; for (int i = 0; i < u.Count; i++) b[i] = (float)u[i]; return new Tensor<float>(shape, b); }
            case ElementType.Int64: { var b = new long[u.Count]; for (int i = 0; i < u.Count; i++) b[i] = (long)u[i]; return new Tensor<long>(shape, b); }
            case ElementType.Int32: { var b = new int[u.Count]; for (int i = 0; i < u.Count; i++) b[i] = (int)u[i]; return new Tensor<int>(shape, b); }
            default: throw new ModelSharpException($"Unique: unsupported dtype {dt}.");
        }
    }
}
