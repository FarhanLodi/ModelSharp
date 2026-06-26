using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Shape;

/// <summary>
/// ONNX <c>Slice</c> (opset 10+): slices <c>data</c> along the axes given by the
/// <c>starts</c>/<c>ends</c>/optional <c>axes</c>/optional <c>steps</c> int64 inputs,
/// with NumPy semantics (negative indices, clamping, negative steps). The input dtype
/// is preserved (Float32/Int64/Int32/Boolean). Opset-1 attribute form is also accepted.
/// </summary>
public sealed class SliceKernel : IKernel
{
    public string OpType => "Slice";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor data = ctx.GetTensor(node.Inputs[0]);
        ReadOnlySpan<int> inDims = data.Shape.Dimensions;
        int rank = inDims.Length;
        int[] inStrides = Nd.Strides(inDims);

        long[] starts = ReadAxisInts(node, ctx, 1, "starts");
        long[] ends = ReadAxisInts(node, ctx, 2, "ends");
        long[] axes = ReadAxisInts(node, ctx, 3, "axes");
        long[] steps = ReadAxisInts(node, ctx, 4, "steps");

        int k = starts.Length;
        if (ends.Length != k)
            throw new ModelSharpException(
                $"Slice 'starts' ({k}) and 'ends' ({ends.Length}) must have the same length.");
        if (axes.Length == 0)
        {
            axes = new long[k];
            for (int i = 0; i < k; i++) axes[i] = i;
        }

        // Identity selection for every axis; overridden below for the sliced axes.
        var effStart = new int[rank];
        var effStep = new int[rank];
        var outDims = inDims.ToArray();
        for (int i = 0; i < rank; i++) effStep[i] = 1;

        for (int i = 0; i < k; i++)
        {
            int axis = (int)axes[i];
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank)
                throw new ModelSharpException($"Slice axis {axes[i]} is out of range for rank {rank}.");

            long step = steps.Length > i ? steps[i] : 1;
            if (step == 0) throw new ModelSharpException("Slice 'steps' values cannot be 0.");

            NormalizeAxis(starts[i], ends[i], step, inDims[axis], out int s, out int count);
            effStart[axis] = s;
            effStep[axis] = (int)step;
            outDims[axis] = count;
        }

        Tensor y = data.Dtype switch
        {
            ElementType.Float32 => Gather(data.AsFloat(), outDims, inStrides, effStart, effStep),
            ElementType.Int64 => Gather(data.AsInt64(), outDims, inStrides, effStart, effStep),
            ElementType.Int32 => Gather(data.AsInt32(), outDims, inStrides, effStart, effStep),
            ElementType.Boolean => Gather(data.AsBool(), outDims, inStrides, effStart, effStep),
            _ => throw new ModelSharpException($"Slice does not support dtype {data.Dtype}."),
        };
        ctx.Set(node.Outputs[0], y);
    }

    /// <summary>Copies the selected strided sub-region into a fresh contiguous tensor of element type T.</summary>
    private static Tensor<T> Gather<T>(Tensor<T> x, int[] outDims, int[] inStrides, int[] effStart, int[] effStep)
        where T : unmanaged
    {
        int rank = outDims.Length;
        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> xs = x.Span, ys = y.Span;
        int n = (int)y.Shape.Length;
        var coord = new int[rank];

        for (int idx = 0; idx < n; idx++)
        {
            int src = 0;
            for (int i = 0; i < rank; i++) src += (effStart[i] + coord[i] * effStep[i]) * inStrides[i];
            ys[idx] = xs[src];
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }
        return y;
    }

    /// <summary>
    /// NumPy/ONNX slice normalization for one axis: resolves negative indices, clamps to the
    /// valid range for the step's sign, and yields the first source index plus the element count.
    /// </summary>
    private static void NormalizeAxis(long start, long end, long step, int dim, out int outStart, out int outCount)
    {
        long lower, upper;
        if (step < 0) { lower = -1; upper = dim - 1; }
        else { lower = 0; upper = dim; }

        long s = start;
        if (s < 0) { s += dim; if (s < lower) s = lower; }
        else if (s > upper) s = upper;

        long e = end;
        if (e < 0) { e += dim; if (e < lower) e = lower; }
        else if (e > upper) e = upper;

        long count;
        if (step > 0) count = e > s ? (e - s + step - 1) / step : 0;
        else count = s > e ? (s - e + (-step) - 1) / (-step) : 0;

        outStart = (int)s;
        outCount = (int)count;
    }

    /// <summary>
    /// Reads the int64 axis-spec at <paramref name="inputIndex"/> (opset 10+ input form,
    /// accepting Int64/Int32/Float32), falling back to the opset-1 attribute
    /// <paramref name="attrName"/>, or an empty array when neither is present.
    /// </summary>
    private static long[] ReadAxisInts(GraphNode node, GraphContext ctx, int inputIndex, string attrName)
    {
        if (node.Inputs.Count > inputIndex && !string.IsNullOrEmpty(node.Inputs[inputIndex]))
            return ReadInts(ctx.GetTensor(node.Inputs[inputIndex]));

        int[]? attr = Attr.Ints(node, attrName);
        if (attr is null) return Array.Empty<long>();
        var r = new long[attr.Length];
        for (int i = 0; i < attr.Length; i++) r[i] = attr[i];
        return r;
    }

    /// <summary>Reads a 1-D integer tensor as int64 regardless of its stored dtype.</summary>
    private static long[] ReadInts(Tensor t)
    {
        switch (t.Dtype)
        {
            case ElementType.Int64:
            {
                Span<long> s = t.AsInt64().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i];
                return r;
            }
            case ElementType.Int32:
            {
                Span<int> s = t.AsInt32().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i];
                return r;
            }
            default:
            {
                Span<float> s = t.AsFloat().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = (long)MathF.Round(s[i]);
                return r;
            }
        }
    }
}
