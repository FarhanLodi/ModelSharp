using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.MathOps;

/// <summary>
/// Shared machinery for the cosine-sum window generators (<c>HannWindow</c>, <c>HammingWindow</c>,
/// <c>BlackmanWindow</c>). Each takes a scalar integer <c>size</c> input and emits a 1-D window of
/// that length. The <c>periodic</c> attribute (default 1) divides the phase by N (for spectral
/// analysis / FFT) instead of N-1 (a symmetric window). <c>output_datatype</c> (default 1=FLOAT)
/// selects the output element type.
/// </summary>
public abstract class CosineWindowKernel : IKernel
{
    /// <inheritdoc />
    public abstract string OpType { get; }

    /// <summary>Window sample at index <paramref name="n"/> of <paramref name="denom"/> (the phase divisor).</summary>
    protected abstract double Sample(int n, double denom);

    /// <inheritdoc />
    public void Execute(GraphNode node, GraphContext ctx)
    {
        int size = (int)TensorInts.Read(ctx.GetTensor(node.Inputs[0]))[0];
        if (size < 0) throw new ModelSharpException($"{OpType}: size must be non-negative; got {size}.");
        bool periodic = Attr.Int(node, "periodic", 1) != 0;
        long dtype = Attr.Int(node, "output_datatype", 1);

        double denom = periodic ? size : Math.Max(1, size - 1);
        var w = new double[size];
        for (int n = 0; n < size; n++) w[n] = Sample(n, denom);

        ctx.Set(node.Outputs[0], Materialize(w, dtype));
    }

    private static Tensor Materialize(double[] w, long dtype)
    {
        var shape = new TensorShape(w.Length);
        switch (dtype)
        {
            case 1: // FLOAT
            {
                var b = new float[w.Length];
                for (int i = 0; i < w.Length; i++) b[i] = (float)w[i];
                return new Tensor<float>(shape, b);
            }
            case 11: // DOUBLE
                return new Tensor<double>(shape, (double[])w.Clone());
            default:
                throw new ModelSharpException($"window output_datatype {dtype} is not supported (use 1=FLOAT or 11=DOUBLE).");
        }
    }
}

/// <summary>ONNX <c>HannWindow</c>: <c>0.5 - 0.5*cos(2*pi*n/denom)</c>.</summary>
public sealed class HannWindowKernel : CosineWindowKernel
{
    public override string OpType => "HannWindow";
    protected override double Sample(int n, double denom) => 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / denom);
}

/// <summary>ONNX <c>HammingWindow</c>: <c>0.54347826 - 0.45652174*cos(2*pi*n/denom)</c> (ONNX coefficients).</summary>
public sealed class HammingWindowKernel : CosineWindowKernel
{
    public override string OpType => "HammingWindow";
    protected override double Sample(int n, double denom)
        => 0.54347826086956520 - 0.45652173913043480 * Math.Cos(2.0 * Math.PI * n / denom);
}

/// <summary>ONNX <c>BlackmanWindow</c>: <c>0.42 - 0.5*cos(2*pi*n/denom) + 0.08*cos(4*pi*n/denom)</c>.</summary>
public sealed class BlackmanWindowKernel : CosineWindowKernel
{
    public override string OpType => "BlackmanWindow";
    protected override double Sample(int n, double denom)
        => 0.42 - 0.5 * Math.Cos(2.0 * Math.PI * n / denom) + 0.08 * Math.Cos(4.0 * Math.PI * n / denom);
}
