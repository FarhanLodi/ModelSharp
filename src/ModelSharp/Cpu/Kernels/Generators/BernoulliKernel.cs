using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Generators;

/// <summary>
/// ONNX <c>Bernoulli</c>: draws each output element as a Bernoulli trial whose success probability is
/// the corresponding input value (interpreted as a probability in <c>[0, 1]</c>). The optional
/// <c>seed</c> attribute makes the draw reproducible. The <c>dtype</c> attribute (TensorProto int)
/// selects the output type; absent, the input dtype is used. Output element is 1 when
/// <c>uniform() &lt; p</c>, else 0.
/// </summary>
public sealed class BernoulliKernel : IKernel
{
    public string OpType => "Bernoulli";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        int n = checked((int)input.Length);
        Span<float> p = input.AsFloat().Span;

        Random rng = RandomGen.MakeRng(node);
        var bits = new bool[n];
        for (int i = 0; i < n; i++) bits[i] = rng.NextDouble() < p[i];

        long to = node.Attributes.TryGetValue("dtype", out object? v) ? Convert.ToInt64(v) : MapBack(input.Dtype);
        ctx.Set(node.Outputs[0], Materialize(bits, input.Shape, to));
    }

    private static long MapBack(ElementType dt) => dt switch
    {
        ElementType.Float32 => 1,
        ElementType.Float64 => 11,
        ElementType.Int32 => 6,
        ElementType.Int64 => 7,
        ElementType.Boolean => 9,
        _ => 1,
    };

    private static Tensor Materialize(bool[] bits, TensorShape shape, long to)
    {
        int n = bits.Length;
        switch (to)
        {
            case 1: { var b = new float[n]; for (int i = 0; i < n; i++) b[i] = bits[i] ? 1f : 0f; return new Tensor<float>(shape, b); }
            case 11: { var b = new double[n]; for (int i = 0; i < n; i++) b[i] = bits[i] ? 1d : 0d; return new Tensor<double>(shape, b); }
            case 6: { var b = new int[n]; for (int i = 0; i < n; i++) b[i] = bits[i] ? 1 : 0; return new Tensor<int>(shape, b); }
            case 7: { var b = new long[n]; for (int i = 0; i < n; i++) b[i] = bits[i] ? 1L : 0L; return new Tensor<long>(shape, b); }
            case 9: return new Tensor<bool>(shape, (bool[])bits.Clone());
            default: throw new ModelSharpException($"Bernoulli: dtype {to} is not supported.");
        }
    }
}
