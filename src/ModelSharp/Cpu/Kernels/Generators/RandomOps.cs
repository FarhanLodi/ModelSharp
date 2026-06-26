using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Generators;

/// <summary>
/// Shared helpers for the ONNX random generators. Sampling is driven by a
/// <see cref="System.Random"/> seeded from the optional <c>seed</c> float attribute so that, when a
/// seed is supplied, the produced sequence is fully reproducible (the test suite relies on this).
/// When no <c>seed</c> is present a fresh, OS-seeded <see cref="System.Random"/> is used.
/// </summary>
internal static class RandomGen
{
    /// <summary>Builds the RNG: seeded from the <c>seed</c> attribute when present, else default-seeded.</summary>
    public static Random MakeRng(GraphNode node) =>
        node.Attributes.TryGetValue("seed", out object? v)
            ? new Random((int)Convert.ToSingle(v))
            : new Random();

    /// <summary>Maps the ONNX <c>dtype</c> attribute (TensorProto data_type int) to an element type.
    /// Defaults to Float32 (the ONNX default) when absent. Only float dtypes are produced.</summary>
    public static ElementType Dtype(GraphNode node, ElementType fallback)
    {
        if (!node.Attributes.TryGetValue("dtype", out object? v)) return fallback;
        long to = Convert.ToInt64(v);
        return to switch
        {
            1 => ElementType.Float32,
            11 => ElementType.Float64,
            _ => throw new ModelSharpException($"Random*: dtype {to} is not supported (Float32/Float64 only)."),
        };
    }

    /// <summary>Fills <paramref name="dst"/> with standard-normal-derived samples mean+scale*N(0,1)
    /// using the Box–Muller transform driven by <paramref name="rng"/>.</summary>
    public static void FillNormal(Span<double> dst, Random rng, double mean, double scale)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            // Box–Muller: two uniforms -> one standard normal (we consume the cosine branch).
            double u1 = 1.0 - rng.NextDouble(); // in (0,1], avoids log(0)
            double u2 = rng.NextDouble();
            double z = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Cos(2.0 * System.Math.PI * u2);
            dst[i] = mean + scale * z;
        }
    }

    /// <summary>Fills <paramref name="dst"/> with uniform samples in [low, high) driven by <paramref name="rng"/>.</summary>
    public static void FillUniform(Span<double> dst, Random rng, double low, double high)
    {
        double range = high - low;
        for (int i = 0; i < dst.Length; i++)
            dst[i] = low + range * rng.NextDouble();
    }

    /// <summary>Emits a sample buffer (computed as double) to the context as the requested float dtype.</summary>
    public static void Emit(GraphContext ctx, string name, TensorShape shape, ElementType dtype, double[] samples)
    {
        switch (dtype)
        {
            case ElementType.Float64:
            {
                ctx.Set(name, new Tensor<double>(shape, samples));
                break;
            }
            default: // Float32
            {
                var buf = new float[samples.Length];
                for (int i = 0; i < buf.Length; i++) buf[i] = (float)samples[i];
                ctx.Set(name, new Tensor<float>(shape, buf));
                break;
            }
        }
    }

    /// <summary>Resolves the output shape from a <c>shape</c> ints attribute (required for the non-Like ops).</summary>
    public static TensorShape ShapeAttr(GraphNode node)
    {
        int[]? dims = Attr.Ints(node, "shape");
        if (dims is null)
            throw new ModelSharpException("Random* requires a 'shape' attribute.");
        return new TensorShape(dims);
    }
}

/// <summary>
/// ONNX <c>RandomNormal</c>: no inputs. Produces a tensor of the attribute-specified <c>shape</c>
/// filled with Gaussian samples (<c>mean</c>, <c>scale</c>) via Box–Muller. <c>dtype</c> selects the
/// float output type (default Float32); the optional <c>seed</c> makes the result reproducible.
/// </summary>
public sealed class RandomNormalKernel : IKernel
{
    public string OpType => "RandomNormal";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        TensorShape shape = RandomGen.ShapeAttr(node);
        ElementType dtype = RandomGen.Dtype(node, ElementType.Float32);
        double mean = Attr.Float(node, "mean", 0f);
        double scale = Attr.Float(node, "scale", 1f);

        var samples = new double[checked((int)shape.Length)];
        RandomGen.FillNormal(samples, RandomGen.MakeRng(node), mean, scale);
        RandomGen.Emit(ctx, node.Outputs[0], shape, dtype, samples);
    }
}

/// <summary>
/// ONNX <c>RandomUniform</c>: no inputs. Produces a tensor of the attribute-specified <c>shape</c>
/// filled with uniform samples in <c>[low, high)</c>. <c>dtype</c> selects the float output type
/// (default Float32); the optional <c>seed</c> makes the result reproducible.
/// </summary>
public sealed class RandomUniformKernel : IKernel
{
    public string OpType => "RandomUniform";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        TensorShape shape = RandomGen.ShapeAttr(node);
        ElementType dtype = RandomGen.Dtype(node, ElementType.Float32);
        double low = Attr.Float(node, "low", 0f);
        double high = Attr.Float(node, "high", 1f);

        var samples = new double[checked((int)shape.Length)];
        RandomGen.FillUniform(samples, RandomGen.MakeRng(node), low, high);
        RandomGen.Emit(ctx, node.Outputs[0], shape, dtype, samples);
    }
}

/// <summary>
/// ONNX <c>RandomNormalLike</c>: takes one input tensor purely to copy its shape (and, by default,
/// its dtype unless the optional <c>dtype</c> attribute overrides it). Fills with Gaussian samples.
/// </summary>
public sealed class RandomNormalLikeKernel : IKernel
{
    public string OpType => "RandomNormalLike";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        TensorShape shape = input.Shape;
        ElementType dtype = RandomGen.Dtype(node, input.Dtype);
        double mean = Attr.Float(node, "mean", 0f);
        double scale = Attr.Float(node, "scale", 1f);

        var samples = new double[checked((int)shape.Length)];
        RandomGen.FillNormal(samples, RandomGen.MakeRng(node), mean, scale);
        RandomGen.Emit(ctx, node.Outputs[0], shape, dtype, samples);
    }
}

/// <summary>
/// ONNX <c>RandomUniformLike</c>: takes one input tensor to copy its shape (and, by default, its
/// dtype unless the optional <c>dtype</c> attribute overrides it). Fills with uniform samples.
/// </summary>
public sealed class RandomUniformLikeKernel : IKernel
{
    public string OpType => "RandomUniformLike";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor input = ctx.GetTensor(node.Inputs[0]);
        TensorShape shape = input.Shape;
        ElementType dtype = RandomGen.Dtype(node, input.Dtype);
        double low = Attr.Float(node, "low", 0f);
        double high = Attr.Float(node, "high", 1f);

        var samples = new double[checked((int)shape.Length)];
        RandomGen.FillUniform(samples, RandomGen.MakeRng(node), low, high);
        RandomGen.Emit(ctx, node.Outputs[0], shape, dtype, samples);
    }
}
