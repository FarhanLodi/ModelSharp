using System;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Activations;

/// <summary>
/// com.microsoft <c>FastGelu</c>: the tanh-approximation GELU
/// <c>0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c>, with an optional bias added to <c>x</c> first
/// (and broadcast over the last dimension). Pervasive in transformer ONNX exports.
/// </summary>
public sealed class FastGeluKernel : IKernel
{
    public string OpType => "FastGelu";

    private const float C0 = 0.7978845608028654f;   // sqrt(2/pi)
    private const float C1 = 0.044715f;

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float>? bias = node.Inputs.Count > 1 && node.Inputs[1].Length > 0
            ? ctx.Get(node.Inputs[1]) : null;

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        Span<float> bs = bias is null ? default : bias.Span;
        int bn = bias?.Span.Length ?? 0;

        for (int i = 0; i < xs.Length; i++)
        {
            float v = xs[i] + (bn > 0 ? bs[i % bn] : 0f);
            float inner = C0 * (v + C1 * v * v * v);
            ys[i] = 0.5f * v * (1f + MathF.Tanh(inner));
        }
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// com.microsoft <c>BiasGelu</c>: <c>Gelu(x + bias)</c> with the exact (erf) GELU and the bias
/// broadcast over the last dimension.
/// </summary>
public sealed class BiasGeluKernel : IKernel
{
    public string OpType => "BiasGelu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> bias = ctx.Get(node.Inputs[1]);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, bs = bias.Span, ys = y.Span;
        int bn = bs.Length;
        for (int i = 0; i < xs.Length; i++)
        {
            float v = xs[i] + (bn > 0 ? bs[i % bn] : 0f);
            ys[i] = 0.5f * v * (1f + MathHelpers.Erf(v * 0.70710678f));
        }
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// com.microsoft <c>QuickGelu</c>: the SiLU-style approximation <c>x·sigmoid(α·x)</c>
/// (α defaults to 1.702). Used by CLIP / some GPT exports.
/// </summary>
public sealed class QuickGeluKernel : IKernel
{
    public string OpType => "QuickGelu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 1.702f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++)
            ys[i] = xs[i] * (1f / (1f + MathF.Exp(-alpha * xs[i])));
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Legacy ONNX <c>Affine</c>: elementwise <c>alpha·x + beta</c> (attributes default α=1, β=0).
/// </summary>
public sealed class AffineKernel : IKernel
{
    public string OpType => "Affine";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 1f);
        float beta = Attr.Float(node, "beta", 0f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = alpha * xs[i] + beta;
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>
/// Legacy ONNX <c>ImageScaler</c>: scales an N×C×H×W image by <c>scale</c> and adds a per-channel
/// <c>bias</c> (one value per channel). <c>y[n,c,...] = scale·x[n,c,...] + bias[c]</c>.
/// </summary>
public sealed class ImageScalerKernel : IKernel
{
    public string OpType => "ImageScaler";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float scale = Attr.Float(node, "scale", 1f);
        float[] bias = node.Attributes.TryGetValue("bias", out object? bv) && bv is float[] bf
            ? bf : Array.Empty<float>();

        ReadOnlySpan<int> d = x.Shape.Dimensions;
        if (d.Length < 2) throw new ModelSharpException("ImageScaler expects an N×C×... tensor.");
        int C = d[1];
        int plane = 1; for (int i = 2; i < d.Length; i++) plane *= d[i];

        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        int idx = 0;
        for (int n = 0; n < d[0]; n++)
        for (int c = 0; c < C; c++)
        {
            float b = bias.Length > c ? bias[c] : 0f;
            for (int p = 0; p < plane; p++, idx++) ys[idx] = scale * xs[idx] + b;
        }
        ctx.Set(node.Outputs[0], y);
    }
}
