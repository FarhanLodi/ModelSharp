using System;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Activations;

/// <summary>Softplus: <c>log(1 + exp(x))</c>.</summary>
public sealed class SoftplusKernel : UnaryKernel
{
    public override string OpType => "Softplus";
    protected override float Apply(float x) => MathF.Log(1f + MathF.Exp(x));
}

/// <summary>Softsign: <c>x / (1 + |x|)</c>.</summary>
public sealed class SoftsignKernel : UnaryKernel
{
    public override string OpType => "Softsign";
    protected override float Apply(float x) => x / (1f + MathF.Abs(x));
}

/// <summary>Mish: <c>x · tanh(softplus(x))</c>.</summary>
public sealed class MishKernel : UnaryKernel
{
    public override string OpType => "Mish";
    protected override float Apply(float x) => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));
}

/// <summary>ELU: <c>x</c> if x ≥ 0 else <c>α·(exp(x) − 1)</c>. <c>alpha</c> default 1.</summary>
public sealed class EluKernel : IKernel
{
    public string OpType => "Elu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 1f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = xs[i] >= 0f ? xs[i] : alpha * (MathF.Exp(xs[i]) - 1f);
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>SELU: <c>γ·x</c> if x &gt; 0 else <c>γ·α·(exp(x) − 1)</c>.
/// Defaults <c>alpha</c>=1.67326319, <c>gamma</c>=1.05070102.</summary>
public sealed class SeluKernel : IKernel
{
    public string OpType => "Selu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 1.67326319217681884765625f);
        float gamma = Attr.Float(node, "gamma", 1.05070102214813232421875f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++)
            ys[i] = xs[i] > 0f ? gamma * xs[i] : gamma * (alpha * (MathF.Exp(xs[i]) - 1f));
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>HardSigmoid: <c>max(0, min(1, α·x + β))</c>. Defaults <c>alpha</c>=0.2, <c>beta</c>=0.5.</summary>
public sealed class HardSigmoidKernel : IKernel
{
    public string OpType => "HardSigmoid";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 0.2f);
        float beta = Attr.Float(node, "beta", 0.5f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++)
        {
            float v = alpha * xs[i] + beta;
            ys[i] = v < 0f ? 0f : (v > 1f ? 1f : v);
        }
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>PReLU: <c>x</c> if x ≥ 0 else <c>slope·x</c>, with the slope broadcast over x (NumPy-style).</summary>
public sealed class PReluKernel : BroadcastBinaryKernel
{
    public override string OpType => "PRelu";
    protected override float Apply(float x, float slope) => x >= 0f ? x : slope * x;
}

/// <summary>HardSwish: <c>x · max(0, min(6, x + 3)) / 6</c> (i.e. x·relu6(x+3)/6).</summary>
public sealed class HardSwishKernel : UnaryKernel
{
    public override string OpType => "HardSwish";

    protected override float Apply(float x)
    {
        float v = x + 3f;
        v = v < 0f ? 0f : (v > 6f ? 6f : v); // relu6(x + 3)
        return x * v / 6f;
    }
}

/// <summary>ThresholdedRelu: <c>x</c> if x &gt; α else 0. <c>alpha</c> default 1.</summary>
public sealed class ThresholdedReluKernel : IKernel
{
    public string OpType => "ThresholdedRelu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 1f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = xs[i] > alpha ? xs[i] : 0f;
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>CELU: <c>max(0, x) + min(0, α·(exp(x/α) − 1))</c>. <c>alpha</c> default 1.</summary>
public sealed class CeluKernel : IKernel
{
    public string OpType => "Celu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 1f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++)
        {
            float v = xs[i];
            float pos = MathF.Max(0f, v);
            float neg = MathF.Min(0f, alpha * (MathF.Exp(v / alpha) - 1f));
            ys[i] = pos + neg;
        }
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>Shrink: <c>x − bias</c> if x &gt; λ, <c>x + bias</c> if x &lt; −λ, else 0.
/// Defaults <c>bias</c>=0, <c>lambd</c>=0.5.</summary>
public sealed class ShrinkKernel : IKernel
{
    public string OpType => "Shrink";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float bias = Attr.Float(node, "bias", 0f);
        float lambd = Attr.Float(node, "lambd", 0.5f);
        var y = new Tensor<float>(x.Shape);
        Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++)
        {
            float v = xs[i];
            ys[i] = v > lambd ? v - bias : (v < -lambd ? v + bias : 0f);
        }
        ctx.Set(node.Outputs[0], y);
    }
}
