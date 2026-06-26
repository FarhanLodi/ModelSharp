using System;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Activations;

public sealed class TanhKernel : UnaryKernel { public override string OpType => "Tanh"; protected override float Apply(float x) => MathF.Tanh(x); }
public sealed class ExpKernel : UnaryKernel { public override string OpType => "Exp"; protected override float Apply(float x) => MathF.Exp(x); }
public sealed class LogKernel : UnaryKernel { public override string OpType => "Log"; protected override float Apply(float x) => MathF.Log(x); }
public sealed class SqrtKernel : UnaryKernel { public override string OpType => "Sqrt"; protected override float Apply(float x) => MathF.Sqrt(x); }
public sealed class AbsKernel : UnaryKernel { public override string OpType => "Abs"; protected override float Apply(float x) => MathF.Abs(x); }
public sealed class NegKernel : UnaryKernel { public override string OpType => "Neg"; protected override float Apply(float x) => -x; }
public sealed class ErfKernel : UnaryKernel { public override string OpType => "Erf"; protected override float Apply(float x) => MathHelpers.Erf(x); }
public sealed class IdentityKernel : UnaryKernel { public override string OpType => "Identity"; protected override float Apply(float x) => x; }

/// <summary>GELU activation: 0.5·x·(1 + erf(x/√2)).</summary>
public sealed class GeluKernel : UnaryKernel
{
    public override string OpType => "Gelu";
    protected override float Apply(float x) => 0.5f * x * (1f + MathHelpers.Erf(x * 0.70710678f));
}

/// <summary>Leaky ReLU: x if x ≥ 0 else α·x.</summary>
public sealed class LeakyReluKernel : IKernel
{
    public string OpType => "LeakyRelu";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float alpha = Attr.Float(node, "alpha", 0.01f);
        var y = new Tensor<float>(x.Shape);
        System.Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = xs[i] >= 0f ? xs[i] : alpha * xs[i];
        ctx.Set(node.Outputs[0], y);
    }
}

/// <summary>Clip: clamp to [min, max] (from inputs in opset 11+, or attributes in opset 6).</summary>
public sealed class ClipKernel : IKernel
{
    public string OpType => "Clip";

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        float lo = float.NegativeInfinity, hi = float.PositiveInfinity;
        if (node.Inputs.Count > 1 && node.Inputs[1].Length > 0) lo = ctx.Get(node.Inputs[1]).Span[0];
        if (node.Inputs.Count > 2 && node.Inputs[2].Length > 0) hi = ctx.Get(node.Inputs[2]).Span[0];
        lo = Attr.Float(node, "min", lo);
        hi = Attr.Float(node, "max", hi);

        var y = new Tensor<float>(x.Shape);
        System.Span<float> xs = x.Span, ys = y.Span;
        for (int i = 0; i < xs.Length; i++) { float v = xs[i]; ys[i] = v < lo ? lo : (v > hi ? hi : v); }
        ctx.Set(node.Outputs[0], y);
    }
}
