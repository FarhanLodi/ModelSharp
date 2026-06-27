using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Activations;
using ModelSharp.Cpu.Kernels.MathOps;
using ModelSharp.Cpu.Kernels.Reduction;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Regression tests for kernels that used to force their inputs to float32 and so threw
/// "Tensor dtype is Int64; expected Float32" when handed the int64 shape/index tensors that real
/// detection and table-structure graphs (PicoDet, RT-DETR, SLANeXt/SLANet, LaTeX-OCR) route through
/// them. Each kernel must now be dtype-preserving on integer inputs, matching the dtype-aware idiom
/// already used by Gather/Concat/Slice and the broadcast binary kernels.
/// </summary>
public class IntegerDtypeKernelTests
{
    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> I64(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    // ---- Identity: dtype-preserving pass-through (RT-DETR / SLANeXt route int64 Shape tensors here) ----

    [Fact]
    public void Identity_PreservesInt64()
    {
        var ctx = Ctx(("x", I64(new[] { 3 }, 7, 8, 9)));
        new IdentityKernel().Execute(Node("Identity", new[] { "x" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new long[] { 7, 8, 9 }, y.AsInt64().Span.ToArray());
    }

    // ---- TopK: dtype-aware data input (PicoDet NMS re-sorts an int64 index tensor) ----

    [Fact]
    public void TopK_Int64Data_PreservesValueDtype()
    {
        // axis -1 (default), largest (default). top-2 of [3,1,4,1] -> values [4,3], indices [2,0].
        var ctx = Ctx(("x", I64(new[] { 1, 4 }, 3, 1, 4, 1)), ("k", I64(new[] { 1 }, 2L)));
        new TopKKernel().Execute(Node("TopK", new[] { "x", "k" }, new[] { "v", "i" }), ctx);

        Tensor v = ctx.GetTensor("v");
        Assert.Equal(ElementType.Int64, v.Dtype);
        Assert.Equal(new long[] { 4, 3 }, v.AsInt64().Span.ToArray());
        Assert.Equal(new long[] { 2, 0 }, ctx.GetTensor("i").AsInt64().Span.ToArray());
    }

    [Fact]
    public void TopK_Float32_StillWorks()
    {
        var ctx = Ctx(("x", F(new[] { 1, 4 }, 3f, 1f, 4f, 1f)), ("k", I64(new[] { 1 }, 2L)));
        new TopKKernel().Execute(Node("TopK", new[] { "x", "k" }, new[] { "v", "i" }), ctx);

        Tensor v = ctx.GetTensor("v");
        Assert.Equal(ElementType.Float32, v.Dtype);
        Assert.Equal(new[] { 4f, 3f }, v.AsFloat().Span.ToArray());
    }

    // ---- Clip: int64 clamp in the integer domain (LaTeX-OCR dynamic 'same'-pad does clamp(pad,0)) ----

    [Fact]
    public void Clip_Int64_ClampsAndPreservesDtype()
    {
        var ctx = Ctx(
            ("x", I64(new[] { 4 }, -5, 0, 3, 10)),
            ("lo", I64(new[] { 1 }, 0L)),
            ("hi", I64(new[] { 1 }, 5L)));
        new ClipKernel().Execute(Node("Clip", new[] { "x", "lo", "hi" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new long[] { 0, 0, 3, 5 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Clip_Float32_StillWorks()
    {
        var ctx = Ctx(
            ("x", F(new[] { 3 }, -1f, 0.5f, 2f)),
            ("lo", F(new[] { 1 }, 0f)),
            ("hi", F(new[] { 1 }, 1f)));
        new ClipKernel().Execute(Node("Clip", new[] { "x", "lo", "hi" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 0f, 0.5f, 1f }, y.AsFloat().Span.ToArray());
    }

    // ---- Ceil/Floor/Round: integer inputs pass through unchanged (round of an int is the int) ----

    [Fact]
    public void Ceil_Int64_PassesThroughUnchanged()
    {
        var ctx = Ctx(("x", I64(new[] { 3 }, 1, 2, 3)));
        new CeilKernel().Execute(Node("Ceil", new[] { "x" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new long[] { 1, 2, 3 }, y.AsInt64().Span.ToArray());
    }

    // ---- Min/Max: integer fold preserving dtype (SLANet/ViT clamp a Gathered dim vs an int const) ----

    [Fact]
    public void Min_Int64_PreservesDtype()
    {
        var ctx = Ctx(("a", I64(new[] { 3 }, 5, 2, 8)), ("b", I64(new[] { 3 }, 3, 7, 1)));
        new MinKernel().Execute(Node("Min", new[] { "a", "b" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new long[] { 3, 2, 1 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Max_Int64_PreservesDtype_WithBroadcast()
    {
        // broadcast a scalar int64 bound against a vector — the (h-1)*42+w <= 504 clamp shape.
        var ctx = Ctx(("a", I64(new[] { 3 }, 5, 2, 8)), ("b", I64(new[] { 1 }, 4L)));
        new MaxKernel().Execute(Node("Max", new[] { "a", "b" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Int64, y.Dtype);
        Assert.Equal(new long[] { 5, 4, 8 }, y.AsInt64().Span.ToArray());
    }

    [Fact]
    public void Max_Float32_StillWorks()
    {
        var ctx = Ctx(("a", F(new[] { 3 }, 5f, 2f, 8f)), ("b", F(new[] { 3 }, 3f, 7f, 1f)));
        new MaxKernel().Execute(Node("Max", new[] { "a", "b" }, new[] { "y" }), ctx);

        Tensor y = ctx.GetTensor("y");
        Assert.Equal(ElementType.Float32, y.Dtype);
        Assert.Equal(new[] { 5f, 7f, 8f }, y.AsFloat().Span.ToArray());
    }
}
