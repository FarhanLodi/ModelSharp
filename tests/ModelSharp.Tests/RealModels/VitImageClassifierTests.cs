using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Graph;
using ModelSharp.ImageSharp;
using ModelSharp.Manifest;
using ModelSharp.Onnx;
using ModelSharp.Pipeline;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Opt-in integration test against a real Vision Transformer image classifier ONNX export
/// (<c>google/vit-base-patch16-224</c>, Xenova/optimum form). Drives the standard
/// <see cref="ModelTask.ImageClassification"/> pipeline on the shared <c>sample.jpg</c> and asserts a
/// sane, dominant top-1. No-ops (green) unless the model + image are present in the resolved models dir.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>vit.onnx</c> (224×224 NCHW classifier, 1000 ImageNet classes), <c>sample.jpg</c>, and optionally
/// <c>imagenet-labels.txt</c>. Source: https://huggingface.co/Xenova/vit-base-patch16-224
/// (onnx/model.onnx). ViT preprocessing is RGB, 224×224, scaled to [0,1] then mean/std = 0.5.</para>
/// </summary>
public class VitImageClassifierTests
{
    private readonly ITestOutputHelper _out;
    public VitImageClassifierTests(ITestOutputHelper output) => _out = output;

    private const string ModelFile = "vit.onnx";
    private const string ImageFile = "sample.jpg";
    private const string LabelsFile = "imagenet-labels.txt";

    // ViT (CLIP-style) preprocessing: RGB, 224×224, scale to [0,1], then (x-0.5)/0.5 -> [-1, 1].
    private static readonly float[] VitMean = { 0.5f, 0.5f, 0.5f };
    private static readonly float[] VitStd = { 0.5f, 0.5f, 0.5f };

    [Fact]
    public void Vit_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath))
        {
            _out.WriteLine($"ViT model not present ({modelPath}); skipping.");
            return;
        }

        ModelGraph g = OnnxModelLoader.LoadModel(modelPath);
        KernelRegistry registry = KernelRegistry.CreateDefault();
        var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
        var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();

        _out.WriteLine($"nodes={g.Nodes.Count}  distinctOps={distinct.Count}  initializers={g.Initializers.Count}");
        _out.WriteLine("ALL OPS: " + string.Join(", ", distinct));
        _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));
        Assert.True(missing.Count == 0, "Unsupported ops: " + string.Join(", ", missing));
    }

    [Fact]
    public void Vit_Produces_Plausible_Top1()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath)
            || !RealModelAssets.TryPath(ImageFile, out string imagePath))
        {
            _out.WriteLine("ViT assets not present; skipping.");
            return;
        }

        ImageSharpRegistration.Ensure();

        IReadOnlyList<string>? labels = RealModelAssets.TryPath(LabelsFile, out string labelsPath)
            ? File.ReadAllLines(labelsPath)
            : null;

        var manifest = new ModelManifest
        {
            Task = ModelTask.ImageClassification,
            Layout = TensorLayout.Nchw,
            Width = 224,
            Height = 224,
            Color = ColorOrder.Rgb,
            Mean = VitMean,
            Std = VitStd,
            Labels = labels,
        };

        using ModelSharp.Pipeline.Pipeline pipeline = ModelSharp.Pipeline.Pipeline.Load(modelPath, manifest);
        var top = pipeline.Run<List<Classification>>(imagePath);

        Assert.NotNull(top);
        Assert.NotEmpty(top);
        Classification best = top[0];
        _out.WriteLine($"top-1: index={best.Index} score={best.Score:F4} label={best.Label ?? "(none)"}");
        for (int i = 0; i < System.Math.Min(5, top.Count); i++)
            _out.WriteLine($"  #{i + 1} {top[i]}");

        Assert.True(best.Score > 0f && best.Score <= 1f, $"top-1 score {best.Score} out of [0,1].");
        Assert.True(best.Index >= 0 && best.Index < 1000, "top-1 class index must be a valid ImageNet id.");
        Assert.True(top[0].Score >= top[^1].Score, "results must be ordered by descending score.");
        // A correctly wired ViT should be confident on a clear subject (sample.jpg).
        Assert.True(best.Score > 0.15f, $"top-1 score {best.Score:F3} is implausibly diffuse for ViT.");
    }
}
