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
/// A2 — Opt-in integration test against a real ResNet50 / MobileNet ONNX classifier.
/// No-ops (green) unless the model + a sample image are present in the resolved models dir.
/// See <c>docs/REAL_MODELS.md</c> for the export recipe.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>resnet50.onnx</c> (224×224 NCHW ImageNet classifier), <c>sample.jpg</c> (an image to classify),
/// and optionally <c>imagenet-labels.txt</c> (one label per line) to attach human-readable labels.</para>
/// </summary>
public class ImageClassifierTests
{
    private readonly ITestOutputHelper _out;
    public ImageClassifierTests(ITestOutputHelper output) => _out = output;

    private const string ModelFile = "resnet50.onnx";
    private const string ImageFile = "sample.jpg";
    private const string LabelsFile = "imagenet-labels.txt";

    // Standard ImageNet preprocessing (torchvision): RGB, 224×224, scale to [0,1] then mean/std.
    private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    [Fact]
    public void ImageClassifier_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath))
        {
            _out.WriteLine($"image classifier model not present ({modelPath}); skipping.");
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
    public void ImageClassifier_Produces_Plausible_Top1()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath)
            || !RealModelAssets.TryPath(ImageFile, out string imagePath))
        {
            _out.WriteLine("image classifier assets not present; skipping.");
            return;
        }

        // Image processors auto-register via a module initializer; force it to have run.
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
            Mean = ImageNetMean,
            Std = ImageNetStd,
            Labels = labels,
        };

        using ModelSharp.Pipeline.Pipeline pipeline = ModelSharp.Pipeline.Pipeline.Load(modelPath, manifest);
        var top = pipeline.Run<List<Classification>>(imagePath);

        Assert.NotNull(top);
        Assert.NotEmpty(top);
        Classification best = top[0];
        _out.WriteLine($"top-1: index={best.Index} score={best.Score:F4} label={best.Label ?? "(none)"}");
        for (int i = 0; i < top.Count; i++)
            _out.WriteLine($"  #{i + 1} {top[i]}");

        // Softmax probabilities are ordered; the leader is a valid, dominant distribution entry.
        Assert.True(best.Score > 0f && best.Score <= 1f, $"top-1 score {best.Score} out of [0,1].");
        Assert.True(best.Index >= 0, "top-1 class index must be non-negative.");
        Assert.True(top[0].Score >= top[^1].Score, "results must be ordered by descending score.");
    }
}
