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
/// A3 — Opt-in integration test against a real YOLOv5 / YOLOv8 ONNX detector.
/// No-ops (green) unless the model + a sample image are present in the resolved models dir.
/// See <c>docs/REAL_MODELS.md</c> for the export recipe.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>yolo.onnx</c> (640×640 NCHW detector), <c>detect.jpg</c> (an image to detect on), and
/// optionally <c>coco-labels.txt</c> (one label per line). The export's output layout is selected
/// via the manifest's <c>det_layout</c> hint; override it with the
/// <c>MODELSHARP_YOLO_LAYOUT</c> env var (<c>yolov5</c> | <c>yolov8</c>), default <c>yolov5</c>.</para>
/// </summary>
public class YoloDetectionTests
{
    private readonly ITestOutputHelper _out;
    public YoloDetectionTests(ITestOutputHelper output) => _out = output;

    private const string ModelFile = "yolo.onnx";
    private const string ImageFile = "detect.jpg";
    private const string LabelsFile = "coco-labels.txt";

    [Fact]
    public void Yolo_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath))
        {
            _out.WriteLine($"YOLO model not present ({modelPath}); skipping.");
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
    public void Yolo_Produces_Plausible_Boxes()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath)
            || !RealModelAssets.TryPath(ImageFile, out string imagePath))
        {
            _out.WriteLine("YOLO assets not present; skipping.");
            return;
        }

        ImageSharpRegistration.Ensure();

        // Default "auto": the postprocessor infers v5 vs v8 from the output shape, so the test is
        // green for either export without a hint. Override with MODELSHARP_YOLO_LAYOUT to force one.
        string layout = System.Environment.GetEnvironmentVariable("MODELSHARP_YOLO_LAYOUT") ?? "auto";

        IReadOnlyList<string>? labels = RealModelAssets.TryPath(LabelsFile, out string labelsPath)
            ? File.ReadAllLines(labelsPath)
            : null;

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Layout = TensorLayout.Nchw,
            Width = 640,
            Height = 640,
            Color = ColorOrder.Rgb,
            // YOLO exports take 0–1 scaled RGB with no mean/std offset.
            Mean = new[] { 0f, 0f, 0f },
            Std = new[] { 1f, 1f, 1f },
            Labels = labels,
            Extra = new Dictionary<string, string>
            {
                ["det_layout"] = layout,
            },
        };

        using ModelSharp.Pipeline.Pipeline pipeline = ModelSharp.Pipeline.Pipeline.Load(modelPath, manifest);
        var detections = pipeline.Run<List<Detection>>(imagePath);

        Assert.NotNull(detections);
        _out.WriteLine($"layout={layout}  detections={detections.Count}");
        foreach (Detection d in detections.Take(20)) _out.WriteLine("  " + d);

        // Post-NMS boxes must be well-formed: positive extent and in-range confidence.
        foreach (Detection d in detections)
        {
            Assert.True(d.X2 >= d.X1 && d.Y2 >= d.Y1, $"box has non-positive extent: {d}");
            Assert.True(d.Score >= 0f && d.Score <= 1f, $"score {d.Score} out of [0,1]: {d}");
            Assert.True(d.ClassId >= 0, $"class id must be non-negative: {d}");
        }
    }
}
