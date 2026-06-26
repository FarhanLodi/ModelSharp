using System.Collections.Generic;
using System.Linq;
using ModelSharp.ImageSharp;
using ModelSharp.Manifest;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Covers the object-detection decoding path: <see cref="NonMaxSuppression"/> (IoU, per-class
/// greedy suppression, score threshold, top-k) and <see cref="DetectionPostprocessor"/> decoding
/// of the two common YOLO output layouts.
/// </summary>
public class DetectionTests
{
    // ---------------------------------------------------------------- Non-Max Suppression

    [Fact]
    public void IoU_Is_Computed_Correctly_On_A_Hand_Checked_Pair()
    {
        // A = [0,0,10,10] (area 100), B = [5,5,15,15] (area 100).
        // Intersection = [5,5,10,10] = 5*5 = 25. Union = 100 + 100 - 25 = 175. IoU = 25/175.
        var a = Box(0, 0, 10, 10, score: 1f, cls: 0);
        var b = Box(5, 5, 15, 15, score: 1f, cls: 0);

        Assert.Equal(25f / 175f, NonMaxSuppression.IoU(a, b), 5);
    }

    [Fact]
    public void IoU_Is_Zero_For_Disjoint_Boxes()
    {
        var a = Box(0, 0, 10, 10, 1f, 0);
        var b = Box(20, 20, 30, 30, 1f, 0);
        Assert.Equal(0f, NonMaxSuppression.IoU(a, b), 5);
    }

    [Fact]
    public void Overlapping_SameClass_Boxes_Suppress_The_Lower_Score()
    {
        // IoU([0,0,10,10],[1,1,11,11]) = 81 / 119 ≈ 0.68 > 0.45 → the 0.8 box is dropped.
        var strong = Box(0, 0, 10, 10, score: 0.9f, cls: 0);
        var weak = Box(1, 1, 11, 11, score: 0.8f, cls: 0);

        List<Detection> kept = NonMaxSuppression.Suppress(
            new[] { weak, strong }, iouThreshold: 0.45f, scoreThreshold: 0.25f);

        Detection only = Assert.Single(kept);
        Assert.Equal(0.9f, only.Score, 5);
    }

    [Fact]
    public void Overlapping_DifferentClass_Boxes_Are_Not_Suppressed()
    {
        // Same geometry as above (IoU ≈ 0.68) but different classes → both survive.
        var classA = Box(0, 0, 10, 10, score: 0.9f, cls: 0);
        var classB = Box(1, 1, 11, 11, score: 0.8f, cls: 1);

        List<Detection> kept = NonMaxSuppression.Suppress(
            new[] { classA, classB }, iouThreshold: 0.45f, scoreThreshold: 0.25f);

        Assert.Equal(2, kept.Count);
        Assert.Contains(kept, d => d.ClassId == 0);
        Assert.Contains(kept, d => d.ClassId == 1);
    }

    [Fact]
    public void Score_Threshold_Drops_Low_Confidence_Boxes()
    {
        var keep = Box(0, 0, 10, 10, score: 0.9f, cls: 0);
        var drop = Box(50, 50, 60, 60, score: 0.1f, cls: 0);   // disjoint, below threshold

        List<Detection> kept = NonMaxSuppression.Suppress(
            new[] { keep, drop }, iouThreshold: 0.45f, scoreThreshold: 0.25f);

        Detection only = Assert.Single(kept);
        Assert.Equal(0.9f, only.Score, 5);
    }

    [Fact]
    public void MaxDetections_Caps_Results_To_TopK_By_Score()
    {
        var a = Box(0, 0, 5, 5, score: 0.9f, cls: 0);
        var b = Box(20, 20, 25, 25, score: 0.8f, cls: 0);
        var c = Box(40, 40, 45, 45, score: 0.7f, cls: 0);   // all disjoint

        List<Detection> kept = NonMaxSuppression.Suppress(
            new[] { a, b, c }, iouThreshold: 0.45f, scoreThreshold: 0.25f, maxDetections: 2);

        Assert.Equal(2, kept.Count);
        Assert.Equal(new[] { 0.9f, 0.8f }, kept.Select(d => d.Score).ToArray());
    }

    // ---------------------------------------------------------------- Layout A ([1, N, 5+C])

    [Fact]
    public void Postprocessor_Decodes_LayoutA_With_Objectness_And_Labels()
    {
        // Two anchors, C = 2 classes, stride = 5 + 2 = 7.
        // Row 0: cx10 cy10 w4 h6 obj0.9  class[0]=0.8 class[1]=0.1 → conf 0.72, class 0, box [8,7,12,13]
        // Row 1: cx20 cy20 w2 h2 obj0.95 class[0]=0.05 class[1]=0.99 → conf 0.9405, class 1, box [19,19,21,21]
        var data = new float[]
        {
            10, 10, 4, 6, 0.9f, 0.8f, 0.1f,
            20, 20, 2, 2, 0.95f, 0.05f, 0.99f,
        };
        var tensor = new Tensor<float>(new TensorShape(1, 2, 7), data);
        var outputs = Outputs("output", tensor);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Labels = new[] { "cat", "dog" },
        };
        var post = new DetectionPostprocessor(manifest);   // default Layout A

        var dets = (List<Detection>)post.Decode(outputs);
        Assert.Equal(2, dets.Count);

        Detection cat = dets.Single(d => d.ClassId == 0);
        Assert.Equal("cat", cat.Label);
        Assert.Equal(0.72f, cat.Score, 4);
        AssertBox(cat, 8, 7, 12, 13);

        Detection dog = dets.Single(d => d.ClassId == 1);
        Assert.Equal("dog", dog.Label);
        Assert.Equal(0.9405f, dog.Score, 4);
        AssertBox(dog, 19, 19, 21, 21);
    }

    // ---------------------------------------------------------------- Layout B ([1, 4+C, N])

    [Fact]
    public void Postprocessor_Decodes_LayoutB_Transposed_NoObjectness()
    {
        // channels = 4 + C = 6, N = 2 anchors, laid out [channel, anchor] => data[ch*N + anchor].
        // anchor 0: cx30 cy30 w4 h4 class[0]=0.7 class[1]=0.2 → 0.7, class 0, box [28,28,32,32]
        // anchor 1: cx40 cy40 w2 h2 class[0]=0.1 class[1]=0.85 → 0.85, class 1, box [39,39,41,41]
        var data = new float[]
        {
            30, 40,       // cx
            30, 40,       // cy
            4, 2,         // w
            4, 2,         // h
            0.7f, 0.1f,   // class 0
            0.2f, 0.85f,  // class 1
        };
        var tensor = new Tensor<float>(new TensorShape(1, 6, 2), data);
        var outputs = Outputs("output", tensor);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Labels = new[] { "cat", "dog" },
            // Select Layout B through the manifest Extra hint to exercise the parser.
            Extra = new Dictionary<string, string> { ["det_layout"] = "yolov8" },
        };
        var post = new DetectionPostprocessor(manifest);

        var dets = (List<Detection>)post.Decode(outputs);
        Assert.Equal(2, dets.Count);

        Detection cat = dets.Single(d => d.ClassId == 0);
        Assert.Equal("cat", cat.Label);
        Assert.Equal(0.7f, cat.Score, 4);
        AssertBox(cat, 28, 28, 32, 32);

        Detection dog = dets.Single(d => d.ClassId == 1);
        Assert.Equal("dog", dog.Label);
        Assert.Equal(0.85f, dog.Score, 4);
        AssertBox(dog, 39, 39, 41, 41);
    }

    [Fact]
    public void Postprocessor_Applies_Nms_Across_Decoded_Boxes()
    {
        // Two near-identical Layout A rows of the same class → NMS keeps the stronger one.
        // Row 0: cx10 cy10 w10 h10 obj0.9 class0=0.9 → conf 0.81, box [5,5,15,15]
        // Row 1: cx11 cy11 w10 h10 obj0.9 class0=0.8 → conf 0.72, box [6,6,16,16]; IoU ≈ 0.68 → dropped
        var data = new float[]
        {
            10, 10, 10, 10, 0.9f, 0.9f,
            11, 11, 10, 10, 0.9f, 0.8f,
        };
        var tensor = new Tensor<float>(new TensorShape(1, 2, 6), data);   // C = 1
        var outputs = Outputs("output", tensor);

        var manifest = new ModelManifest { Task = ModelTask.ObjectDetection };
        var post = new DetectionPostprocessor(manifest, iouThreshold: 0.45f, scoreThreshold: 0.25f);

        var dets = (List<Detection>)post.Decode(outputs);
        Detection only = Assert.Single(dets);
        Assert.Equal(0.81f, only.Score, 4);
        Assert.Null(only.Label);   // no labels on the manifest
    }

    // ---------------------------------------------------------------- helpers

    private static Detection Box(float x1, float y1, float x2, float y2, float score, int cls) =>
        new() { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Score = score, ClassId = cls };

    private static IReadOnlyDictionary<string, NamedTensor> Outputs(string name, Tensor<float> t) =>
        new Dictionary<string, NamedTensor> { [name] = new NamedTensor(name, t) };

    private static void AssertBox(Detection d, float x1, float y1, float x2, float y2)
    {
        Assert.Equal(x1, d.X1, 4);
        Assert.Equal(y1, d.Y1, 4);
        Assert.Equal(x2, d.X2, 4);
        Assert.Equal(y2, d.Y2, 4);
    }
}
