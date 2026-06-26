using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.ImageSharp;
using ModelSharp.Manifest;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// End-to-end <see cref="DetectionPostprocessor.Decode"/> coverage with hand-built YOLO output
/// tensors: layout selection from the manifest, Extra-driven confidence / IoU / max_det thresholds
/// (including the <c>conf_threshold</c>/<c>iou_threshold</c> aliases), center-to-corner box decoding,
/// per-class NMS over the decoded boxes, and configuration-error reporting. No model or image needed.
/// </summary>
public class DetectionPipelineTests
{
    // ---------------------------------------------------------------- layout selection from Extra

    [Theory]
    [InlineData("yolov5")]
    [InlineData("v5")]
    [InlineData("yolov7")]
    [InlineData("a")]
    [InlineData("A")]
    public void LayoutA_Selected_By_Various_Extra_Aliases(string alias)
    {
        // One Layout A row: cx20 cy20 w4 h4 obj0.9 class0=0.9 => conf 0.81, box [18,18,22,22].
        var data = new[] { 20f, 20f, 4f, 4f, 0.9f, 0.9f };
        var outputs = Outputs(new TensorShape(1, 1, 6), data);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Extra = new Dictionary<string, string> { ["det_layout"] = alias },
        };

        var dets = (List<Detection>)new DetectionPostprocessor(manifest).Decode(outputs);
        Detection only = Assert.Single(dets);
        Assert.Equal(0, only.ClassId);
        Assert.Equal(0.81f, only.Score, 4);
        AssertBox(only, 18, 18, 22, 22);
    }

    [Theory]
    [InlineData("yolov8")]
    [InlineData("v8")]
    [InlineData("b")]
    [InlineData("B")]
    public void LayoutB_Selected_By_Various_Extra_Aliases(string alias)
    {
        // One Layout B anchor (transposed [4+C, N] with N=1): cx50 cy50 w10 h10 class0=0.6.
        // channels = 5 (4 box + 1 class) => box [45,45,55,55], conf 0.6, class 0.
        var data = new[] { 50f, 50f, 10f, 10f, 0.6f };
        var outputs = Outputs(new TensorShape(1, 5, 1), data);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Extra = new Dictionary<string, string> { ["det_layout"] = alias },
        };

        var dets = (List<Detection>)new DetectionPostprocessor(manifest).Decode(outputs);
        Detection only = Assert.Single(dets);
        Assert.Equal(0, only.ClassId);
        Assert.Equal(0.6f, only.Score, 4);
        AssertBox(only, 45, 45, 55, 55);
    }

    // ---------------------------------------------------------------- Extra thresholds

    [Fact]
    public void Conf_And_Iou_Threshold_Aliases_From_Extra_Are_Honored()
    {
        // Two overlapping same-class Layout A boxes plus one weak box.
        // Row 0: cx10 cy10 w10 h10 obj1 class0=0.9 => 0.90, box [5,5,15,15]
        // Row 1: cx11 cy11 w10 h10 obj1 class0=0.8 => 0.80, box [6,6,16,16]; IoU(row0,row1) ~ 0.68
        // Row 2: cx80 cy80 w4 h4  obj1 class0=0.4 => 0.40, box [78,78,82,82] (disjoint)
        var data = new[]
        {
            10f, 10f, 10f, 10f, 1f, 0.9f,
            11f, 11f, 10f, 10f, 1f, 0.8f,
            80f, 80f, 4f, 4f, 1f, 0.4f,
        };
        var outputs = Outputs(new TensorShape(1, 3, 6), data);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Extra = new Dictionary<string, string>
            {
                // Alias keys (not the canonical iou / score_threshold) must still be read.
                ["conf_threshold"] = "0.5",   // drops the 0.40 box
                ["iou_threshold"] = "0.45",   // suppresses the overlapping 0.80 box
            },
        };

        var dets = (List<Detection>)new DetectionPostprocessor(manifest).Decode(outputs);
        Detection only = Assert.Single(dets);
        Assert.Equal(0.90f, only.Score, 4);
        AssertBox(only, 5, 5, 15, 15);
    }

    [Fact]
    public void High_Iou_Threshold_Keeps_Both_Overlapping_Boxes()
    {
        // Same overlapping pair as above (IoU ~ 0.68). With iou_threshold 0.9 neither suppresses.
        var data = new[]
        {
            10f, 10f, 10f, 10f, 1f, 0.9f,
            11f, 11f, 10f, 10f, 1f, 0.8f,
        };
        var outputs = Outputs(new TensorShape(1, 2, 6), data);

        var manifest = new ModelManifest { Task = ModelTask.ObjectDetection };
        var post = new DetectionPostprocessor(manifest, iouThreshold: 0.9f);

        var dets = (List<Detection>)post.Decode(outputs);
        Assert.Equal(2, dets.Count);
        Assert.Equal(new[] { 0.9f, 0.8f }, dets.Select(d => d.Score).ToArray());
    }

    [Fact]
    public void MaxDet_From_Extra_Caps_Returned_Detections()
    {
        // Three disjoint same-class boxes; max_det=2 keeps the two strongest.
        var data = new[]
        {
            10f, 10f, 4f, 4f, 1f, 0.9f,
            40f, 40f, 4f, 4f, 1f, 0.8f,
            70f, 70f, 4f, 4f, 1f, 0.7f,
        };
        var outputs = Outputs(new TensorShape(1, 3, 6), data);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Extra = new Dictionary<string, string> { ["max_det"] = "2" },
        };

        var dets = (List<Detection>)new DetectionPostprocessor(manifest).Decode(outputs);
        Assert.Equal(2, dets.Count);
        Assert.Equal(new[] { 0.9f, 0.8f }, dets.Select(d => d.Score).ToArray());
    }

    // ---------------------------------------------------------------- multi-class decoding

    [Fact]
    public void LayoutB_Argmax_Picks_The_Highest_Class_Per_Anchor()
    {
        // channels = 4 + 3 classes = 7, N = 2. Layout: data[ch*N + anchor].
        // anchor 0: cx30 cy30 w6 h6 class[0]=0.1 class[1]=0.8 class[2]=0.2 => class 1, box [27,27,33,33]
        // anchor 1: cx60 cy60 w8 h8 class[0]=0.7 class[1]=0.1 class[2]=0.3 => class 0, box [56,56,64,64]
        var data = new[]
        {
            30f, 60f,         // cx
            30f, 60f,         // cy
            6f, 8f,           // w
            6f, 8f,           // h
            0.1f, 0.7f,       // class 0
            0.8f, 0.1f,       // class 1
            0.2f, 0.3f,       // class 2
        };
        var outputs = Outputs(new TensorShape(1, 7, 2), data);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Labels = new[] { "a", "b", "c" },
            Extra = new Dictionary<string, string> { ["det_layout"] = "yolov8" },
        };

        var dets = (List<Detection>)new DetectionPostprocessor(manifest).Decode(outputs);
        Assert.Equal(2, dets.Count);

        Detection b = dets.Single(d => d.ClassId == 1);
        Assert.Equal("b", b.Label);
        Assert.Equal(0.8f, b.Score, 4);
        AssertBox(b, 27, 27, 33, 33);

        Detection a = dets.Single(d => d.ClassId == 0);
        Assert.Equal("a", a.Label);
        Assert.Equal(0.7f, a.Score, 4);
        AssertBox(a, 56, 56, 64, 64);
    }

    // ---------------------------------------------------------------- squeezed rank-2 input

    [Fact]
    public void Rank2_LayoutA_Tensor_With_Batch_Squeezed_Is_Decoded()
    {
        // [N, 5+C] with the batch axis dropped. One row: cx20 cy20 w4 h4 obj0.9 class0=0.9.
        var data = new[] { 20f, 20f, 4f, 4f, 0.9f, 0.9f };
        var outputs = Outputs(new TensorShape(1, 6), data);

        var manifest = new ModelManifest { Task = ModelTask.ObjectDetection };
        var dets = (List<Detection>)new DetectionPostprocessor(manifest).Decode(outputs);

        Detection only = Assert.Single(dets);
        Assert.Equal(0.81f, only.Score, 4);
        AssertBox(only, 18, 18, 22, 22);
    }

    // ---------------------------------------------------------------- configuration errors

    [Fact]
    public void Unrecognized_Layout_Hint_Throws_A_Clear_Error()
    {
        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Extra = new Dictionary<string, string> { ["det_layout"] = "yolov99" },
        };

        var ex = Assert.Throws<NotSupportedException>(() => new DetectionPostprocessor(manifest));
        Assert.Contains("det_layout", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void LayoutA_Tensor_With_Too_Small_Stride_Throws()
    {
        // Last dim = 4 (< 5): cannot carry [cx,cy,w,h,obj,...].
        var outputs = Outputs(new TensorShape(1, 1, 4), new[] { 1f, 2f, 3f, 4f });
        var manifest = new ModelManifest { Task = ModelTask.ObjectDetection };   // default Layout A

        Assert.Throws<NotSupportedException>(
            () => new DetectionPostprocessor(manifest).Decode(outputs));
    }

    [Fact]
    public void LayoutB_Tensor_With_Too_Few_Channels_Throws()
    {
        // channel dim = 3 (< 4): cannot carry [cx,cy,w,h,...].
        var outputs = Outputs(new TensorShape(1, 3, 2), new[] { 1f, 2f, 3f, 4f, 5f, 6f });
        var manifest = new ModelManifest
        {
            Task = ModelTask.ObjectDetection,
            Extra = new Dictionary<string, string> { ["det_layout"] = "yolov8" },
        };

        Assert.Throws<NotSupportedException>(
            () => new DetectionPostprocessor(manifest).Decode(outputs));
    }

    [Fact]
    public void Boxes_Are_Returned_In_Model_Coordinate_Space()
    {
        // A normalized-coordinate box (cx,cy,w,h in [0,1]) decodes to corners in the same space —
        // the postprocessor does not rescale; that is the caller's job.
        var data = new[] { 0.5f, 0.5f, 0.2f, 0.4f, 1f, 0.9f };
        var outputs = Outputs(new TensorShape(1, 1, 6), data);

        var manifest = new ModelManifest { Task = ModelTask.ObjectDetection };
        var dets = (List<Detection>)new DetectionPostprocessor(manifest).Decode(outputs);

        Detection only = Assert.Single(dets);
        AssertBox(only, 0.4f, 0.3f, 0.6f, 0.7f);
    }

    // ---------------------------------------------------------------- helpers

    private static IReadOnlyDictionary<string, NamedTensor> Outputs(TensorShape shape, float[] data) =>
        new Dictionary<string, NamedTensor>
        {
            ["output"] = new NamedTensor("output", new Tensor<float>(shape, data)),
        };

    private static void AssertBox(Detection d, float x1, float y1, float x2, float y2)
    {
        Assert.Equal(x1, d.X1, 4);
        Assert.Equal(y1, d.Y1, 4);
        Assert.Equal(x2, d.X2, 4);
        Assert.Equal(y2, d.Y2, 4);
    }
}
