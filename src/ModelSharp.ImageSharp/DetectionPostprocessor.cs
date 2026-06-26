using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using ModelSharp.Manifest;
using ModelSharp.Pipeline;
using ModelSharp.Tensors;

namespace ModelSharp.ImageSharp;

/// <summary>The raw output layout a detector emits — selects how a row is decoded.</summary>
public enum DetectionLayout
{
    /// <summary>
    /// Layout A — YOLOv5 / YOLOv7. A single <c>[1, N, 5+C]</c> tensor whose rows are
    /// <c>[cx, cy, w, h, objectness, class_scores…]</c>. Final confidence = objectness × max class score.
    /// </summary>
    YoloV5 = 0,

    /// <summary>
    /// Layout B — YOLOv8. A single <em>transposed</em> <c>[1, 4+C, N]</c> tensor laid out
    /// <c>[cx, cy, w, h, class_scores…]</c> with no objectness channel. Confidence = max class score.
    /// </summary>
    YoloV8 = 1,

    /// <summary>
    /// Auto — infer v5 vs v8 from the output shape at decode time. The anchor count N greatly
    /// exceeds the channel count (5+C / 4+C), so the smaller of the last two dims is the channel
    /// dim: channels-last ⇒ <see cref="YoloV5"/>, channels-first ⇒ <see cref="YoloV8"/>. This is the
    /// default when no <c>det_layout</c> is given, so a model "just runs" without the caller
    /// knowing its variant.
    /// </summary>
    Auto = 2,
}

/// <summary>
/// Decodes a YOLO-style detector's raw output into a <see cref="List{T}"/> of
/// <see cref="Detection"/>: argmax over class scores, threshold, convert center form
/// (cx, cy, w, h) to corners, then run greedy per-class <see cref="NonMaxSuppression"/>.
/// <para>
/// Layouts are chosen from the manifest (default <see cref="DetectionLayout.YoloV5"/>; pass
/// <c>det_layout="auto"</c> to infer v5/v8 from the output shape):
/// </para>
/// <list type="bullet">
///   <item><description><b>Layout A</b> (<c>[1, N, 5+C]</c>, YOLOv5/v7) — rows carry an objectness term.</description></item>
///   <item><description><b>Layout B</b> (<c>[1, 4+C, N]</c>, YOLOv8) — transposed, no objectness.</description></item>
/// </list>
/// <para>
/// Tunables come from <see cref="ModelManifest.Extra"/> (parsed with invariant culture):
/// <c>"det_layout"</c>/<c>"layout"</c> (<c>auto</c>|<c>A</c>|<c>B</c>|<c>yolov5</c>|<c>yolov7</c>|<c>yolov8</c>),
/// <c>"iou"</c>/<c>"iou_threshold"</c> (default 0.45),
/// <c>"score_threshold"</c>/<c>"conf_threshold"</c> (default 0.25) and <c>"max_det"</c> (default 300).
/// Explicit constructor arguments win over <c>Extra</c>, which wins over the defaults.
/// An explicit but unrecognized <c>det_layout</c> value is a configuration error and throws.
/// </para>
/// <para>
/// Boxes are returned in the model's own coordinate space (input pixels or normalized);
/// rescaling to the original image is the caller's responsibility. Labels are attached from
/// <see cref="ModelManifest.Labels"/> when present.
/// </para>
/// </summary>
public sealed class DetectionPostprocessor : IPostprocessor
{
    /// <summary>Default IoU threshold for suppression.</summary>
    public const float DefaultIou = 0.45f;

    /// <summary>Default minimum confidence for a detection to be kept.</summary>
    public const float DefaultScoreThreshold = 0.25f;

    /// <summary>Default cap on the number of detections returned.</summary>
    public const int DefaultMaxDetections = 300;

    private readonly ModelManifest _manifest;
    private readonly DetectionLayout _layout;
    private readonly float _iou;
    private readonly float _scoreThreshold;
    private readonly int _maxDetections;

    /// <summary>
    /// Builds a detector postprocessor. Any argument left <see langword="null"/> falls back to the
    /// matching <see cref="ModelManifest.Extra"/> hint, then to the documented default.
    /// </summary>
    public DetectionPostprocessor(
        ModelManifest manifest,
        DetectionLayout? layout = null,
        float? iouThreshold = null,
        float? scoreThreshold = null,
        int? maxDetections = null)
    {
        _manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        IReadOnlyDictionary<string, string> extra = manifest.Extra;
        _layout = layout ?? ParseLayout(extra);
        _iou = iouThreshold ?? ParseFloat(extra, "iou", "iou_threshold", DefaultIou);
        _scoreThreshold = scoreThreshold ?? ParseFloat(extra, "score_threshold", "conf_threshold", DefaultScoreThreshold);
        _maxDetections = maxDetections ?? ParseInt(extra, "max_det", DefaultMaxDetections);
    }

    /// <inheritdoc />
    public object Decode(IReadOnlyDictionary<string, NamedTensor> outputs)
    {
        NamedTensor first = outputs.Values.First();
        float[] data = first.Data.Span.ToArray();
        (int d1, int d2) = LastTwoDims(first.Tensor.Shape);

        // Resolve Auto from the shape: the channel dim (5+C / 4+C) is far smaller than the anchor
        // count N, so channels-last ⇒ v5, channels-first ⇒ v8.
        DetectionLayout layout = _layout == DetectionLayout.Auto
            ? (d2 <= d1 ? DetectionLayout.YoloV5 : DetectionLayout.YoloV8)
            : _layout;

        List<Detection> raw = layout switch
        {
            DetectionLayout.YoloV5 => DecodeYoloV5(data, n: d1, stride: d2),
            DetectionLayout.YoloV8 => DecodeYoloV8(data, channels: d1, n: d2),
            _ => throw new NotSupportedException($"Unsupported detection layout '{layout}'."),
        };

        return NonMaxSuppression.Suppress(raw, _iou, _scoreThreshold, _maxDetections);
    }

    /// <summary>Layout A: each of the <paramref name="n"/> rows is [cx, cy, w, h, obj, class_scores…].</summary>
    private List<Detection> DecodeYoloV5(float[] data, int n, int stride)
    {
        if (stride < 5)
            throw new NotSupportedException(
                $"Layout A expects a [1, N, 5+C] tensor with the last dim ≥ 5; got {stride}.");

        int classCount = stride - 5;
        IReadOnlyList<string>? labels = _manifest.Labels;
        var result = new List<Detection>();

        for (int i = 0; i < n; i++)
        {
            int row = i * stride;
            float objectness = data[row + 4];
            (int classId, float classScore) = ArgMax(data, row + 5, classCount);

            float confidence = objectness * classScore;
            if (confidence < _scoreThreshold) continue;

            result.Add(ToDetection(
                data[row + 0], data[row + 1], data[row + 2], data[row + 3],
                confidence, classId, labels));
        }

        return result;
    }

    /// <summary>Layout B: transposed [4+C, N]; channel <c>ch</c>, anchor <c>i</c> lives at <c>ch*N + i</c>.</summary>
    private List<Detection> DecodeYoloV8(float[] data, int channels, int n)
    {
        if (channels < 4)
            throw new NotSupportedException(
                $"Layout B expects a [1, 4+C, N] tensor with the channel dim ≥ 4; got {channels}.");

        int classCount = channels - 4;
        IReadOnlyList<string>? labels = _manifest.Labels;
        var result = new List<Detection>();

        for (int i = 0; i < n; i++)
        {
            int bestClass = 0;
            float bestScore = classCount > 0 ? data[4 * n + i] : 1f;
            for (int c = 1; c < classCount; c++)
            {
                float s = data[(4 + c) * n + i];
                if (s > bestScore) { bestScore = s; bestClass = c; }
            }

            if (bestScore < _scoreThreshold) continue;

            result.Add(ToDetection(
                data[0 * n + i], data[1 * n + i], data[2 * n + i], data[3 * n + i],
                bestScore, bestClass, labels));
        }

        return result;
    }

    /// <summary>Argmax over a contiguous slice; returns (index 0, 1.0) when the slice is empty.</summary>
    private static (int Index, float Value) ArgMax(float[] data, int offset, int count)
    {
        if (count <= 0) return (0, 1f);
        int best = 0;
        float bestVal = data[offset];
        for (int c = 1; c < count; c++)
        {
            float v = data[offset + c];
            if (v > bestVal) { bestVal = v; best = c; }
        }
        return (best, bestVal);
    }

    /// <summary>Converts a center-form box (cx, cy, w, h) to a corner-form <see cref="Detection"/>.</summary>
    private static Detection ToDetection(
        float cx, float cy, float w, float h,
        float score, int classId, IReadOnlyList<string>? labels)
    {
        float halfW = w * 0.5f, halfH = h * 0.5f;
        return new Detection
        {
            X1 = cx - halfW,
            Y1 = cy - halfH,
            X2 = cx + halfW,
            Y2 = cy + halfH,
            Score = score,
            ClassId = classId,
            Label = labels is not null && classId >= 0 && classId < labels.Count ? labels[classId] : null,
        };
    }

    /// <summary>Extracts the two anchor-bearing dims, tolerating an optional leading batch axis.</summary>
    private static (int D1, int D2) LastTwoDims(TensorShape shape) => shape.Rank switch
    {
        3 => (shape[1], shape[2]),   // [1, D1, D2]
        2 => (shape[0], shape[1]),   // [D1, D2] (batch squeezed)
        _ => throw new NotSupportedException(
            $"Detector output must be rank 2 or 3; got shape {shape}."),
    };

    private static DetectionLayout ParseLayout(IReadOnlyDictionary<string, string> extra)
    {
        if ((extra.TryGetValue("det_layout", out string? raw) || extra.TryGetValue("layout", out raw))
            && !string.IsNullOrWhiteSpace(raw))
        {
            switch (raw.Trim().ToLowerInvariant())
            {
                case "a":
                case "v5":
                case "v7":
                case "yolov5":
                case "yolov7":
                    return DetectionLayout.YoloV5;
                case "b":
                case "v8":
                case "yolov8":
                    return DetectionLayout.YoloV8;
                case "auto":
                    return DetectionLayout.Auto;
                default:
                    throw new NotSupportedException(
                        $"Unrecognized 'det_layout' value '{raw.Trim()}'. " +
                        "Expected one of: auto, A, B, yolov5, yolov7, yolov8 (v5/v7/v8).");
            }
        }
        // Default stays Layout A (YOLOv5) for back-compat; callers wanting shape inference pass
        // det_layout="auto" explicitly (small synthetic tensors are ambiguous, so Auto is opt-in).
        return DetectionLayout.YoloV5;
    }

    private static float ParseFloat(IReadOnlyDictionary<string, string> extra, string key, string altKey, float fallback)
        => (extra.TryGetValue(key, out string? s) || extra.TryGetValue(altKey, out s))
           && float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float v)
            ? v : fallback;

    private static int ParseInt(IReadOnlyDictionary<string, string> extra, string key, int fallback)
        => extra.TryGetValue(key, out string? s)
           && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
            ? v : fallback;
}
