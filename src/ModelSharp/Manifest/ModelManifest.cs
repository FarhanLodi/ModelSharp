using System.Collections.Generic;

namespace ModelSharp.Manifest;

/// <summary>
/// The self-describing recipe that lets a model "just run": how to turn input into
/// feed tensors and decode outputs into a typed result. Resolved (in order) from
/// ONNX metadata embedded in the model, a sidecar JSON, or a built-in registry.
/// </summary>
public sealed class ModelManifest
{
    /// <summary>The task this model performs.</summary>
    public ModelTask Task { get; init; } = ModelTask.Unknown;

    /// <summary>Image tensor layout.</summary>
    public TensorLayout Layout { get; init; } = TensorLayout.Nchw;

    /// <summary>Expected input width in pixels (0 = dynamic).</summary>
    public int Width { get; init; }

    /// <summary>Expected input height in pixels (0 = dynamic).</summary>
    public int Height { get; init; }

    /// <summary>Per-channel mean subtracted during normalization.</summary>
    public IReadOnlyList<float> Mean { get; init; } = new[] { 0f, 0f, 0f };

    /// <summary>Per-channel standard deviation divided during normalization.</summary>
    public IReadOnlyList<float> Std { get; init; } = new[] { 1f, 1f, 1f };

    /// <summary>Channel order the model expects.</summary>
    public ColorOrder Color { get; init; } = ColorOrder.Rgb;

    /// <summary>Optional class labels, indexed by output id.</summary>
    public IReadOnlyList<string>? Labels { get; init; }

    /// <summary>Free-form extra hints for task-specific processors.</summary>
    public IReadOnlyDictionary<string, string> Extra { get; init; }
        = new Dictionary<string, string>();
}
