using System.Globalization;

namespace ModelSharp.ImageSharp;

/// <summary>
/// A single detected object: an axis-aligned bounding box in <em>corner form</em>
/// (top-left <see cref="X1"/>,<see cref="Y1"/> and bottom-right <see cref="X2"/>,<see cref="Y2"/>)
/// together with its predicted class and confidence.
/// <para>
/// Coordinates are in whatever space the model emits — usually pixels of the network
/// <em>input</em> (e.g. 640×640) or normalized [0,1]. Mapping them back onto the original
/// image is the caller's responsibility; see <see cref="DetectionPostprocessor"/>.
/// </para>
/// </summary>
public sealed class Detection
{
    /// <summary>Left edge — x of the top-left corner.</summary>
    public float X1 { get; init; }

    /// <summary>Top edge — y of the top-left corner.</summary>
    public float Y1 { get; init; }

    /// <summary>Right edge — x of the bottom-right corner.</summary>
    public float X2 { get; init; }

    /// <summary>Bottom edge — y of the bottom-right corner.</summary>
    public float Y2 { get; init; }

    /// <summary>Confidence score, typically in [0,1].</summary>
    public float Score { get; init; }

    /// <summary>Predicted class index.</summary>
    public int ClassId { get; init; }

    /// <summary>Optional human-readable label, attached from the manifest when available.</summary>
    public string? Label { get; init; }

    /// <summary>Box width (<see cref="X2"/> − <see cref="X1"/>).</summary>
    public float Width => X2 - X1;

    /// <summary>Box height (<see cref="Y2"/> − <see cref="Y1"/>).</summary>
    public float Height => Y2 - Y1;

    public override string ToString() =>
        $"{Label ?? ClassId.ToString(CultureInfo.InvariantCulture)} ({Score:P1}) " +
        $"[{X1:F1}, {Y1:F1}, {X2:F1}, {Y2:F1}]";
}
