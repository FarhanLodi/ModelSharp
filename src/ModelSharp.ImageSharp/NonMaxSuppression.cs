using System;
using System.Collections.Generic;

namespace ModelSharp.ImageSharp;

/// <summary>
/// Greedy, per-class Non-Maximum Suppression for axis-aligned boxes. Detections are
/// ranked by score; the highest-scoring box is kept and any lower-scoring box of the
/// <em>same class</em> whose Intersection-over-Union (IoU) exceeds the threshold is
/// dropped. Boxes of different classes never suppress each other.
/// </summary>
public static class NonMaxSuppression
{
    /// <summary>
    /// Filters overlapping detections with greedy per-class NMS.
    /// </summary>
    /// <param name="detections">Candidate detections (boxes + scores + class ids).</param>
    /// <param name="iouThreshold">Same-class boxes overlapping by more than this IoU are suppressed (default 0.45).</param>
    /// <param name="scoreThreshold">Detections scoring below this are discarded up front (default 0.25).</param>
    /// <param name="maxDetections">Optional cap on the number kept (top-k by score). 0 or negative = unlimited.</param>
    /// <returns>The surviving detections, ordered by descending score.</returns>
    public static List<Detection> Suppress(
        IReadOnlyList<Detection> detections,
        float iouThreshold = 0.45f,
        float scoreThreshold = 0.25f,
        int maxDetections = 0)
    {
        if (detections is null) throw new ArgumentNullException(nameof(detections));

        // 1) Drop sub-threshold candidates.
        var candidates = new List<Detection>(detections.Count);
        foreach (Detection d in detections)
            if (d.Score >= scoreThreshold)
                candidates.Add(d);

        // 2) Rank by descending score (greedy NMS keeps the strongest box first).
        candidates.Sort((a, b) => b.Score.CompareTo(a.Score));

        var kept = new List<Detection>();
        var suppressed = new bool[candidates.Count];

        for (int i = 0; i < candidates.Count; i++)
        {
            if (suppressed[i]) continue;

            Detection winner = candidates[i];
            kept.Add(winner);
            if (maxDetections > 0 && kept.Count >= maxDetections) break;

            // 3) Suppress lower-scoring boxes of the same class that overlap too much.
            for (int j = i + 1; j < candidates.Count; j++)
            {
                if (suppressed[j]) continue;
                Detection other = candidates[j];
                if (other.ClassId != winner.ClassId) continue;   // per-class only
                if (IoU(winner, other) > iouThreshold)
                    suppressed[j] = true;
            }
        }

        return kept;
    }

    /// <summary>Intersection-over-Union of two detections' boxes.</summary>
    public static float IoU(Detection a, Detection b) =>
        IoU(a.X1, a.Y1, a.X2, a.Y2, b.X1, b.Y1, b.X2, b.Y2);

    /// <summary>
    /// Intersection-over-Union of two axis-aligned boxes in corner form. Returns 0 when the
    /// boxes do not overlap or either has non-positive area.
    /// </summary>
    public static float IoU(
        float ax1, float ay1, float ax2, float ay2,
        float bx1, float by1, float bx2, float by2)
    {
        float interX1 = MathF.Max(ax1, bx1);
        float interY1 = MathF.Max(ay1, by1);
        float interX2 = MathF.Min(ax2, bx2);
        float interY2 = MathF.Min(ay2, by2);

        float interW = interX2 - interX1;
        float interH = interY2 - interY1;
        if (interW <= 0f || interH <= 0f) return 0f;

        float intersection = interW * interH;
        float areaA = MathF.Max(0f, ax2 - ax1) * MathF.Max(0f, ay2 - ay1);
        float areaB = MathF.Max(0f, bx2 - bx1) * MathF.Max(0f, by2 - by1);
        float union = areaA + areaB - intersection;
        return union <= 0f ? 0f : intersection / union;
    }
}
