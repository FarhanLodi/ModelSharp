using System;

namespace ModelSharp.Cpu.Kernels.Internal;

internal static class MathHelpers
{
    /// <summary>Error function (Abramowitz &amp; Stegun 7.1.26, |error| ≤ 1.5e-7). Used by Erf/Gelu.</summary>
    public static float Erf(float x)
    {
        float sign = MathF.Sign(x);
        float ax = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * ax);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f)
                  * t * MathF.Exp(-ax * ax);
        return sign * y;
    }
}
