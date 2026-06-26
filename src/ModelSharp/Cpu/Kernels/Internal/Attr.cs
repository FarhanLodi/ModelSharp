using System;
using ModelSharp.Graph;

namespace ModelSharp.Cpu.Kernels.Internal;

/// <summary>Typed readers over a node's boxed attribute values.</summary>
internal static class Attr
{
    public static long Int(GraphNode n, string name, long dflt)
        => n.Attributes.TryGetValue(name, out object? v) ? Convert.ToInt64(v) : dflt;

    public static float Float(GraphNode n, string name, float dflt)
        => n.Attributes.TryGetValue(name, out object? v) ? Convert.ToSingle(v) : dflt;

    public static string Str(GraphNode n, string name, string dflt)
        => n.Attributes.TryGetValue(name, out object? v) && v is string s ? s : dflt;

    public static int[]? Ints(GraphNode n, string name)
    {
        if (!n.Attributes.TryGetValue(name, out object? v)) return null;
        return v switch
        {
            long[] la => Array.ConvertAll(la, x => (int)x),
            int[] ia => ia,
            long l => new[] { (int)l },
            _ => null,
        };
    }

    public static int[] Ints(GraphNode n, string name, int[] dflt) => Ints(n, name) ?? dflt;
}
