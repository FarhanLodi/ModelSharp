using System;

namespace ModelSharp.Cpu.Kernels.Internal;

/// <summary>Row-major shape/stride helpers and NumPy-style broadcasting math.</summary>
internal static class Nd
{
    /// <summary>Row-major strides for a shape.</summary>
    public static int[] Strides(ReadOnlySpan<int> dims)
    {
        var s = new int[dims.Length];
        int acc = 1;
        for (int i = dims.Length - 1; i >= 0; i--) { s[i] = acc; acc *= dims[i]; }
        return s;
    }

    /// <summary>Strides over <paramref name="rank"/> axes for broadcasting; 0 where the axis is size-1 or padded.</summary>
    public static int[] BroadcastStrides(ReadOnlySpan<int> dims, int rank)
    {
        int[] own = Strides(dims);
        var res = new int[rank];
        int offset = rank - dims.Length;
        for (int i = 0; i < rank; i++)
        {
            int j = i - offset;
            res[i] = (j < 0 || dims[j] == 1) ? 0 : own[j];
        }
        return res;
    }

    /// <summary>The broadcast result shape of two shapes, or throws if incompatible.</summary>
    public static int[] BroadcastShape(ReadOnlySpan<int> a, ReadOnlySpan<int> b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var outd = new int[rank];
        int oa = rank - a.Length, ob = rank - b.Length;
        for (int i = 0; i < rank; i++)
        {
            int da = i < oa ? 1 : a[i - oa];
            int db = i < ob ? 1 : b[i - ob];
            if (da != db && da != 1 && db != 1)
                throw new ModelSharpException($"Shapes are not broadcast-compatible at axis {i}: {da} vs {db}.");
            outd[i] = Math.Max(da, db);
        }
        return outd;
    }

    /// <summary>SAME_UPPER / SAME_LOWER auto-pad amounts for one spatial axis.</summary>
    public static void SamePad(int inSize, int k, int stride, int dilation, bool upper, out int begin, out int end)
    {
        int outSize = (inSize + stride - 1) / stride;
        int needed = (outSize - 1) * stride + ((k - 1) * dilation + 1);
        int total = Math.Max(0, needed - inSize);
        begin = upper ? total / 2 : total - total / 2;
        end = total - begin;
    }
}
