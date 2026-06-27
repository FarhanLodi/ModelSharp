using System;
using System.Numerics;
using System.Threading.Tasks;
using ModelSharp.Cpu.Kernels.Internal;
using ModelSharp.Cpu.Kernels.Linear;
using ModelSharp.Cpu.Kernels.Llm;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Cpu.Kernels.Nn;

/// <summary>
/// Convolution supporting 1-D (NCW) and 2-D (NCHW) inputs. Strides, pads, auto_pad, dilations,
/// group and an optional bias are all honored. The 1-D case (e.g. wav2vec2's feature extractor)
/// is lifted to the 2-D path with a singleton height axis (H=kH=1), so the arithmetic is shared
/// and the 2-D result is byte-for-byte what it always was.
/// </summary>
public sealed class ConvKernel : IKernel
{
    public string OpType => "Conv";

    /// <summary>Below this many output MACs the conv stays serial (dispatch not worth it).</summary>
    private const long ParallelThreshold = 1L << 16;

    public void Execute(GraphNode node, GraphContext ctx)
    {
        Tensor<float> x = ctx.Get(node.Inputs[0]);
        Tensor<float> w = ctx.Get(node.Inputs[1]);
        bool hasBias = node.Inputs.Count > 2 && node.Inputs[2].Length > 0;
        Tensor<float>? bias = hasBias ? ctx.Get(node.Inputs[2]) : null;

        System.ReadOnlySpan<int> xd = x.Shape.Dimensions;
        System.ReadOnlySpan<int> wd = w.Shape.Dimensions;
        int spatial = xd.Length - 2;
        if ((spatial != 1 && spatial != 2) || wd.Length != xd.Length)
            throw new ModelSharpException("Conv supports 1-D (NCW) or 2-D (NCHW) tensors only.");
        bool is1d = spatial == 1;

        // Lift the 1-D layout into the 2-D one with a singleton height axis.
        int N = xd[0];
        int H = is1d ? 1 : xd[2];
        int W = is1d ? xd[2] : xd[3];
        int cout = wd[0], cinPerGroup = wd[1];
        int kH = is1d ? 1 : wd[2];
        int kW = is1d ? wd[2] : wd[3];

        int group = (int)Attr.Int(node, "group", 1);
        int[] strides = Attr.Ints(node, "strides", is1d ? new[] { 1 } : new[] { 1, 1 });
        int[] dil = Attr.Ints(node, "dilations", is1d ? new[] { 1 } : new[] { 1, 1 });
        int sH = is1d ? 1 : strides[0];
        int sW = is1d ? strides[0] : strides[1];
        int dH = is1d ? 1 : dil[0];
        int dW = is1d ? dil[0] : dil[1];

        int padTop, padLeft, padBottom, padRight;
        string autoPad = Attr.Str(node, "auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            bool upper = autoPad == "SAME_UPPER";
            if (is1d) { padTop = padBottom = 0; }
            else Nd.SamePad(H, kH, sH, dH, upper, out padTop, out padBottom);
            Nd.SamePad(W, kW, sW, dW, upper, out padLeft, out padRight);
        }
        else if (autoPad == "VALID")
        {
            padTop = padLeft = padBottom = padRight = 0;
        }
        else if (is1d)
        {
            // 1-D pads are [begin_w, end_w]; height has no padding.
            int[] p = Attr.Ints(node, "pads", new[] { 0, 0 });
            padTop = padBottom = 0; padLeft = p[0]; padRight = p[1];
        }
        else
        {
            int[] p = Attr.Ints(node, "pads", new[] { 0, 0, 0, 0 });
            padTop = p[0]; padLeft = p[1]; padBottom = p[2]; padRight = p[3];
        }

        int outH = (H + padTop + padBottom - (dH * (kH - 1) + 1)) / sH + 1;
        int outW = (W + padLeft + padRight - (dW * (kW - 1) + 1)) / sW + 1;
        int outPerGroup = cout / group;

        // The flat NCHW layout with outH==1 is identical to the squeezed NCW layout, so 1-D
        // output is allocated already-squeezed and the indexing below is unchanged.
        var y = new Tensor<float>(is1d
            ? new TensorShape(N, cout, outW)
            : new TensorShape(N, cout, outH, outW));
        float[] xs = KernelSimd.Array(x), ws = KernelSimd.Array(w), ys = KernelSimd.Array(y);
        float[]? bs = bias is null ? null : KernelSimd.Array(bias);

        int xN = xd[1] * H * W, xC = H * W;
        int wO = cinPerGroup * kH * kW, wC = kH * kW;
        int yN = cout * outH * outW, yC = outH * outW;

        // ---- im2col + BlockedGemm path (heavy general convolutions). ----------------------------
        // Lower each (batch n, group g) to the canonical row-major GEMM Y_g[M,P] = W_g[M,K]·Col[K,P]
        // with M = outPerGroup, K = cinPerGroup·kH·kW, P = outH·outW. Weights are already [M,K]
        // row-major per group and the output is already [M,P] row-major per group, so only the
        // column matrix is materialized. Depthwise (outPerGroup==1) / tiny-K convs keep the
        // cache-friendly direct loop below (im2col's copy is not worth it there).
        {
            int Mg = outPerGroup;
            int Kg = wO;                 // contraction length == weight row length
            int P = outH * outW;         // == yC
            bool useGemm = Kg >= 16 && outPerGroup >= 2 && P >= 8;

            if (useGemm)
            {
                bool fast1x1 =
                    kH == 1 && kW == 1 && sH == 1 && sW == 1 && dH == 1 && dW == 1 &&
                    padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0;

                if (fast1x1)
                {
                    // Col == the input channel block verbatim: X[n, g*cinPerGroup.., :, :] is
                    // already [K,P] row-major (ldb = xC == P), so feed X directly — zero copy.
                    for (int n = 0; n < N; n++)
                        for (int g = 0; g < group; g++)
                            BlockedGemm.Multiply(
                                ws, g * outPerGroup * wO, /*lda*/ Kg,
                                xs, n * xN + g * cinPerGroup * xC, /*ldb*/ P,
                                ys, n * yN + g * outPerGroup * yC, /*ldc*/ P,
                                Mg, P, Kg);
                }
                else
                {
                    float[] col = new float[checked(Kg * P)]; // reused across (n, g)
                    for (int n = 0; n < N; n++)
                        for (int g = 0; g < group; g++)
                        {
                            BuildCol(col, xs, n, g, cinPerGroup, kH, kW, wC, H, W,
                                     outH, outW, sH, sW, dH, dW, padTop, padLeft, xN, xC, P);
                            BlockedGemm.Multiply(
                                ws, g * outPerGroup * wO, /*lda*/ Kg,
                                col, 0, /*ldb*/ P,
                                ys, n * yN + g * outPerGroup * yC, /*ldc*/ P,
                                Mg, P, Kg);
                        }
                }

                // Bias epilogue: add bs[oc] across each output plane.
                if (bs is not null)
                {
                    int wv = Vector<float>.Count;
                    for (int n = 0; n < N; n++)
                        for (int oc = 0; oc < cout; oc++)
                        {
                            float b0 = bs[oc];
                            if (b0 == 0f) continue;
                            int plane = n * yN + oc * yC;
                            var bvec = new Vector<float>(b0);
                            int p = 0, lastv = yC - wv;
                            for (; p <= lastv; p += wv)
                                (new Vector<float>(ys, plane + p) + bvec).CopyTo(ys, plane + p);
                            for (; p < yC; p++) ys[plane + p] += b0;
                        }
                }

                ctx.Set(node.Outputs[0], y);
                return;
            }
        }

        // When the kernel runs contiguously along W (dilation 1) we can SIMD the inner
        // kx dot over an in-bounds sub-window; otherwise fall back to the scalar gather.
        bool simdW = dW == 1 && kW > 1;

        // Parallelize over the independent (batch × output-channel) outer index; each unit
        // owns a disjoint [outH×outW] output plane, so there are no write conflicts.
        int units = N * cout;

        void Unit(int unit)
        {
            int oc = unit % cout;
            int n = unit / cout;
            int g = oc / outPerGroup;
            float b0 = bs is null ? 0f : bs[oc];
            int yPlane = n * yN + oc * yC;

            for (int oy = 0; oy < outH; oy++)
            for (int ox = 0; ox < outW; ox++)
            {
                float sum = b0;
                for (int icg = 0; icg < cinPerGroup; icg++)
                {
                    int ic = g * cinPerGroup + icg;
                    int xBase = n * xN + ic * xC;
                    int wBase = oc * wO + icg * wC;
                    for (int ky = 0; ky < kH; ky++)
                    {
                        int iy = oy * sH - padTop + ky * dH;
                        if (iy < 0 || iy >= H) continue;
                        int xRow = xBase + iy * W;
                        int wRow = wBase + ky * kW;

                        if (simdW)
                        {
                            // ix = ox*sW - padLeft + kx; the in-bounds kx span is contiguous.
                            int ixStart = ox * sW - padLeft;
                            int kxLo = ixStart < 0 ? -ixStart : 0;
                            int kxHi = kW;                          // exclusive
                            if (ixStart + kxHi > W) kxHi = W - ixStart;
                            int len = kxHi - kxLo;
                            if (len > 0)
                                sum += KernelSimd.Dot(xs, xRow + ixStart + kxLo, ws, wRow + kxLo, len);
                        }
                        else
                        {
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = ox * sW - padLeft + kx * dW;
                                if (ix < 0 || ix >= W) continue;
                                sum += xs[xRow + ix] * ws[wRow + kx];
                            }
                        }
                    }
                }
                ys[yPlane + oy * outW + ox] = sum;
            }
        }

        long macs = (long)units * outH * outW * cinPerGroup * kH * kW;
        if (macs >= ParallelThreshold && units > 1)
            Parallel.For(0, units, Unit);
        else
            for (int unit = 0; unit < units; unit++) Unit(unit);

        ctx.Set(node.Outputs[0], y);
    }

    /// <summary>
    /// Builds the im2col matrix Col[K,P] for one (batch n, group g): row k = (icg·kH·kW + ky·kW + kx),
    /// column p = (oy·outW + ox), value = X[n, g·cinPerGroup+icg, iy, ix] with the same
    /// iy/ix/in-bounds expressions as the direct kernel; out-of-bounds taps are written as 0 (the
    /// zero-pad that is bit-identical to the direct path adding nothing). Every entry is
    /// unconditionally written so the reused buffer never leaks stale data.
    /// </summary>
    private static void BuildCol(
        float[] col, float[] xs, int n, int g, int cinPerGroup, int kH, int kW, int wC,
        int H, int W, int outH, int outW, int sH, int sW, int dH, int dW,
        int padTop, int padLeft, int xN, int xC, int P)
    {
        for (int icg = 0; icg < cinPerGroup; icg++)
        {
            int ic = g * cinPerGroup + icg;
            int xBase = n * xN + ic * xC;
            for (int ky = 0; ky < kH; ky++)
                for (int kx = 0; kx < kW; kx++)
                {
                    int colRow = (icg * wC + ky * kW + kx) * P;
                    for (int oy = 0; oy < outH; oy++)
                    {
                        int iy = oy * sH - padTop + ky * dH;
                        int dstRow = colRow + oy * outW;
                        if (iy < 0 || iy >= H) { System.Array.Clear(col, dstRow, outW); continue; }

                        int xRow = xBase + iy * W;
                        if (sW == 1 && dW == 1)
                        {
                            // ix = ox + (kx - padLeft); the valid ox window is one contiguous run.
                            int shift = kx - padLeft;
                            int oxLo = Math.Clamp(-shift, 0, outW);
                            int oxHi = Math.Clamp(W - shift, 0, outW);
                            if (oxHi < oxLo) oxHi = oxLo;
                            if (oxLo > 0) System.Array.Clear(col, dstRow, oxLo);
                            if (oxHi > oxLo) System.Array.Copy(xs, xRow + oxLo + shift, col, dstRow + oxLo, oxHi - oxLo);
                            if (oxHi < outW) System.Array.Clear(col, dstRow + oxHi, outW - oxHi);
                        }
                        else
                        {
                            for (int ox = 0; ox < outW; ox++)
                            {
                                int ix = ox * sW - padLeft + kx * dW;
                                col[dstRow + ox] = (ix >= 0 && ix < W) ? xs[xRow + ix] : 0f;
                            }
                        }
                    }
                }
        }
    }
}
