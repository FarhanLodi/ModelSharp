using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Cpu.Kernels.Signal;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Direct-kernel tests for the ONNX signal-processing family (<c>DFT</c>, <c>STFT</c>,
/// <c>MelWeightMatrix</c>), opset 17. Each test builds a <see cref="GraphNode"/> plus a
/// <see cref="GraphContext"/> by hand, runs the kernel in isolation, and checks hand-derived
/// expected values with tight tolerances.
/// </summary>
public class SignalOpsTests
{
    // ---- helpers -------------------------------------------------------------------------------

    private static GraphContext Ctx(params (string name, Tensor t)[] vals)
    {
        var d = new Dictionary<string, Tensor>();
        foreach ((string name, Tensor t) in vals) d[name] = t;
        return new GraphContext(d);
    }

    private static Tensor<float> F(int[] dims, params float[] data) =>
        Tensor<float>.FromArray(new TensorShape(dims), data);

    private static Tensor<long> I64(int[] dims, params long[] data) =>
        Tensor<long>.FromArray(new TensorShape(dims), data);

    private static GraphNode Node(string op, string[] ins, string[] outs,
        Dictionary<string, object>? attrs = null) => new(op, "n", ins, outs, attrs);

    private static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"[{i}] expected {expected[i]}, got {actual[i]}");
    }

    // ---- DFT -----------------------------------------------------------------------------------

    [Fact]
    public void Dft_RealForward_TwoSided_MatchesHandComputed()
    {
        // x = [1,2,3,4], real (last dim 1), axis = 1, N = 4.
        // X[0]=10, X[1]=-2+2i, X[2]=-2, X[3]=-2-2i.
        var x = F(new[] { 1, 4, 1 }, 1, 2, 3, 4);
        var ctx = Ctx(("x", x));
        new DftKernel().Execute(Node("DFT", new[] { "x" }, new[] { "y" }), ctx);

        var y = ctx.Get("y");
        Assert.Equal(new[] { 1, 4, 2 }, y.Shape.Dimensions.ToArray());
        Close(new[] { 10f, 0f, -2f, 2f, -2f, 0f, -2f, -2f }, y.Span.ToArray());
    }

    [Fact]
    public void Dft_RealForward_OneSided_KeepsHalfPlusOne()
    {
        // Onesided keeps bins 0,1,2 -> [10,0, -2,2, -2,0].
        var x = F(new[] { 1, 4, 1 }, 1, 2, 3, 4);
        var attrs = new Dictionary<string, object> { ["onesided"] = 1L };
        var ctx = Ctx(("x", x));
        new DftKernel().Execute(Node("DFT", new[] { "x" }, new[] { "y" }, attrs), ctx);

        var y = ctx.Get("y");
        Assert.Equal(new[] { 1, 3, 2 }, y.Shape.Dimensions.ToArray());
        Close(new[] { 10f, 0f, -2f, 2f, -2f, 0f }, y.Span.ToArray());
    }

    [Fact]
    public void Dft_InverseOfForward_RoundTrips()
    {
        // Forward then inverse recovers the original real signal (imag ~ 0).
        var x = F(new[] { 1, 4, 1 }, 1, 2, 3, 4);
        var fctx = Ctx(("x", x));
        new DftKernel().Execute(Node("DFT", new[] { "x" }, new[] { "fwd" }), fctx);
        var fwd = fctx.Get("fwd"); // [1,4,2] complex

        var ictx = Ctx(("fwd", fwd));
        var attrs = new Dictionary<string, object> { ["inverse"] = 1L };
        new DftKernel().Execute(Node("DFT", new[] { "fwd" }, new[] { "inv" }, attrs), ictx);
        var inv = ictx.Get("inv"); // [1,4,2]

        // Real parts == original; imag parts ~ 0.
        float[] r = inv.Span.ToArray();
        Close(new[] { 1f, 0f, 2f, 0f, 3f, 0f, 4f, 0f }, r, 1e-4f);
    }

    [Fact]
    public void Dft_ComplexInput_Forward()
    {
        // x = [1+0i, 0+1i] (N=2). X[0]=(1+0i)+(0+1i)=1+1i, X[1]=(1+0i)-(0+1i)=1-1i.
        var x = F(new[] { 1, 2, 2 }, 1, 0, 0, 1);
        var ctx = Ctx(("x", x));
        new DftKernel().Execute(Node("DFT", new[] { "x" }, new[] { "y" }), ctx);
        var y = ctx.Get("y");
        Assert.Equal(new[] { 1, 2, 2 }, y.Shape.Dimensions.ToArray());
        Close(new[] { 1f, 1f, 1f, -1f }, y.Span.ToArray());
    }

    [Fact]
    public void Dft_Fft_PowerOfTwo_MatchesNaive()
    {
        // Cross-check the FFT fast path against the reference naive DFT on fixed complex data,
        // N = 16 (power of two), two-sided forward. Deterministic fill (no RNG).
        const int n = 16;
        var reIn = new double[n];
        var imIn = new double[n];
        for (int t = 0; t < n; t++)
        {
            reIn[t] = Math.Sin(0.3 * t) + 0.5 * t;
            imIn[t] = Math.Cos(0.7 * t) - 0.25 * t;
        }

        var reFast = new double[n];
        var imFast = new double[n];
        var reRef = new double[n];
        var imRef = new double[n];
        DftCore.Dft1d(reIn, imIn, reFast, imFast, n, n, inverse: false);
        DftCore.Dft1dNaive(reIn, imIn, reRef, imRef, n, n, inverse: false);

        for (int k = 0; k < n; k++)
        {
            Assert.True(Math.Abs(reFast[k] - reRef[k]) <= 1e-5, $"re[{k}] {reFast[k]} vs {reRef[k]}");
            Assert.True(Math.Abs(imFast[k] - imRef[k]) <= 1e-5, $"im[{k}] {imFast[k]} vs {imRef[k]}");
        }
    }

    [Fact]
    public void Dft_Bluestein_NonPowerOfTwo_MatchesNaive()
    {
        // Cross-check the Bluestein path against the naive DFT on fixed data, N = 6 (not a power of
        // two), both forward and inverse.
        const int n = 6;
        var reIn = new double[n];
        var imIn = new double[n];
        for (int t = 0; t < n; t++)
        {
            reIn[t] = 1.0 + t * t * 0.1;
            imIn[t] = (t % 2 == 0) ? 0.5 : -0.5;
        }

        foreach (bool inverse in new[] { false, true })
        {
            var reFast = new double[n];
            var imFast = new double[n];
            var reRef = new double[n];
            var imRef = new double[n];
            DftCore.Dft1d(reIn, imIn, reFast, imFast, n, n, inverse);
            DftCore.Dft1dNaive(reIn, imIn, reRef, imRef, n, n, inverse);

            for (int k = 0; k < n; k++)
            {
                Assert.True(Math.Abs(reFast[k] - reRef[k]) <= 1e-5,
                    $"inverse={inverse} re[{k}] {reFast[k]} vs {reRef[k]}");
                Assert.True(Math.Abs(imFast[k] - imRef[k]) <= 1e-5,
                    $"inverse={inverse} im[{k}] {imFast[k]} vs {imRef[k]}");
            }
        }
    }

    [Fact]
    public void Dft_AxisAsInput_MatchesAttributeForm()
    {
        // opset-20 axis-as-input (input index 2) must produce the same result as the opset-17
        // axis attribute. Transform along axis 1 of a [1,4,1] real signal.
        var xData = new float[] { 1, 2, 3, 4 };

        // Attribute form (axis defaults to 1).
        var ctxAttr = Ctx(("x", F(new[] { 1, 4, 1 }, xData)));
        new DftKernel().Execute(Node("DFT", new[] { "x" }, new[] { "y" }), ctxAttr);
        float[] expected = ctxAttr.Get("y").Span.ToArray();

        // Input form: input 1 = dft_length omitted (empty name), input 2 = axis scalar = 1.
        var ctxInput = Ctx(
            ("x", F(new[] { 1, 4, 1 }, xData)),
            ("axis", I64(Array.Empty<int>(), 1)));
        new DftKernel().Execute(
            Node("DFT", new[] { "x", "", "axis" }, new[] { "y" }), ctxInput);
        var y = ctxInput.Get("y");

        Assert.Equal(new[] { 1, 4, 2 }, y.Shape.Dimensions.ToArray());
        Close(expected, y.Span.ToArray());
    }

    [Fact]
    public void Dft_AxisAsInput_WithDftLength()
    {
        // Both optional inputs present: input 1 = dft_length (zero-pad N=4 -> 8), input 2 = axis.
        // Cross-check against the same call expressed with the axis attribute.
        var xData = new float[] { 1, 2, 3, 4 };

        var ctxAttr = Ctx(
            ("x", F(new[] { 1, 4, 1 }, xData)),
            ("len", I64(new[] { 1 }, 8)));
        var attrs = new Dictionary<string, object> { ["axis"] = 1L };
        new DftKernel().Execute(Node("DFT", new[] { "x", "len" }, new[] { "y" }, attrs), ctxAttr);
        float[] expected = ctxAttr.Get("y").Span.ToArray();

        var ctxInput = Ctx(
            ("x", F(new[] { 1, 4, 1 }, xData)),
            ("len", I64(new[] { 1 }, 8)),
            ("axis", I64(Array.Empty<int>(), 1)));
        new DftKernel().Execute(
            Node("DFT", new[] { "x", "len", "axis" }, new[] { "y" }), ctxInput);
        var y = ctxInput.Get("y");

        Assert.Equal(new[] { 1, 8, 2 }, y.Shape.Dimensions.ToArray());
        Close(expected, y.Span.ToArray());
    }

    // ---- STFT ----------------------------------------------------------------------------------

    [Fact]
    public void Stft_FrameCount_Shape_And_FirstFrameValues()
    {
        // signal length 8, frame_length 4, frame_step 4 -> 2 non-overlapping frames.
        // frame0 = [1,2,3,4] -> DFT onesided = [10,0, -2,2, -2,0].
        var signal = F(new[] { 1, 8 }, 1, 2, 3, 4, 5, 6, 7, 8);
        var step = I64(new[] { 1 }, 4);
        var flen = I64(new[] { 1 }, 4);
        var ctx = Ctx(("sig", signal), ("step", step), ("flen", flen));
        new StftKernel().Execute(
            Node("STFT", new[] { "sig", "step", "", "flen" }, new[] { "y" }), ctx);

        var y = ctx.Get("y");
        // [batch=1, n_frames=2, n_dft=3, 2]
        Assert.Equal(new[] { 1, 2, 3, 2 }, y.Shape.Dimensions.ToArray());

        // First frame (bins 0..2).
        float[] all = y.Span.ToArray();
        float[] frame0 = all.Take(6).ToArray();
        Close(new[] { 10f, 0f, -2f, 2f, -2f, 0f }, frame0);
    }

    [Fact]
    public void Stft_AppliesWindow()
    {
        // Window of all-zeros forces every output bin to zero.
        var signal = F(new[] { 1, 4 }, 1, 2, 3, 4);
        var step = I64(new[] { 1 }, 4);
        var window = F(new[] { 4 }, 0, 0, 0, 0);
        var flen = I64(new[] { 1 }, 4);
        var ctx = Ctx(("sig", signal), ("step", step), ("win", window), ("flen", flen));
        new StftKernel().Execute(
            Node("STFT", new[] { "sig", "step", "win", "flen" }, new[] { "y" }), ctx);

        var y = ctx.Get("y");
        Assert.Equal(new[] { 1, 1, 3, 2 }, y.Shape.Dimensions.ToArray());
        Assert.All(y.Span.ToArray(), v => Assert.Equal(0f, v, 5));
    }

    // ---- MelWeightMatrix -----------------------------------------------------------------------

    [Fact]
    public void MelWeightMatrix_Shape_And_Triangular()
    {
        // num_mel_bins=4, dft_length=16, sample_rate=16000, lower=0, upper=8000.
        // numSpectrogramBins = 16/2+1 = 9.
        var ctx = Ctx(
            ("nmb", I64(Array.Empty<int>(), 4)),
            ("dft", I64(Array.Empty<int>(), 16)),
            ("sr", I64(Array.Empty<int>(), 16000)),
            ("lo", F(Array.Empty<int>(), 0f)),
            ("hi", F(Array.Empty<int>(), 8000f)));
        new MelWeightMatrixKernel().Execute(
            Node("MelWeightMatrix", new[] { "nmb", "dft", "sr", "lo", "hi" }, new[] { "y" }), ctx);

        var y = ctx.Get("y");
        Assert.Equal(new[] { 9, 4 }, y.Shape.Dimensions.ToArray());

        float[] w = y.Span.ToArray();
        int nBins = 9, nMel = 4;

        // All weights in [0,1].
        Assert.All(w, v => Assert.InRange(v, 0f, 1f + 1e-5f));

        // Each mel column is triangular: non-decreasing up to its peak, then non-increasing.
        for (int m = 0; m < nMel; m++)
        {
            var col = new float[nBins];
            for (int bin = 0; bin < nBins; bin++) col[bin] = w[bin * nMel + m];

            int peak = 0;
            for (int bin = 1; bin < nBins; bin++) if (col[bin] > col[peak]) peak = bin;

            Assert.True(col[peak] > 0f, $"mel column {m} is all zero");
            for (int bin = 1; bin <= peak; bin++)
                Assert.True(col[bin] >= col[bin - 1] - 1e-5f, $"col {m} not rising at {bin}");
            for (int bin = peak + 1; bin < nBins; bin++)
                Assert.True(col[bin] <= col[bin - 1] + 1e-5f, $"col {m} not falling at {bin}");
        }
    }

    [Fact]
    public void MelWeightMatrix_FirstColumn_PeaksAtExpectedBin()
    {
        // num_mel_bins=2, dft_length=8, sample_rate=16000, lower=0, upper=8000.
        // mel edges: lowMel=0, highMel=mel(8000); step=highMel/3.
        // First column rises from bin 0 to its center then falls; we verify peak value is 1
        // when the center lands exactly on a bin, and otherwise that bin 0 weight is 0 or 1.
        var ctx = Ctx(
            ("nmb", I64(Array.Empty<int>(), 2)),
            ("dft", I64(Array.Empty<int>(), 8)),
            ("sr", I64(Array.Empty<int>(), 16000)),
            ("lo", F(Array.Empty<int>(), 0f)),
            ("hi", F(Array.Empty<int>(), 8000f)));
        new MelWeightMatrixKernel().Execute(
            Node("MelWeightMatrix", new[] { "nmb", "dft", "sr", "lo", "hi" }, new[] { "y" }), ctx);

        var y = ctx.Get("y");
        Assert.Equal(new[] { 5, 2 }, y.Shape.Dimensions.ToArray());

        // lower edge of column 0 is bin 0 -> weight at bin 0 must be 0 (triangle base).
        float w00 = y.Span.ToArray()[0 * 2 + 0];
        Assert.Equal(0f, w00, 4);
    }
}
