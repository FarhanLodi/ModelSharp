using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ModelSharp.Engine;
using ModelSharp.Graph;
using ModelSharp.Tensors;

namespace ModelSharp.Gpu;

/// <summary>
/// GPU execution engine built on ILGPU. The kernels are written in C# and JIT-compiled to
/// CUDA / OpenCL on a real GPU — or to a CPU accelerator when no GPU is present, which is how
/// this runs (and is tested) on any machine. It plugs into the same <see cref="IExecutionEngine"/>
/// seam as the managed CPU engine.
///
/// Covers the elementwise ops (Add/Sub/Mul/Div, with NumPy-style broadcasting) and elementwise
/// activations (Relu/Sigmoid/Tanh/Gelu/LeakyRelu/Exp/Sqrt); the data-movement/reduction ops
/// Transpose, Softmax, ReduceSum and ReduceMean; LayerNormalization; the shape/index ops Gather,
/// Concat, Slice and Cast; and the two heavyweight tensor ops: batched MatMul (NumPy matmul
/// semantics) and Conv2D (NCHW, with stride/padding/dilation/groups/bias).
///
/// Dtypes: float32 tensors live on the device (the float-heavy compute path). Integer/bool tensors
/// (token ids, attention masks, shape/axes/index vectors) flow <em>host-side</em> as dtype-carrying
/// <see cref="Tensor"/> values — see <see cref="DeviceValue"/>. ILGPU's portable backends have no
/// native int64, and these tensors are consumed almost entirely by index math (Gather indices,
/// Slice starts/ends, Cast), so keeping them on the host is both correct and pragmatic; only Cast
/// crosses the host/device boundary (int→float uploads, float→int downloads). The float device
/// kernels live in <see cref="GpuKernels"/> and (for the per-element activations) here, with the
/// host-side stride/offset precomputation in this class.
///
/// The integer/boolean mask &amp; position-id prologue ops (Range, ConstantOfShape, Equal, Greater, Trilu,
/// ScatterND) are also dispatched: they are pure control-flow over the host-resident int/bool tensors (never
/// on the float compute path), so they compute host-side and store a dtype-carrying host value — exactly as
/// Shape/Constant do. With these added the whole distilgpt2 graph is GPU-dispatchable.
/// </summary>
public sealed class IlgpuEngine : IExecutionEngine
{
    private readonly ModelGraph _graph;
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _add;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _sub;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _mul;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _div;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _relu;

    // Extra elementwise activations (one-thread-per-element, mirroring the Relu style).
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _sigmoid;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _tanh;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _gelu;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _exp;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _sqrt;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, float> _leakyRelu;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _erf;

    // Data-movement / reduction ops whose strides are precomputed on the host.
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, int> _transpose;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _softmax;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, float> _reduce;

    // LayerNormalization: one thread per normalization group.
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>, int, float, int> _layerNorm;

    // Gather: one thread per output element, host-precomputed source offsets.
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _gather;

    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _broadcast;

    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, int, int, int> _matmul;

    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>, ConvParams> _conv;

    // B5 decoder ops: broadcasting Pow and Where (the latter with a float-encoded condition).
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _pow;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _where;

    /// <inheritdoc />
    public IReadOnlyList<TensorInfo> Inputs { get; }

    /// <inheritdoc />
    public IReadOnlyList<TensorInfo> Outputs { get; }

    /// <summary>The selected accelerator's name (e.g. a CUDA device, or "CPUAccelerator").</summary>
    public string AcceleratorName => _accelerator.Name;

    /// <summary>Whether a real GPU was selected (Cuda/OpenCL) versus the CPU fallback.</summary>
    public bool IsHardwareGpu => _accelerator.AcceleratorType != AcceleratorType.CPU;

    public IlgpuEngine(ModelGraph graph, bool preferCpu = false)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
        // EnableAlgorithms() registers ILGPU.Algorithms' intrinsic implementations (ExpF/TanhF/SqrtF/…)
        // for the *hardware* backends. Without it the PTX/CUDA backend has no intrinsic for these
        // transcendental MathF calls and throws at JIT time ("'ExpF' does not have an intrinsic
        // implementation for this backend"); the ILGPU CPU accelerator tolerated their absence by
        // falling back to .NET MathF, which is why the CPU-only tests never surfaced this.
        _context = Context.Create(builder => builder.Default().EnableAlgorithms());
        Device device = _context.GetPreferredDevice(preferCPU: preferCpu);
        _accelerator = device.CreateAccelerator(_context);

        _add = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(AddK);
        _sub = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SubK);
        _mul = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(MulK);
        _div = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(GpuKernels.DivK);
        _relu = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ReluK);

        _sigmoid = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SigmoidK);
        _tanh = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(TanhK);
        _gelu = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(GeluK);
        _exp = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ExpK);
        _sqrt = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SqrtK);
        _leakyRelu = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(LeakyReluK);
        _erf = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ErfK);

        _transpose = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, int>(GpuKernels.TransposeK);
        _softmax = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(GpuKernels.SoftmaxK);
        _reduce = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, float>(GpuKernels.ReduceK);

        _layerNorm = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, float, int>(GpuKernels.LayerNormK);
        _gather = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(GpuKernels.GatherK);

        _broadcast = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(GpuKernels.BroadcastBinaryK);
        _matmul = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, int, int, int>(GpuKernels.MatMulK);
        _conv = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ConvParams>(GpuKernels.Conv2DK);

        _pow = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(GpuKernels.PowK);
        _where = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(GpuKernels.WhereK);

        Inputs = graph.Inputs.Select(n => new TensorInfo(n, ElementType.Float32, Array.Empty<int>())).ToList();
        Outputs = graph.Outputs.Select(n => new TensorInfo(n, ElementType.Float32, Array.Empty<int>())).ToList();
    }

    // --- GPU kernels (compiled by ILGPU to PTX / OpenCL / CPU) ---
    private static void AddK(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> y) => y[i] = a[i] + b[i];
    private static void SubK(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> y) => y[i] = a[i] - b[i];
    private static void MulK(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> y) => y[i] = a[i] * b[i];
    private static void ReluK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = a[i] > 0f ? a[i] : 0f;

    // Extra activations. MathF.{Exp,Tanh,Sqrt} are recognized by ILGPU's frontend (ExpF/TanhF/SqrtF) and
    // lowered per backend, so these compile identically to the CPU kernels they mirror.
    private static void SigmoidK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = 1f / (1f + MathF.Exp(-a[i]));
    private static void TanhK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = MathF.Tanh(a[i]);
    private static void ExpK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = MathF.Exp(a[i]);
    private static void SqrtK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = MathF.Sqrt(a[i]);
    private static void GeluK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = 0.5f * a[i] * (1f + Erf(a[i] * 0.70710678f));
    private static void LeakyReluK(Index1D i, ArrayView<float> a, ArrayView<float> y, float alpha) => y[i] = a[i] >= 0f ? a[i] : alpha * a[i];
    private static void ErfK(Index1D i, ArrayView<float> a, ArrayView<float> y) => y[i] = Erf(a[i]);

    /// <summary>
    /// Device-side error function (Abramowitz &amp; Stegun 7.1.26), with the same constants as the CPU
    /// <c>MathHelpers.Erf</c> so <see cref="GeluK"/> matches the CPU GELU. Avoids <c>MathF.Sign</c>/
    /// <c>MathF.Abs</c> (computed inline) to stay on the always-available arithmetic intrinsics.
    /// </summary>
    private static float Erf(float x)
    {
        float sign = x < 0f ? -1f : 1f;
        float ax = x < 0f ? -x : x;
        float t = 1f / (1f + 0.3275911f * ax);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f)
                  * t * MathF.Exp(-ax * ax);
        return sign * y;
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
    {
        var values = new Dictionary<string, DeviceValue>();
        var temps = new List<MemoryBuffer>();   // auxiliary buffers (strides, batch offsets, dummy bias)
        try
        {
            foreach (KeyValuePair<string, Tensor> init in _graph.Initializers)
                values[init.Key] = Load(init.Value);
            foreach (string input in _graph.Inputs)
            {
                if (!feeds.TryGetValue(input, out NamedTensor? nt))
                    throw new ModelSharpException($"Missing feed for required input '{input}'.");
                values[input] = Load(nt.Tensor);
            }

            foreach (GraphNode node in _graph.Nodes)
            {
                switch (node.OpType)
                {
                    case "Add": RunBinary(GpuKernels.OpAdd, node, values, temps); break;
                    case "Sub": RunBinary(GpuKernels.OpSub, node, values, temps); break;
                    case "Mul": RunBinary(GpuKernels.OpMul, node, values, temps); break;
                    case "Div": RunBinary(GpuKernels.OpDiv, node, values, temps); break;
                    case "Relu": RunUnary(_relu, node, values); break;
                    case "Sigmoid": RunUnary(_sigmoid, node, values); break;
                    case "Tanh": RunUnary(_tanh, node, values); break;
                    case "Gelu": RunUnary(_gelu, node, values); break;
                    case "Exp": RunUnary(_exp, node, values); break;
                    case "Sqrt": RunUnary(_sqrt, node, values); break;
                    case "LeakyRelu": RunLeakyRelu(node, values); break;
                    case "Transpose": RunTranspose(node, values, temps); break;
                    case "Softmax": RunSoftmax(node, values); break;
                    case "ReduceSum": RunReduce(node, values, temps, mean: false); break;
                    case "ReduceMean": RunReduce(node, values, temps, mean: true); break;
                    case "LayerNormalization": RunLayerNorm(node, values, temps); break;
                    case "Gather": RunGather(node, values, temps); break;
                    case "Concat": RunConcat(node, values); break;
                    case "Slice": RunSlice(node, values, temps); break;
                    case "Cast": RunCast(node, values); break;
                    case "MatMul": RunMatMul(node, values, temps); break;
                    case "Gemm": RunGemm(node, values, temps); break;
                    case "Conv": RunConv(node, values, temps); break;
                    case "Erf": RunUnary(_erf, node, values); break;
                    case "Pow": RunPow(node, values, temps); break;
                    case "Where": RunWhere(node, values, temps); break;
                    case "Reshape": RunReshape(node, values); break;
                    case "Unsqueeze": RunUnsqueeze(node, values); break;
                    case "Squeeze": RunSqueeze(node, values); break;
                    case "Shape": RunShape(node, values); break;
                    case "Constant": RunConstant(node, values); break;
                    case "Expand": RunExpand(node, values, temps); break;
                    case "Split": RunSplit(node, values, temps); break;
                    // Integer/boolean control-flow ops that build the causal mask and position ids.
                    // These never touch the float compute path, so they run host-side (on the CPU-resident
                    // int/bool tensors) and store a dtype-carrying host DeviceValue — exactly as Shape/Constant
                    // already do for int outputs. Float results (rare) upload back to the device via Load.
                    case "Range": RunRange(node, values); break;
                    case "ConstantOfShape": RunConstantOfShape(node, values); break;
                    case "Equal": RunCompare(node, values, equal: true); break;
                    case "Greater": RunCompare(node, values, equal: false); break;
                    case "Trilu": RunTrilu(node, values); break;
                    case "ScatterND": RunScatterND(node, values); break;
                    default:
                        throw new UnsupportedOperatorException(node.OpType,
                            $"node '{node.Name}' — the GPU engine covers elementwise/activations, Softmax, " +
                            "Transpose, ReduceSum/ReduceMean, LayerNormalization, Gather/Concat/Slice/Cast, " +
                            "MatMul/Gemm, Conv, the decoder shape ops Reshape/Unsqueeze/Squeeze/Shape/" +
                            "Constant/Expand/Split plus Pow/Where/Erf, and the integer/mask prologue ops " +
                            "Range/ConstantOfShape/Equal/Greater/Trilu/ScatterND");
                }
            }
            _accelerator.Synchronize();

            var result = new Dictionary<string, NamedTensor>();
            foreach (string outName in _graph.Outputs)
                result[outName] = new NamedTensor(outName, values[outName].ToHost());
            return result;
        }
        finally
        {
            // Dispose each device buffer at most once. A few ops (e.g. an op whose output name reuses an input
            // name, or shared scalar buffers) can leave the same handle reachable from more than one dictionary
            // entry; disposing a buffer twice corrupts ILGPU's accelerator child-object tracking and surfaces as
            // a NullReferenceException when the accelerator itself is later disposed. Deduping by reference makes
            // cleanup robust even when Run unwinds mid-graph (e.g. on an unsupported op).
            var seen = new HashSet<MemoryBuffer>(ReferenceEqualityComparer.Instance);
            foreach (DeviceValue v in values.Values)
                if (v.FloatBuf is not null && seen.Add(v.FloatBuf)) v.FloatBuf.Dispose();
            foreach (MemoryBuffer buf in temps)
                if (seen.Add(buf)) buf.Dispose();
        }
    }

    // --- Dtype-aware value plumbing (B2) ---

    /// <summary>
    /// A tensor in flight through the engine. Float32 tensors live on the device (<see cref="FloatBuf"/>);
    /// integer/bool tensors (token ids, masks, shape/index vectors) are kept host-side as
    /// dtype-carrying <see cref="HostInt"/> tensors. Exactly one of the two is set.
    /// </summary>
    private readonly struct DeviceValue
    {
        public readonly MemoryBuffer1D<float, Stride1D.Dense>? FloatBuf;
        public readonly Tensor? HostInt;
        public readonly TensorShape Shape;

        private DeviceValue(MemoryBuffer1D<float, Stride1D.Dense>? f, Tensor? h, TensorShape shape)
        {
            FloatBuf = f; HostInt = h; Shape = shape;
        }

        public static DeviceValue Device(MemoryBuffer1D<float, Stride1D.Dense> buf, TensorShape shape)
            => new(buf, null, shape);

        public static DeviceValue Host(Tensor t) => new(null, t, t.Shape);

        public bool IsFloat => FloatBuf is not null;

        /// <summary>Materializes the value as a dtype-carrying host tensor (downloading floats if needed).</summary>
        public Tensor ToHost()
        {
            if (HostInt is not null) return HostInt;
            float[] raw = FloatBuf!.GetAsArray1D();
            long len = Shape.Length;
            // Index/empty fast-paths over-allocate the device buffer (min size 1); trim to the real length.
            if (raw.LongLength != len)
            {
                var trimmed = new float[len];
                Array.Copy(raw, trimmed, len);
                raw = trimmed;
            }
            return new Tensor<float>(Shape, raw);
        }
    }

    /// <summary>Loads an input/initializer: float32 onto the device, integer/bool kept host-side.</summary>
    private DeviceValue Load(Tensor t)
    {
        if (t.Dtype == ElementType.Float32)
        {
            // A zero-length float tensor (e.g. an empty past_key_values at prefill, shape [b,h,0,d]) must
            // not hit ILGPU's Allocate1D(emptyArray): a 0-length buffer's View can't be safely SubView'd or
            // CopyTo'd later. Back an empty tensor with a sentinel 1-element buffer but carry the true
            // (length-0) shape, matching the over-allocate/trim convention DeviceValue.ToHost documents.
            if (t.Shape.Length == 0)
                return DeviceValue.Device(AllocFloat(0), t.Shape);
            return DeviceValue.Device(_accelerator.Allocate1D(t.AsFloat().Span.ToArray()), t.Shape);
        }
        return DeviceValue.Host(t);
    }

    /// <summary>
    /// Allocates a device float buffer of <paramref name="len"/> elements, but never of length 0: a
    /// zero-length tensor is backed by a sentinel 1-element buffer (the carried <see cref="TensorShape"/> still
    /// records length 0, so <see cref="DeviceValue.ToHost"/> trims back to empty). This keeps the empty
    /// <c>past_key_values</c> at prefill — and any other zero-length intermediate — from producing a buffer whose
    /// <c>View</c>/<c>SubView</c>/<c>CopyTo</c> would fault, while contributing nothing to Concat/Gather/MatMul.
    /// </summary>
    private MemoryBuffer1D<float, Stride1D.Dense> AllocFloat(long len)
        => _accelerator.Allocate1D<float>(len <= 0 ? 1 : len);

    /// <summary>The device float buffer for <paramref name="name"/>; throws if the value is host-side integer.</summary>
    private MemoryBuffer1D<float, Stride1D.Dense> FloatBuf(Dictionary<string, DeviceValue> values, string name)
    {
        DeviceValue v = values[name];
        if (v.FloatBuf is null)
            throw new ModelSharpException($"GPU op expected a float32 tensor for '{name}' but it is {v.HostInt!.Dtype}.");
        return v.FloatBuf;
    }

    /// <summary>Reads an integer/index input as int64 regardless of where it lives (host int, or a float buffer).</summary>
    private long[] ReadInts(Dictionary<string, DeviceValue> values, string name)
    {
        DeviceValue v = values[name];
        if (v.HostInt is not null) return TensorAsInts(v.HostInt);
        return Array.ConvertAll(v.FloatBuf!.GetAsArray1D(), f => (long)MathF.Round(f));
    }

    /// <summary>Reads any integer-ish host tensor as int64 (mirrors the CPU <c>TensorInts.Read</c>).</summary>
    private static long[] TensorAsInts(Tensor t)
    {
        switch (t.Dtype)
        {
            case ElementType.Int64:
            {
                ReadOnlySpan<long> s = t.AsInt64().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i];
                return r;
            }
            case ElementType.Int32:
            {
                ReadOnlySpan<int> s = t.AsInt32().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i];
                return r;
            }
            case ElementType.Boolean:
            {
                ReadOnlySpan<bool> s = t.AsBool().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = s[i] ? 1 : 0;
                return r;
            }
            default:
            {
                ReadOnlySpan<float> s = t.AsFloat().Span;
                var r = new long[s.Length];
                for (int i = 0; i < s.Length; i++) r[i] = (long)MathF.Round(s[i]);
                return r;
            }
        }
    }

    // --- Op implementations ---

    /// <summary>
    /// Runs a binary elementwise op (Add/Sub/Mul/Div), selected by <paramref name="op"/>. Equal shapes
    /// take the contiguous fast path; differing shapes use NumPy-style broadcasting where the per-axis
    /// broadcast strides are precomputed on the host and passed to the kernel as device arrays.
    /// </summary>
    private void RunBinary(
        int op,
        GraphNode node,
        Dictionary<string, DeviceValue> values,
        List<MemoryBuffer> temps)
    {
        DeviceValue av = values[node.Inputs[0]];
        DeviceValue bv = values[node.Inputs[1]];

        // Integer index math (shape arithmetic in the mask/position prologue: Add/Sub/Mul/Div on int tensors)
        // runs host-side on the int/bool-resident tensors, mirroring the CPU elementwise kernels. Only the
        // float compute path touches the device.
        if (!av.IsFloat || !bv.IsFloat)
        {
            RunBinaryHost(op, node, av, bv, values);
            return;
        }

        MemoryBuffer1D<float, Stride1D.Dense> a = av.FloatBuf!;
        MemoryBuffer1D<float, Stride1D.Dense> b = bv.FloatBuf!;
        TensorShape sa = av.Shape;
        TensorShape sb = bv.Shape;

        if (sa.Equals(sb))
        {
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> fast = op switch
            {
                GpuKernels.OpAdd => _add,
                GpuKernels.OpSub => _sub,
                GpuKernels.OpMul => _mul,
                _ => _div,
            };
            MemoryBuffer1D<float, Stride1D.Dense> yEqual = AllocFloat(sa.Length);
            if (sa.Length > 0) fast((int)sa.Length, a.View, b.View, yEqual.View);
            values[node.Outputs[0]] = DeviceValue.Device(yEqual, sa);
            return;
        }

        int[] outd = BroadcastShape(sa.Dimensions, sb.Dimensions);
        int rank = outd.Length;
        int[] outStrides = Strides(outd);
        int[] strideA = BroadcastStrides(sa.Dimensions, rank);
        int[] strideB = BroadcastStrides(sb.Dimensions, rank);
        int n = 1;
        foreach (int d in outd) n *= d;

        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(n);
        ArrayView<int> vOut = Upload(outStrides, temps);
        ArrayView<int> vA = Upload(strideA, temps);
        ArrayView<int> vB = Upload(strideB, temps);

        if (n > 0) _broadcast(n, a.View, b.View, y.View, vOut, vA, vB, rank, op);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outd));
    }

    /// <summary>
    /// Host-side integer elementwise Add/Sub/Mul/Div with NumPy broadcasting, used for the shape/index math in
    /// distilgpt2's mask &amp; position-id prologue (where both operands are int64/int32 host tensors). Mirrors the
    /// CPU <c>BroadcastBinaryKernel</c> exactly — including integer (truncating) division — and preserves the
    /// integer dtype so a downstream Gather/Slice/Reshape that needs int indices still receives integers.
    /// </summary>
    private void RunBinaryHost(int op, GraphNode node, DeviceValue av, DeviceValue bv, Dictionary<string, DeviceValue> values)
    {
        Tensor a = av.ToHost();
        Tensor b = bv.ToHost();
        Tensor outT = a.Dtype switch
        {
            ElementType.Int64 => BinaryHost<long>(a.AsInt64().Span, b.AsInt64().Span, av.Shape, bv.Shape, op),
            ElementType.Int32 => BinaryHost<int>(a.AsInt32().Span, b.AsInt32().Span, av.Shape, bv.Shape, op),
            _ => BinaryHost<float>(a.AsFloat().Span, b.AsFloat().Span, av.Shape, bv.Shape, op),
        };
        values[node.Outputs[0]] = DeviceValue.Host(outT);
    }

    private static Tensor<T> BinaryHost<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, TensorShape sa, TensorShape sb, int op)
        where T : unmanaged
    {
        Func<T, T, T> f = BinaryFunc<T>(op);
        if (sa.Equals(sb))
        {
            var yEqual = new T[(int)sa.Length];
            for (int i = 0; i < yEqual.Length; i++) yEqual[i] = f(a[i], b[i]);
            return new Tensor<T>(sa, yEqual);
        }

        int[] outd = BroadcastShape(sa.Dimensions, sb.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = BroadcastStrides(sa.Dimensions, rank);
        int[] strideB = BroadcastStrides(sb.Dimensions, rank);

        var buf = new T[(int)outShape.Length];
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < buf.Length; idx++)
        {
            buf[idx] = f(a[aOff], b[bOff]);
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                aOff += strideA[ax];
                bOff += strideB[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                aOff -= strideA[ax] * outd[ax];
                bOff -= strideB[ax] * outd[ax];
            }
        }
        return new Tensor<T>(outShape, buf);
    }

    /// <summary>The integer/float scalar op for a host-side binary, matching the CPU kernels (int Div truncates).</summary>
    private static Func<T, T, T> BinaryFunc<T>(int op) where T : unmanaged
    {
        if (typeof(T) == typeof(long))
        {
            Func<long, long, long> g = op switch
            {
                GpuKernels.OpAdd => (x, y) => x + y,
                GpuKernels.OpSub => (x, y) => x - y,
                GpuKernels.OpMul => (x, y) => x * y,
                _ => (x, y) => x / y,
            };
            return (Func<T, T, T>)(object)g;
        }
        if (typeof(T) == typeof(int))
        {
            Func<int, int, int> g = op switch
            {
                GpuKernels.OpAdd => (x, y) => x + y,
                GpuKernels.OpSub => (x, y) => x - y,
                GpuKernels.OpMul => (x, y) => x * y,
                _ => (x, y) => x / y,
            };
            return (Func<T, T, T>)(object)g;
        }
        Func<float, float, float> h = op switch
        {
            GpuKernels.OpAdd => (x, y) => x + y,
            GpuKernels.OpSub => (x, y) => x - y,
            GpuKernels.OpMul => (x, y) => x * y,
            _ => (x, y) => x / y,
        };
        return (Func<T, T, T>)(object)h;
    }

    /// <summary>Runs a unary elementwise op (currently Relu) over the input buffer.</summary>
    private void RunUnary(
        Action<Index1D, ArrayView<float>, ArrayView<float>> kernel,
        GraphNode node,
        Dictionary<string, DeviceValue> values)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        TensorShape uShape = values[node.Inputs[0]].Shape;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(uShape.Length);
        if (uShape.Length > 0) kernel((int)uShape.Length, a.View, y.View);
        values[node.Outputs[0]] = DeviceValue.Device(y, uShape);
    }

    /// <summary>Leaky ReLU (<c>x</c> if x ≥ 0 else <c>α·x</c>); <c>alpha</c> defaults to 0.01. Mirrors the CPU <c>LeakyReluKernel</c>.</summary>
    private void RunLeakyRelu(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        TensorShape lShape = values[node.Inputs[0]].Shape;
        float alpha = AttrFloat(node, "alpha", 0.01f);
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(lShape.Length);
        if (lShape.Length > 0) _leakyRelu((int)lShape.Length, a.View, y.View, alpha);
        values[node.Outputs[0]] = DeviceValue.Device(y, lShape);
    }

    /// <summary>
    /// General N-D axis permutation (ONNX <c>Transpose</c>; <c>perm</c> defaults to a full reverse). The
    /// output row-major strides and the per-output-axis source strides (<c>inStrides[perm[i]]</c>) are
    /// precomputed on the host and uploaded; the kernel is a single gather. Mirrors the CPU <c>TransposeKernel</c>.
    /// </summary>
    private void RunTranspose(
        GraphNode node,
        Dictionary<string, DeviceValue> values,
        List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> x = FloatBuf(values, node.Inputs[0]);
        TensorShape xS = values[node.Inputs[0]].Shape;
        ReadOnlySpan<int> dims = xS.Dimensions;
        int rank = dims.Length;

        int[] perm = node.Attributes.ContainsKey("perm")
            ? AttrInts(node, "perm", Array.Empty<int>())
            : ReverseAxes(rank);

        int[] inStrides = Strides(dims);
        var outDims = new int[rank];
        var srcStrides = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            outDims[i] = dims[perm[i]];
            srcStrides[i] = inStrides[perm[i]];
        }
        int[] outStrides = Strides(outDims);

        int n = 1;
        foreach (int d in outDims) n *= d;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(n);
        ArrayView<int> vOut = Upload(outStrides, temps);
        ArrayView<int> vSrc = Upload(srcStrides, temps);

        if (n > 0) _transpose(n, x.View, y.View, vOut, vSrc, rank);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outDims));
    }

    /// <summary>
    /// Softmax along an axis (default the last), max-subtraction stabilized. The axis is reduced into
    /// (outer, inner) extents on the host and one GPU thread handles each row. Mirrors the CPU <c>SoftmaxKernel</c>.
    /// </summary>
    private void RunSoftmax(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        MemoryBuffer1D<float, Stride1D.Dense> x = FloatBuf(values, node.Inputs[0]);
        TensorShape xS = values[node.Inputs[0]].Shape;
        ReadOnlySpan<int> dims = xS.Dimensions;
        int rank = dims.Length;
        long axisAttr = AttrInt(node, "axis", -1);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);

        int axisSize = dims[axis];
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        int total = 1;
        foreach (int d in dims) total *= d;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(total);

        if (total > 0) _softmax(outer * inner, x.View, y.View, axisSize, inner);
        values[node.Outputs[0]] = DeviceValue.Device(y, xS);
    }

    /// <summary>
    /// ReduceSum / ReduceMean over the given <c>axes</c> with <c>keepdims</c> (axes from the attribute, the
    /// optional second input, or all axes when absent; <c>noop_with_empty_axes</c> honored). For each output
    /// element the host precomputes its zero-reduced-coordinate input offset and the reduced-axis strides;
    /// the kernel folds in the same order as the CPU reduce kernels. <paramref name="mean"/> divides by the
    /// reduced-element count.
    /// </summary>
    private void RunReduce(
        GraphNode node,
        Dictionary<string, DeviceValue> values,
        List<MemoryBuffer> temps,
        bool mean)
    {
        MemoryBuffer1D<float, Stride1D.Dense> x = FloatBuf(values, node.Inputs[0]);
        TensorShape xS = values[node.Inputs[0]].Shape;
        ReadOnlySpan<int> inDims = xS.Dimensions;
        int rank = inDims.Length;

        int[]? axes = null;
        if (node.Attributes.ContainsKey("axes"))
            axes = AttrInts(node, "axes", Array.Empty<int>());
        else if (node.Inputs.Count > 1 && node.Inputs[1].Length > 0)
            axes = Array.ConvertAll(ReadInts(values, node.Inputs[1]), v => (int)v);

        bool keepdims = AttrInt(node, "keepdims", 1) != 0;
        bool noopEmpty = AttrInt(node, "noop_with_empty_axes", 0) != 0;

        if ((axes is null || axes.Length == 0) && noopEmpty)
        {
            // Identity: copy the input through unchanged, device-to-device (no host round-trip).
            MemoryBuffer1D<float, Stride1D.Dense> copy = AllocFloat(xS.Length);
            if (xS.Length > 0) x.View.CopyTo(_accelerator.DefaultStream, copy.View);
            values[node.Outputs[0]] = DeviceValue.Device(copy, xS);
            return;
        }

        var reduced = new bool[rank];
        if (axes is null || axes.Length == 0)
            for (int i = 0; i < rank; i++) reduced[i] = true;
        else
            foreach (int ax in axes) reduced[ax < 0 ? ax + rank : ax] = true;

        int[] inStrides = Strides(inDims);
        var keepDims = new int[rank];
        for (int i = 0; i < rank; i++) keepDims[i] = reduced[i] ? 1 : inDims[i];

        int outLen = 1;
        foreach (int d in keepDims) outLen *= d;
        int redCount = 1;
        for (int i = 0; i < rank; i++) if (reduced[i]) redCount *= inDims[i];

        // Reduced axes (ascending order) and their input strides; row-major strides over the reduced
        // index space drive the kernel's fold order (last reduced axis fastest = CPU order).
        var redDimsList = new List<int>();
        var redStridesList = new List<int>();
        for (int i = 0; i < rank; i++)
            if (reduced[i]) { redDimsList.Add(inDims[i]); redStridesList.Add(inStrides[i]); }
        int numRed = redDimsList.Count;
        int[] redStrides = redStridesList.ToArray();
        int[] redOutStrides = Strides(redDimsList.ToArray());

        // For each output element (row-major over keepDims) the input offset with all reduced coords = 0.
        var outBase = new int[outLen];
        var coord = new int[rank];
        for (int o = 0; o < outLen; o++)
        {
            int off = 0;
            for (int ax = 0; ax < rank; ax++) off += coord[ax] * inStrides[ax];
            outBase[o] = off;
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < keepDims[ax]) break; coord[ax] = 0; }
        }

        float divisor = mean ? redCount : 1f;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(outLen);
        ArrayView<int> vBase = Upload(outBase, temps);
        ArrayView<int> vRedOut = Upload(redOutStrides, temps);
        ArrayView<int> vRedStr = Upload(redStrides, temps);

        if (outLen > 0) _reduce(outLen, x.View, y.View, vBase, vRedOut, vRedStr, numRed, redCount, divisor);

        int[] finalDims;
        if (keepdims)
        {
            finalDims = keepDims;
        }
        else
        {
            var list = new List<int>();
            for (int i = 0; i < rank; i++) if (!reduced[i]) list.Add(inDims[i]);
            finalDims = list.ToArray();
        }
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(finalDims));
    }

    /// <summary>
    /// Layer normalization over the axes from <c>axis</c> to the end:
    /// <c>y = (x − mean)/√(var + ε) · scale + bias</c>. One GPU thread per normalization group; the per-group
    /// mean/variance and the same fold order as the CPU <c>LayerNormalizationKernel</c> make the result match.
    /// Real device kernel (<see cref="GpuKernels.LayerNormK"/>).
    /// </summary>
    private void RunLayerNorm(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> x = FloatBuf(values, node.Inputs[0]);
        MemoryBuffer1D<float, Stride1D.Dense> scale = FloatBuf(values, node.Inputs[1]);
        TensorShape xS = values[node.Inputs[0]].Shape;
        ReadOnlySpan<int> dims = xS.Dimensions;
        int rank = dims.Length;

        int axis = (int)AttrInt(node, "axis", -1);
        if (axis < 0) axis += rank;
        float eps = AttrFloat(node, "epsilon", 1e-5f);

        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int norm = 1; for (int i = axis; i < rank; i++) norm *= dims[i];

        bool hasBias = node.Inputs.Count > 2 && node.Inputs[2].Length > 0;
        ArrayView<float> biasView;
        if (hasBias)
        {
            biasView = FloatBuf(values, node.Inputs[2]).View;
        }
        else
        {
            // Unused by the kernel (hasBias=0), but sized to norm so any indexing stays in bounds.
            MemoryBuffer1D<float, Stride1D.Dense> dummy = _accelerator.Allocate1D<float>(norm);
            temps.Add(dummy);
            biasView = dummy.View;
        }

        int total = outer * norm;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(total);
        if (outer > 0 && norm > 0) _layerNorm(outer, x.View, scale.View, biasView, y.View, norm, eps, hasBias ? 1 : 0);
        values[node.Outputs[0]] = DeviceValue.Device(y, xS);
    }

    /// <summary>
    /// ONNX <c>Gather</c>: gathers slices of float <c>data</c> along <c>axis</c> by integer indices.
    /// The (outer × q × inner) source offset for each output element is precomputed on the host (matching
    /// the CPU <c>GatherKernel</c> layout) and uploaded; the kernel is a single copy. Indices may be int64,
    /// int32 or float — they are read host-side via <see cref="ReadInts"/>.
    /// </summary>
    private void RunGather(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        DeviceValue dataV = values[node.Inputs[0]];
        TensorShape dS = dataV.Shape;
        ReadOnlySpan<int> dd = dS.Dimensions;
        int rank = dd.Length;

        long[] idx = ReadInts(values, node.Inputs[1]);
        int[] idims = values[node.Inputs[1]].Shape.Dimensions.ToArray();

        int axis = (int)AttrInt(node, "axis", 0);
        if (axis < 0) axis += rank;
        int axisDim = dd[axis];
        int outer = 1; for (int i = 0; i < axis; i++) outer *= dd[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dd[i];
        int q = idx.Length;

        var outDims = new int[axis + idims.Length + (rank - axis - 1)];
        int p = 0;
        for (int i = 0; i < axis; i++) outDims[p++] = dd[i];
        for (int i = 0; i < idims.Length; i++) outDims[p++] = idims[i];
        for (int i = axis + 1; i < rank; i++) outDims[p++] = dd[i];

        // Per-output-element source offset (element units). Mirrors the CPU GatherKernel's
        // (outer × q) rows of <inner> contiguous elements, expanded to one offset per element.
        int rows = outer * q;
        int total = rows * inner;
        var srcOff = new int[total == 0 ? 1 : total];
        int pos = 0;
        for (int o = 0; o < outer; o++)
        for (int k = 0; k < q; k++)
        {
            int gi = (int)idx[k];
            if (gi < 0) gi += axisDim;
            int srcBase = (o * axisDim + gi) * inner;
            for (int e = 0; e < inner; e++) srcOff[pos++] = srcBase + e;
        }

        // Integer/bool data (e.g. gathering elements out of a Shape vector or token ids) stays host-side.
        if (!dataV.IsFloat)
        {
            values[node.Outputs[0]] = DeviceValue.Host(HostGatherFlat(dataV.HostInt!, outDims, srcOff, total));
            return;
        }

        MemoryBuffer1D<float, Stride1D.Dense> data = dataV.FloatBuf!;
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(total == 0 ? 1 : total);
        ArrayView<int> vOff = Upload(srcOff, temps);
        if (total > 0) _gather(total, data.View, y.View, vOff);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outDims));
    }

    /// <summary>
    /// ONNX <c>Concat</c> along <c>axis</c>. Dtype-generic, assembled host-side via the same contiguous
    /// block-copy layout as the CPU <c>ConcatKernel</c>: float inputs are downloaded, concatenated, and the
    /// result re-uploaded to the device; integer/bool inputs stay host-side. Block copies are pure index math
    /// (no float arithmetic), so the result is bit-identical to the CPU kernel.
    /// </summary>
    private void RunConcat(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        var first = values[node.Inputs[0]];
        int rank = first.Shape.Rank;
        long axisAttr = AttrInt(node, "axis", 0);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);

        var tensors = new Tensor[node.Inputs.Count];
        for (int i = 0; i < tensors.Length; i++) tensors[i] = values[node.Inputs[i]].ToHost();

        if (first.IsFloat)
        {
            Tensor<float> y = HostConcat<float>(tensors, axis);
            values[node.Outputs[0]] = Load(y);
            return;
        }

        Tensor outT = tensors[0].Dtype switch
        {
            ElementType.Int64 => HostConcat<long>(tensors, axis),
            ElementType.Int32 => HostConcat<int>(tensors, axis),
            ElementType.Boolean => HostConcat<bool>(tensors, axis),
            _ => HostConcat<float>(tensors, axis),
        };
        values[node.Outputs[0]] = DeviceValue.Host(outT);
    }

    private static Tensor<T> HostConcat<T>(Tensor[] tensors, int axis) where T : unmanaged
    {
        ReadOnlySpan<int> dims0 = tensors[0].Shape.Dimensions;
        int rank = dims0.Length;
        int outAxis = 0;
        foreach (Tensor t in tensors) outAxis += t.Shape[axis];
        int[] outDims = dims0.ToArray();
        outDims[axis] = outAxis;
        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> ys = y.Span;

        int outer = 1; for (int i = 0; i < axis; i++) outer *= outDims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= outDims[i];

        int axisOffset = 0;
        foreach (Tensor tb in tensors)
        {
            var t = (Tensor<T>)tb;
            int ta = t.Shape[axis];
            Span<T> ts = t.Span;
            int block = ta * inner;
            for (int o = 0; o < outer; o++)
            {
                int src = o * block;
                int dst = (o * outAxis + axisOffset) * inner;
                ts.Slice(src, block).CopyTo(ys.Slice(dst, block));
            }
            axisOffset += ta;
        }
        return y;
    }

    /// <summary>
    /// ONNX <c>Slice</c> (opset 10+): slices <c>data</c> along the axes given by the
    /// <c>starts</c>/<c>ends</c>/optional <c>axes</c>/optional <c>steps</c> integer inputs (or opset-1
    /// attributes), with NumPy semantics (negative indices, clamping, negative steps). Float data is sliced
    /// on the device (host-precomputed per-output source offset, one thread per element via the Gather
    /// kernel); integer/bool data is sliced host-side. Mirrors the CPU <c>SliceKernel</c>.
    /// </summary>
    private void RunSlice(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        DeviceValue dataV = values[node.Inputs[0]];
        ReadOnlySpan<int> inDims = dataV.Shape.Dimensions;
        int rank = inDims.Length;
        int[] inStrides = Strides(inDims);

        long[] starts = ReadAxisInts(node, values, 1, "starts");
        long[] ends = ReadAxisInts(node, values, 2, "ends");
        long[] axes = ReadAxisInts(node, values, 3, "axes");
        long[] steps = ReadAxisInts(node, values, 4, "steps");

        int k = starts.Length;
        if (ends.Length != k)
            throw new ModelSharpException($"Slice 'starts' ({k}) and 'ends' ({ends.Length}) must have the same length.");
        if (axes.Length == 0)
        {
            axes = new long[k];
            for (int i = 0; i < k; i++) axes[i] = i;
        }

        var effStart = new int[rank];
        var effStep = new int[rank];
        var outDims = inDims.ToArray();
        for (int i = 0; i < rank; i++) effStep[i] = 1;

        for (int i = 0; i < k; i++)
        {
            int axis = (int)axes[i];
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank)
                throw new ModelSharpException($"Slice axis {axes[i]} is out of range for rank {rank}.");

            long step = steps.Length > i ? steps[i] : 1;
            if (step == 0) throw new ModelSharpException("Slice 'steps' values cannot be 0.");

            NormalizeSliceAxis(starts[i], ends[i], step, inDims[axis], out int s, out int count);
            effStart[axis] = s;
            effStep[axis] = (int)step;
            outDims[axis] = count;
        }

        int n = 1; foreach (int d in outDims) n *= d;

        // Precompute, for each output element, the flat source offset (element units).
        var srcOff = new int[n == 0 ? 1 : n];
        var coord = new int[rank];
        for (int idx = 0; idx < n; idx++)
        {
            int src = 0;
            for (int i = 0; i < rank; i++) src += (effStart[i] + coord[i] * effStep[i]) * inStrides[i];
            srcOff[idx] = src;
            for (int ax = rank - 1; ax >= 0; ax--) { if (++coord[ax] < outDims[ax]) break; coord[ax] = 0; }
        }

        if (!dataV.IsFloat)
        {
            values[node.Outputs[0]] = DeviceValue.Host(HostGatherFlat(dataV.HostInt!, outDims, srcOff, n));
            return;
        }

        // Float path: gather on the device using the precomputed per-element source offsets.
        MemoryBuffer1D<float, Stride1D.Dense> data = dataV.FloatBuf!;
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(n == 0 ? 1 : n);
        ArrayView<int> vOff = Upload(srcOff, temps);
        if (n > 0) _gather(n, data.View, y.View, vOff);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outDims));
    }

    private static Tensor HostGatherFlat(Tensor data, int[] outDims, int[] srcOff, int n)
    {
        return data.Dtype switch
        {
            ElementType.Int64 => HostGatherFlat<long>(data.AsInt64(), outDims, srcOff, n),
            ElementType.Int32 => HostGatherFlat<int>(data.AsInt32(), outDims, srcOff, n),
            ElementType.Boolean => HostGatherFlat<bool>(data.AsBool(), outDims, srcOff, n),
            _ => HostGatherFlat<float>(data.AsFloat(), outDims, srcOff, n),
        };
    }

    private static Tensor<T> HostGatherFlat<T>(Tensor<T> x, int[] outDims, int[] srcOff, int n) where T : unmanaged
    {
        var y = new Tensor<T>(new TensorShape(outDims));
        Span<T> xs = x.Span, ys = y.Span;
        for (int i = 0; i < n; i++) ys[i] = xs[srcOff[i]];
        return y;
    }

    /// <summary>
    /// ONNX <c>Cast</c> to the dtype named by the <c>to</c> attribute. Float→float is a copy; int/bool→float
    /// uploads the converted values to the device (so downstream float ops can use the result); float→int and
    /// any int/bool target produce a host-side integer/bool tensor. Truncation toward zero and bool-from-nonzero
    /// match the CPU <c>CastKernel</c>.
    /// </summary>
    private void RunCast(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        DeviceValue src = values[node.Inputs[0]];
        ElementType target = MapOnnxType(AttrInt(node, "to", 0));

        // Read source as double (the CPU kernel's common intermediate).
        double[] s = ReadAsDoubles(src);
        int n = s.Length;
        TensorShape shape = src.Shape;

        if (target == ElementType.Float32)
        {
            var buf = new float[n];
            for (int i = 0; i < n; i++) buf[i] = (float)s[i];
            values[node.Outputs[0]] = DeviceValue.Device(_accelerator.Allocate1D(buf), shape);
            return;
        }

        Tensor result;
        switch (target)
        {
            case ElementType.Float64:
                result = new Tensor<double>(shape, s);
                break;
            case ElementType.Int32:
            {
                var buf = new int[n];
                for (int i = 0; i < n; i++) buf[i] = (int)s[i]; // truncate toward zero
                result = new Tensor<int>(shape, buf);
                break;
            }
            case ElementType.Int64:
            {
                var buf = new long[n];
                for (int i = 0; i < n; i++) buf[i] = (long)s[i]; // truncate toward zero
                result = new Tensor<long>(shape, buf);
                break;
            }
            case ElementType.Boolean:
            {
                var buf = new bool[n];
                for (int i = 0; i < n; i++) buf[i] = s[i] != 0d;
                result = new Tensor<bool>(shape, buf);
                break;
            }
            default:
                throw new ModelSharpException($"Cast target dtype {target} is not supported.");
        }
        values[node.Outputs[0]] = DeviceValue.Host(result);
    }

    /// <summary>Reads a value (device float or host int/bool) as doubles, mirroring the CPU Cast intermediate.</summary>
    private static double[] ReadAsDoubles(DeviceValue v)
    {
        if (v.IsFloat)
        {
            float[] f = v.FloatBuf!.GetAsArray1D();
            var r = new double[f.Length];
            for (int i = 0; i < f.Length; i++) r[i] = f[i];
            return r;
        }

        Tensor t = v.HostInt!;
        int n = checked((int)t.Length);
        var rr = new double[n];
        switch (t.Dtype)
        {
            case ElementType.Int64:
            {
                ReadOnlySpan<long> ss = t.AsInt64().Span;
                for (int i = 0; i < n; i++) rr[i] = ss[i];
                break;
            }
            case ElementType.Int32:
            {
                ReadOnlySpan<int> ss = t.AsInt32().Span;
                for (int i = 0; i < n; i++) rr[i] = ss[i];
                break;
            }
            case ElementType.Boolean:
            {
                ReadOnlySpan<bool> ss = t.AsBool().Span;
                for (int i = 0; i < n; i++) rr[i] = ss[i] ? 1d : 0d;
                break;
            }
            default:
                throw new ModelSharpException($"Cast source dtype {t.Dtype} is not supported.");
        }
        return rr;
    }

    private static ElementType MapOnnxType(long to) => to switch
    {
        1 => ElementType.Float32,
        6 => ElementType.Int32,
        7 => ElementType.Int64,
        9 => ElementType.Boolean,
        11 => ElementType.Float64,
        _ => throw new ModelSharpException($"Cast 'to' data_type {to} is not supported."),
    };

    /// <summary>
    /// Reads the integer axis-spec at <paramref name="inputIndex"/> (opset 10+ input form), falling back to
    /// the opset-1 attribute <paramref name="attrName"/>, or an empty array when neither is present.
    /// </summary>
    private long[] ReadAxisInts(GraphNode node, Dictionary<string, DeviceValue> values, int inputIndex, string attrName)
    {
        if (node.Inputs.Count > inputIndex && !string.IsNullOrEmpty(node.Inputs[inputIndex]))
            return ReadInts(values, node.Inputs[inputIndex]);

        int[] attr = AttrInts(node, attrName, Array.Empty<int>());
        if (attr.Length == 0) return Array.Empty<long>();
        var r = new long[attr.Length];
        for (int i = 0; i < attr.Length; i++) r[i] = attr[i];
        return r;
    }

    /// <summary>
    /// NumPy/ONNX slice normalization for one axis: resolves negative indices, clamps to the
    /// valid range for the step's sign, and yields the first source index plus the element count.
    /// Mirrors the CPU <c>SliceKernel.NormalizeAxis</c>.
    /// </summary>
    private static void NormalizeSliceAxis(long start, long end, long step, int dim, out int outStart, out int outCount)
    {
        long lower, upper;
        if (step < 0) { lower = -1; upper = dim - 1; }
        else { lower = 0; upper = dim; }

        long s = start;
        if (s < 0) { s += dim; if (s < lower) s = lower; }
        else if (s > upper) s = upper;

        long e = end;
        if (e < 0) { e += dim; if (e < lower) e = lower; }
        else if (e > upper) e = upper;

        long count;
        if (step > 0) count = e > s ? (e - s + step - 1) / step : 0;
        else count = s > e ? (s - e + (-step) - 1) / (-step) : 0;

        outStart = (int)s;
        outCount = (int)count;
    }

    /// <summary>The default Transpose permutation: axes fully reversed.</summary>
    private static int[] ReverseAxes(int rank)
    {
        var perm = new int[rank];
        for (int i = 0; i < rank; i++) perm[i] = rank - 1 - i;
        return perm;
    }

    /// <summary>
    /// Matrix multiply with full NumPy semantics: 2-D, stacked, and batched n-D × n-D with broadcasting
    /// of the leading (batch) dimensions; 1-D operands are promoted per NumPy rules. The batch is flattened
    /// on the host and each flattened batch index's operand base offsets are precomputed and uploaded, so the
    /// kernel is a simple one-thread-per-output-element dot product. Mirrors the CPU <c>MatMulKernel</c>.
    /// </summary>
    private void RunMatMul(
        GraphNode node,
        Dictionary<string, DeviceValue> values,
        List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        MemoryBuffer1D<float, Stride1D.Dense> b = FloatBuf(values, node.Inputs[1]);
        TensorShape saS = values[node.Inputs[0]].Shape;
        TensorShape sbS = values[node.Inputs[1]].Shape;

        bool aWas1D = saS.Rank == 1;
        bool bWas1D = sbS.Rank == 1;

        int[] adims = aWas1D ? new[] { 1, saS[0] } : saS.Dimensions.ToArray();
        int[] bdims = bWas1D ? new[] { sbS[0], 1 } : sbS.Dimensions.ToArray();

        int M = adims[adims.Length - 2], K = adims[adims.Length - 1];
        int Kb = bdims[bdims.Length - 2], N = bdims[bdims.Length - 1];
        if (K != Kb) throw new ModelSharpException($"MatMul inner dimensions disagree: {K} vs {Kb}.");

        int[] aBatch = adims[..^2];
        int[] bBatch = bdims[..^2];
        int[] outBatch = BroadcastShape(aBatch, bBatch);
        int batchRank = outBatch.Length;
        int[] aBS = BroadcastStrides(aBatch, batchRank);   // strides in matrix units
        int[] bBS = BroadcastStrides(bBatch, batchRank);

        int aMat = M * K, bMat = K * N, oMat = M * N;
        int totalBatch = 1;
        foreach (int d in outBatch) totalBatch *= d;

        // Precompute per-batch operand base offsets (element units), walking the batch coordinates.
        var aOffArr = new int[totalBatch];
        var bOffArr = new int[totalBatch];
        var coord = new int[batchRank];
        int aOff = 0, bOff = 0;
        for (int bi = 0; bi < totalBatch; bi++)
        {
            aOffArr[bi] = aOff * aMat;
            bOffArr[bi] = bOff * bMat;
            for (int ax = batchRank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                aOff += aBS[ax];
                bOff += bBS[ax];
                if (coord[ax] < outBatch[ax]) break;
                coord[ax] = 0;
                aOff -= aBS[ax] * outBatch[ax];
                bOff -= bBS[ax] * outBatch[ax];
            }
        }

        var outDims = new List<int>(outBatch);
        if (!aWas1D) outDims.Add(M);
        if (!bWas1D) outDims.Add(N);

        int total = totalBatch * oMat;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(total);
        ArrayView<int> vAOff = Upload(aOffArr.Length == 0 ? new[] { 0 } : aOffArr, temps);
        ArrayView<int> vBOff = Upload(bOffArr.Length == 0 ? new[] { 0 } : bOffArr, temps);

        // K==0 (a contraction over an empty past) still produces a well-defined all-zero output: the kernel
        // runs (total>0) and its inner k-loop executes zero times, so every output element is written as 0.
        if (total > 0) _matmul(total, a.View, b.View, y.View, vAOff, vBOff, M, K, N);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outDims.ToArray()));
    }

    /// <summary>
    /// 2-D convolution (NCHW) with strides, pads/auto_pad, dilations, group and optional bias. A direct
    /// convolution: one GPU thread per output element. Mirrors the CPU <c>ConvKernel</c>.
    /// </summary>
    private void RunConv(
        GraphNode node,
        Dictionary<string, DeviceValue> values,
        List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> x = FloatBuf(values, node.Inputs[0]);
        MemoryBuffer1D<float, Stride1D.Dense> w = FloatBuf(values, node.Inputs[1]);
        TensorShape xS = values[node.Inputs[0]].Shape;
        TensorShape wS = values[node.Inputs[1]].Shape;
        ReadOnlySpan<int> xd = xS.Dimensions;
        ReadOnlySpan<int> wd = wS.Dimensions;
        if (xd.Length != 4 || wd.Length != 4)
            throw new ModelSharpException("Conv currently supports 4-D NCHW tensors only.");

        int N = xd[0], C = xd[1], H = xd[2], W = xd[3];
        int cout = wd[0], cinPerGroup = wd[1], kH = wd[2], kW = wd[3];

        int group = (int)AttrInt(node, "group", 1);
        int[] strides = AttrInts(node, "strides", new[] { 1, 1 });
        int[] dil = AttrInts(node, "dilations", new[] { 1, 1 });
        int sH = strides[0], sW = strides[1], dH = dil[0], dW = dil[1];

        int padTop, padLeft, padBottom, padRight;
        string autoPad = AttrStr(node, "auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            bool upper = autoPad == "SAME_UPPER";
            SamePad(H, kH, sH, dH, upper, out padTop, out padBottom);
            SamePad(W, kW, sW, dW, upper, out padLeft, out padRight);
        }
        else if (autoPad == "VALID")
        {
            padTop = padLeft = padBottom = padRight = 0;
        }
        else
        {
            int[] p = AttrInts(node, "pads", new[] { 0, 0, 0, 0 });
            padTop = p[0]; padLeft = p[1]; padBottom = p[2]; padRight = p[3];
        }

        int outH = (H + padTop + padBottom - (dH * (kH - 1) + 1)) / sH + 1;
        int outW = (W + padLeft + padRight - (dW * (kW - 1) + 1)) / sW + 1;
        int outPerGroup = cout / group;

        bool hasBias = node.Inputs.Count > 2 && node.Inputs[2].Length > 0;
        ArrayView<float> biasView;
        if (hasBias)
        {
            biasView = FloatBuf(values, node.Inputs[2]).View;
        }
        else
        {
            // Unused by the kernel (HasBias=0), but sized to cout so any indexing stays in bounds.
            MemoryBuffer1D<float, Stride1D.Dense> dummy = _accelerator.Allocate1D<float>(cout);
            temps.Add(dummy);
            biasView = dummy.View;
        }

        int total = N * cout * outH * outW;
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(total);

        var prm = new ConvParams(N, C, H, W, cout, cinPerGroup, kH, kW, outH, outW,
            outPerGroup, sH, sW, dH, dW, padTop, padLeft, hasBias ? 1 : 0);

        _conv(total, x.View, w.View, biasView, y.View, prm);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(N, cout, outH, outW));
    }

    // --- B5 decoder shape / compute ops ---

    /// <summary>
    /// Re-emits <paramref name="src"/> under a new shape without changing its element order. Float values are
    /// copied device→device into a fresh buffer (so the new dictionary entry owns its own buffer, avoiding the
    /// double-dispose that aliasing would cause in <c>Run</c>'s finally); integer/bool values stay host-side
    /// via <see cref="Tensor.WithShape"/>. Used by Reshape/Unsqueeze/Squeeze.
    /// </summary>
    private DeviceValue Reshaped(DeviceValue src, int[] outDims)
    {
        var shape = new TensorShape(outDims);
        if (src.IsFloat)
        {
            // Copy only the real (possibly-empty) length; the source buffer may be a size-1 sentinel that
            // backs a length-0 tensor, so copy through SubView guarded on the carried shape length.
            MemoryBuffer1D<float, Stride1D.Dense> copy = AllocFloat(shape.Length);
            if (shape.Length > 0)
                src.FloatBuf!.View.CopyTo(_accelerator.DefaultStream, copy.View);
            return DeviceValue.Device(copy, shape);
        }
        return DeviceValue.Host(src.HostInt!.WithShape(shape));
    }

    /// <summary>ONNX <c>Reshape</c> (shape from the 2nd input; 0 = keep, -1 = infer). Mirrors the CPU <c>ReshapeKernel</c>.</summary>
    private void RunReshape(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        DeviceValue src = values[node.Inputs[0]];
        ReadOnlySpan<int> inDims = src.Shape.Dimensions;
        long total = src.Shape.Length;
        long[] sp = ReadInts(values, node.Inputs[1]);

        int n = sp.Length;
        var dims = new int[n];
        int inferIdx = -1;
        long known = 1;
        for (int i = 0; i < n; i++)
        {
            long d = sp[i];
            if (d == -1) inferIdx = i;
            else if (d == 0) { dims[i] = inDims[i]; known *= dims[i]; }
            else { dims[i] = (int)d; known *= d; }
        }
        if (inferIdx >= 0) dims[inferIdx] = (int)(total / known);
        values[node.Outputs[0]] = Reshaped(src, dims);
    }

    /// <summary>ONNX <c>Unsqueeze</c> (axes from attribute or 2nd input). Mirrors the CPU <c>UnsqueezeKernel</c>.</summary>
    private void RunUnsqueeze(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        DeviceValue src = values[node.Inputs[0]];
        int[]? axes = node.Attributes.ContainsKey("axes") ? AttrInts(node, "axes", Array.Empty<int>()) : null;
        if (axes is null && node.Inputs.Count > 1)
            axes = Array.ConvertAll(ReadInts(values, node.Inputs[1]), v => (int)v);
        axes ??= Array.Empty<int>();

        ReadOnlySpan<int> inDims = src.Shape.Dimensions;
        int outRank = inDims.Length + axes.Length;
        var inserted = new HashSet<int>();
        foreach (int ax in axes) inserted.Add(ax < 0 ? ax + outRank : ax);

        var outDims = new int[outRank];
        int j = 0;
        for (int i = 0; i < outRank; i++) outDims[i] = inserted.Contains(i) ? 1 : inDims[j++];
        values[node.Outputs[0]] = Reshaped(src, outDims);
    }

    /// <summary>ONNX <c>Squeeze</c> (specified axes, or all size-1 axes). Mirrors the CPU <c>SqueezeKernel</c>.</summary>
    private void RunSqueeze(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        DeviceValue src = values[node.Inputs[0]];
        ReadOnlySpan<int> inDims = src.Shape.Dimensions;
        int rank = inDims.Length;

        int[]? axes = node.Attributes.ContainsKey("axes") ? AttrInts(node, "axes", Array.Empty<int>()) : null;
        if (axes is null && node.Inputs.Count > 1)
            axes = Array.ConvertAll(ReadInts(values, node.Inputs[1]), v => (int)v);

        var kept = new List<int>();
        if (axes is null)
        {
            for (int i = 0; i < rank; i++) if (inDims[i] != 1) kept.Add(inDims[i]);
        }
        else
        {
            var remove = new HashSet<int>();
            foreach (int ax in axes) remove.Add(ax < 0 ? ax + rank : ax);
            for (int i = 0; i < rank; i++) if (!remove.Contains(i)) kept.Add(inDims[i]);
        }
        values[node.Outputs[0]] = Reshaped(src, kept.ToArray());
    }

    /// <summary>
    /// ONNX <c>Shape</c> (opset 15+ start/end honored): emits the input's dimensions as a 1-D int64 host tensor.
    /// Pure shape metadata, so this never touches the device buffer. Mirrors the CPU <c>ShapeKernel</c>.
    /// </summary>
    private void RunShape(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        TensorShape shape = values[node.Inputs[0]].Shape;
        int rank = shape.Rank;
        int start = ClampShapeAxis(AttrInt(node, "start", 0), rank);
        int end = ClampShapeAxis(AttrInt(node, "end", rank), rank);
        int count = end - start;
        if (count < 0) count = 0;

        ReadOnlySpan<int> dims = shape.Dimensions;
        var buf = new long[count];
        for (int i = 0; i < count; i++) buf[i] = dims[start + i];
        values[node.Outputs[0]] = DeviceValue.Host(new Tensor<long>(new TensorShape(count), buf));
    }

    private static int ClampShapeAxis(long value, int rank)
    {
        if (value < 0) value += rank;
        if (value < 0) value = 0;
        if (value > rank) value = rank;
        return (int)value;
    }

    /// <summary>
    /// ONNX <c>Constant</c>: materializes the node's <c>value</c>/<c>value_int(s)</c>/<c>value_float(s)</c>
    /// attribute. Float tensors are uploaded to the device; integer tensors stay host-side. Mirrors the CPU
    /// <c>ConstantKernel</c>.
    /// </summary>
    private void RunConstant(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        string outName = node.Outputs[0];
        if (node.Attributes.TryGetValue("value", out object? tv) && tv is Tensor t)
        {
            values[outName] = Load(t);
            return;
        }
        if (node.Attributes.TryGetValue("value_ints", out object? viv) && viv is long[] ints)
        {
            values[outName] = DeviceValue.Host(new Tensor<long>(new TensorShape(ints.Length), ints));
            return;
        }
        if (node.Attributes.TryGetValue("value_floats", out object? vfv) && vfv is float[] floats)
        {
            values[outName] = DeviceValue.Device(_accelerator.Allocate1D(floats), new TensorShape(floats.Length));
            return;
        }
        if (node.Attributes.TryGetValue("value_int", out object? iv))
        {
            values[outName] = DeviceValue.Host(new Tensor<long>(new TensorShape(Array.Empty<int>()), new[] { Convert.ToInt64(iv) }));
            return;
        }
        if (node.Attributes.TryGetValue("value_float", out object? fv))
        {
            values[outName] = DeviceValue.Device(_accelerator.Allocate1D(new[] { Convert.ToSingle(fv) }), new TensorShape(Array.Empty<int>()));
            return;
        }
        throw new ModelSharpException(
            $"Constant node '{node.Name}' has no supported value attribute (value/value_int(s)/value_float(s)).");
    }

    /// <summary>
    /// ONNX <c>Pow</c>: elementwise base^exp with NumPy broadcasting (float path on the device). Integer/bool
    /// operands are not expected here; falls through to broadcasting on the float device buffers. Mirrors the
    /// CPU <c>PowKernel</c>.
    /// </summary>
    private void RunPow(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        MemoryBuffer1D<float, Stride1D.Dense> b = FloatBuf(values, node.Inputs[1]);
        TensorShape sa = values[node.Inputs[0]].Shape;
        TensorShape sb = values[node.Inputs[1]].Shape;

        int[] outd = BroadcastShape(sa.Dimensions, sb.Dimensions);
        int rank = outd.Length;
        int n = 1; foreach (int d in outd) n *= d;
        int[] outStrides = Strides(outd);
        int[] strideA = BroadcastStrides(sa.Dimensions, rank);
        int[] strideB = BroadcastStrides(sb.Dimensions, rank);

        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(n == 0 ? 1 : n);
        ArrayView<int> vOut = Upload(outStrides, temps);
        ArrayView<int> vA = Upload(strideA, temps);
        ArrayView<int> vB = Upload(strideB, temps);
        if (n > 0) _pow(n, a.View, b.View, y.View, vOut, vA, vB, rank);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outd));
    }

    /// <summary>
    /// ONNX <c>Where</c>: <c>cond ? X : Y</c> with NumPy broadcasting over all three inputs, float value path on
    /// the device. The boolean condition (carried host-side) is uploaded as a 0/1 float buffer. Mirrors the CPU
    /// <c>WhereKernel</c> float path.
    /// </summary>
    private void RunWhere(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        DeviceValue condV = values[node.Inputs[0]];
        DeviceValue xV = values[node.Inputs[1]];
        DeviceValue yV = values[node.Inputs[2]];

        // Integer/bool value path: do it host-side (rare; keeps dtype correctness).
        if (!xV.IsFloat || !yV.IsFloat)
        {
            RunWhereHost(node, values, condV, xV, yV);
            return;
        }

        // Condition → float 0/1 device buffer (it arrives as a host bool/int tensor, or already a float buffer).
        MemoryBuffer1D<float, Stride1D.Dense> condBuf;
        bool condTemp = false;
        if (condV.IsFloat)
        {
            condBuf = condV.FloatBuf!;
        }
        else
        {
            long[] ci = TensorAsInts(condV.HostInt!);
            var cf = new float[ci.Length];
            for (int i = 0; i < ci.Length; i++) cf[i] = ci[i] != 0 ? 1f : 0f;
            condBuf = _accelerator.Allocate1D(cf);
            temps.Add(condBuf);
            condTemp = true;
        }
        _ = condTemp;

        TensorShape sc = condV.Shape, sx = xV.Shape, sy = yV.Shape;
        int[] outd = BroadcastShape(BroadcastShape(sc.Dimensions, sx.Dimensions), sy.Dimensions);
        int rank = outd.Length;
        int n = 1; foreach (int d in outd) n *= d;
        int[] outStrides = Strides(outd);
        int[] sCs = BroadcastStrides(sc.Dimensions, rank);
        int[] sXs = BroadcastStrides(sx.Dimensions, rank);
        int[] sYs = BroadcastStrides(sy.Dimensions, rank);

        MemoryBuffer1D<float, Stride1D.Dense> outBuf = _accelerator.Allocate1D<float>(n == 0 ? 1 : n);
        ArrayView<int> vOut = Upload(outStrides, temps);
        ArrayView<int> vC = Upload(sCs, temps);
        ArrayView<int> vX = Upload(sXs, temps);
        ArrayView<int> vY = Upload(sYs, temps);
        if (n > 0) _where(n, condBuf.View, xV.FloatBuf!.View, yV.FloatBuf!.View, outBuf.View, vOut, vC, vX, vY, rank);
        values[node.Outputs[0]] = DeviceValue.Device(outBuf, new TensorShape(outd));
    }

    private void RunWhereHost(GraphNode node, Dictionary<string, DeviceValue> values,
        DeviceValue condV, DeviceValue xV, DeviceValue yV)
    {
        long[] cond = condV.IsFloat
            ? Array.ConvertAll(condV.FloatBuf!.GetAsArray1D(), f => f != 0f ? 1L : 0L)
            : TensorAsInts(condV.HostInt!);
        Tensor xt = xV.ToHost(), yt = yV.ToHost();

        TensorShape sc = condV.Shape, sx = xV.Shape, sy = yV.Shape;
        int[] outd = BroadcastShape(BroadcastShape(sc.Dimensions, sx.Dimensions), sy.Dimensions);
        int rank = outd.Length;
        int n = 1; foreach (int d in outd) n *= d;
        int[] sCs = BroadcastStrides(sc.Dimensions, rank);
        int[] sXs = BroadcastStrides(sx.Dimensions, rank);
        int[] sYs = BroadcastStrides(sy.Dimensions, rank);

        Tensor outT = xt.Dtype switch
        {
            ElementType.Int64 => WhereHost<long>(cond, xt.AsInt64().Span, yt.AsInt64().Span, outd, n, rank, sCs, sXs, sYs),
            ElementType.Int32 => WhereHost<int>(cond, xt.AsInt32().Span, yt.AsInt32().Span, outd, n, rank, sCs, sXs, sYs),
            ElementType.Boolean => WhereHost<bool>(cond, xt.AsBool().Span, yt.AsBool().Span, outd, n, rank, sCs, sXs, sYs),
            _ => WhereHost<float>(cond, xt.AsFloat().Span, yt.AsFloat().Span, outd, n, rank, sCs, sXs, sYs),
        };
        values[node.Outputs[0]] = DeviceValue.Host(outT);
    }

    private static Tensor<T> WhereHost<T>(long[] cond, Span<T> x, Span<T> y, int[] outd, int n, int rank,
        int[] sC, int[] sX, int[] sY) where T : unmanaged
    {
        var buf = new T[n == 0 ? 0 : n];
        var coord = new int[rank];
        int cOff = 0, xOff = 0, yOff = 0;
        for (int idx = 0; idx < n; idx++)
        {
            buf[idx] = cond[cOff] != 0 ? x[xOff] : y[yOff];
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                cOff += sC[ax]; xOff += sX[ax]; yOff += sY[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                cOff -= sC[ax] * outd[ax]; xOff -= sX[ax] * outd[ax]; yOff -= sY[ax] * outd[ax];
            }
        }
        return new Tensor<T>(new TensorShape(outd), buf);
    }

    /// <summary>
    /// ONNX <c>Expand</c>: broadcast to (broadcast of input shape and) the requested 1-D shape input. Float path
    /// runs the broadcast as a device gather (host-precomputed per-output source offsets); integer/bool stays
    /// host-side. Mirrors the CPU <c>ExpandKernel</c>.
    /// </summary>
    private void RunExpand(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        DeviceValue src = values[node.Inputs[0]];
        long[] requested = ReadInts(values, node.Inputs[1]);
        var reqDims = new int[requested.Length];
        for (int i = 0; i < requested.Length; i++) reqDims[i] = (int)requested[i];

        int[] outd = BroadcastShape(src.Shape.Dimensions, reqDims);
        int rank = outd.Length;
        int n = 1; foreach (int d in outd) n *= d;
        int[] stride = BroadcastStrides(src.Shape.Dimensions, rank);

        // Per-output-element source offset (element units).
        var srcOff = new int[n == 0 ? 1 : n];
        var coord = new int[rank];
        int off = 0;
        for (int idx = 0; idx < n; idx++)
        {
            srcOff[idx] = off;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                off += stride[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                off -= stride[ax] * outd[ax];
            }
        }

        if (!src.IsFloat)
        {
            values[node.Outputs[0]] = DeviceValue.Host(HostGatherFlat(src.HostInt!, outd, srcOff, n));
            return;
        }
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(n == 0 ? 1 : n);
        ArrayView<int> vOff = Upload(srcOff, temps);
        if (n > 0) _gather(n, src.FloatBuf!.View, y.View, vOff);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outd));
    }

    /// <summary>
    /// ONNX <c>Split</c> along <c>axis</c> into the node's outputs (sizes from the 2nd input, the <c>split</c>
    /// attribute, or even division by <c>num_outputs</c>/output count). Float chunks are sliced on the device
    /// (per-output-element source offset via the Gather kernel); integer/bool chunks host-side. Mirrors the CPU
    /// <c>SplitKernel</c>.
    /// </summary>
    private void RunSplit(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        DeviceValue src = values[node.Inputs[0]];
        ReadOnlySpan<int> dims = src.Shape.Dimensions;
        int rank = dims.Length;
        long axisAttr = AttrInt(node, "axis", 0);
        int axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        int axisDim = dims[axis];

        int[] sizes;
        int[]? splitAttr = node.Attributes.ContainsKey("split") ? AttrInts(node, "split", Array.Empty<int>()) : null;
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
        {
            long[] sp = ReadInts(values, node.Inputs[1]);
            sizes = new int[sp.Length];
            for (int i = 0; i < sp.Length; i++) sizes[i] = (int)sp[i];
        }
        else if (splitAttr is not null && splitAttr.Length > 0)
        {
            sizes = splitAttr;
        }
        else
        {
            int no = (int)AttrInt(node, "num_outputs", node.Outputs.Count);
            if (no <= 0) no = node.Outputs.Count;
            int chunk = (axisDim + no - 1) / no;
            sizes = new int[no];
            int rem = axisDim;
            for (int i = 0; i < no; i++) { int c = Math.Min(chunk, Math.Max(0, rem)); sizes[i] = c; rem -= c; }
        }

        int outer = 1; for (int i = 0; i < axis; i++) outer *= dims[i];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= dims[i];

        int offset = 0;
        for (int p = 0; p < sizes.Length; p++)
        {
            int sz = sizes[p];
            int[] outDims = dims.ToArray();
            outDims[axis] = sz;
            int block = sz * inner;
            int total = outer * block;

            if (!src.IsFloat)
            {
                // Host-side block copy mirroring the CPU SplitKernel.
                var srcOffH = new int[total == 0 ? 1 : total];
                int posH = 0;
                for (int o = 0; o < outer; o++)
                {
                    int srcBase = (o * axisDim + offset) * inner;
                    for (int e = 0; e < block; e++) srcOffH[posH++] = srcBase + e;
                }
                values[node.Outputs[p]] = DeviceValue.Host(HostGatherFlat(src.HostInt!, outDims, srcOffH, total));
                offset += sz;
                continue;
            }

            var srcOff = new int[total == 0 ? 1 : total];
            int pos = 0;
            for (int o = 0; o < outer; o++)
            {
                int srcBase = (o * axisDim + offset) * inner;
                for (int e = 0; e < block; e++) srcOff[pos++] = srcBase + e;
            }
            MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(total == 0 ? 1 : total);
            ArrayView<int> vOff = Upload(srcOff, temps);
            if (total > 0) _gather(total, src.FloatBuf!.View, y.View, vOff);
            values[node.Outputs[p]] = DeviceValue.Device(y, new TensorShape(outDims));
            offset += sz;
        }
    }

    // --- Integer / boolean mask & position-id prologue ops (host-side) ---
    //
    // distilgpt2's causal-mask / position-id prologue uses these six integer/boolean control-flow ops
    // (Range/ConstantOfShape/Equal/Greater/Trilu/ScatterND). They are never on the float compute path, so —
    // consistent with the engine's design where int/bool tensors flow host-side — they compute on the
    // CPU-resident tensors and store a dtype-carrying host value (via Load, which routes float→device,
    // int/bool→host). Each mirrors the ONNX semantics of the corresponding CPU kernel so GPU-vs-CPU parity is exact.

    /// <summary>
    /// ONNX <c>Range</c>: the 1-D sequence <c>start, start+delta, …</c> stopping before <c>limit</c>; element
    /// count = max(ceil((limit−start)/delta), 0). The output dtype follows the (scalar) input dtype
    /// (Int64/Int32/Float32). Mirrors the CPU <c>RangeKernel</c>.
    /// </summary>
    private void RunRange(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        DeviceValue startV = values[node.Inputs[0]];
        DeviceValue limitV = values[node.Inputs[1]];
        DeviceValue deltaV = values[node.Inputs[2]];

        // Float inputs live on the device; int inputs host-side. Read each scalar accordingly.
        if (startV.IsFloat)
        {
            float s = startV.FloatBuf!.GetAsArray1D()[0];
            float l = limitV.FloatBuf!.GetAsArray1D()[0];
            float d = deltaV.FloatBuf!.GetAsArray1D()[0];
            int n = RangeCount(s, l, d);
            var buf = new float[n];
            for (int i = 0; i < n; i++) buf[i] = s + i * d;
            values[node.Outputs[0]] = Load(new Tensor<float>(new TensorShape(n), buf));
            return;
        }

        Tensor st = startV.HostInt!;
        if (st.Dtype == ElementType.Int32)
        {
            int s = st.AsInt32().Span[0], l = limitV.HostInt!.AsInt32().Span[0], d = deltaV.HostInt!.AsInt32().Span[0];
            int n = RangeCount(s, l, d);
            var buf = new int[n];
            for (int i = 0; i < n; i++) buf[i] = s + i * d;
            values[node.Outputs[0]] = DeviceValue.Host(new Tensor<int>(new TensorShape(n), buf));
            return;
        }

        // Default / Int64.
        long[] sa = TensorAsInts(st), la = TensorAsInts(limitV.HostInt!), da = TensorAsInts(deltaV.HostInt!);
        long ls = sa[0], ll = la[0], ld = da[0];
        int cnt = RangeCount(ls, ll, ld);
        var lbuf = new long[cnt];
        for (int i = 0; i < cnt; i++) lbuf[i] = ls + (long)i * ld;
        values[node.Outputs[0]] = DeviceValue.Host(new Tensor<long>(new TensorShape(cnt), lbuf));
    }

    /// <summary>Range element count = max(ceil((limit−start)/delta), 0). Mirrors the CPU <c>RangeKernel.Count</c>.</summary>
    private static int RangeCount(double start, double limit, double delta)
    {
        if (delta == 0d) throw new ModelSharpException("Range 'delta' cannot be 0.");
        double c = Math.Ceiling((limit - start) / delta);
        return c <= 0d ? 0 : (int)c;
    }

    /// <summary>
    /// ONNX <c>ConstantOfShape</c>: a tensor of the shape given by the 1-D integer input, filled with the scalar
    /// <c>value</c> attribute (default a single float32 zero). Output dtype follows <c>value</c>'s dtype; float
    /// fills upload to the device, int/bool stay host-side. Mirrors the CPU <c>ConstantOfShapeKernel</c>.
    /// </summary>
    private void RunConstantOfShape(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        long[] shapeVals = ReadInts(values, node.Inputs[0]);
        var dims = new int[shapeVals.Length];
        for (int i = 0; i < shapeVals.Length; i++) dims[i] = checked((int)shapeVals[i]);
        var shape = new TensorShape(dims);
        int count = checked((int)shape.Length);

        Tensor? value = node.Attributes.TryGetValue("value", out object? v) ? v as Tensor : null;
        ElementType dtype = value?.Dtype ?? ElementType.Float32;

        switch (dtype)
        {
            case ElementType.Int64:
            {
                ReadOnlySpan<long> vs = value!.AsInt64().Span;
                var buf = new long[count];
                Array.Fill(buf, vs.Length > 0 ? vs[0] : 0L);
                values[node.Outputs[0]] = DeviceValue.Host(new Tensor<long>(shape, buf));
                break;
            }
            case ElementType.Int32:
            {
                ReadOnlySpan<int> vs = value!.AsInt32().Span;
                var buf = new int[count];
                Array.Fill(buf, vs.Length > 0 ? vs[0] : 0);
                values[node.Outputs[0]] = DeviceValue.Host(new Tensor<int>(shape, buf));
                break;
            }
            case ElementType.Boolean:
            {
                ReadOnlySpan<bool> vs = value!.AsBool().Span;
                var buf = new bool[count];
                Array.Fill(buf, vs.Length > 0 && vs[0]);
                values[node.Outputs[0]] = DeviceValue.Host(new Tensor<bool>(shape, buf));
                break;
            }
            case ElementType.Float32:
            {
                float fill = 0f;
                if (value is not null)
                {
                    ReadOnlySpan<float> vs = value.AsFloat().Span;
                    if (vs.Length > 0) fill = vs[0];
                }
                var buf = new float[count];
                Array.Fill(buf, fill);
                values[node.Outputs[0]] = Load(new Tensor<float>(shape, buf)); // float → device
                break;
            }
            default:
                throw new ModelSharpException($"ConstantOfShape: unsupported 'value' dtype {dtype}.");
        }
    }

    /// <summary>
    /// ONNX <c>Equal</c>/<c>Greater</c>: elementwise same-dtype comparison with NumPy broadcasting, producing a
    /// Boolean tensor (always host-side). Both operands are read at their native dtype; float NaN comparisons
    /// follow IEEE semantics. Mirrors the CPU <c>EqualKernel</c>/<c>GreaterKernel</c>.
    /// </summary>
    private void RunCompare(GraphNode node, Dictionary<string, DeviceValue> values, bool equal)
    {
        Tensor a = values[node.Inputs[0]].ToHost();
        Tensor b = values[node.Inputs[1]].ToHost();

        Tensor<bool> y;
        switch (a.Dtype)
        {
            case ElementType.Int64:
            {
                Func<long, long, bool> cmp = equal ? (x, z) => x == z : (x, z) => x > z;
                y = CompareHost(a.AsInt64().Span, b.AsInt64().Span, a.Shape, b.Shape, cmp);
                break;
            }
            case ElementType.Int32:
            {
                Func<int, int, bool> cmp = equal ? (x, z) => x == z : (x, z) => x > z;
                y = CompareHost(a.AsInt32().Span, b.AsInt32().Span, a.Shape, b.Shape, cmp);
                break;
            }
            case ElementType.Boolean:
            {
                // Greater is undefined for bool in ONNX; only Equal is expected here.
                Func<bool, bool, bool> cmp = equal ? (x, z) => x == z : (x, z) => (x ? 1 : 0) > (z ? 1 : 0);
                y = CompareHost(a.AsBool().Span, b.AsBool().Span, a.Shape, b.Shape, cmp);
                break;
            }
            default:
            {
                Func<float, float, bool> cmp = equal ? (x, z) => x == z : (x, z) => x > z;
                y = CompareHost(a.AsFloat().Span, b.AsFloat().Span, a.Shape, b.Shape, cmp);
                break;
            }
        }
        values[node.Outputs[0]] = DeviceValue.Host(y);
    }

    private static Tensor<bool> CompareHost<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, TensorShape sa, TensorShape sb,
        Func<T, T, bool> cmp) where T : unmanaged
    {
        if (sa.Equals(sb))
        {
            var yEqual = new bool[(int)sa.Length];
            for (int i = 0; i < yEqual.Length; i++) yEqual[i] = cmp(a[i], b[i]);
            return new Tensor<bool>(sa, yEqual);
        }

        int[] outd = BroadcastShape(sa.Dimensions, sb.Dimensions);
        var outShape = new TensorShape(outd);
        int rank = outd.Length;
        int[] strideA = BroadcastStrides(sa.Dimensions, rank);
        int[] strideB = BroadcastStrides(sb.Dimensions, rank);

        var buf = new bool[(int)outShape.Length];
        var coord = new int[rank];
        int aOff = 0, bOff = 0;
        for (int idx = 0; idx < buf.Length; idx++)
        {
            buf[idx] = cmp(a[aOff], b[bOff]);
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                coord[ax]++;
                aOff += strideA[ax];
                bOff += strideB[ax];
                if (coord[ax] < outd[ax]) break;
                coord[ax] = 0;
                aOff -= strideA[ax] * outd[ax];
                bOff -= strideB[ax] * outd[ax];
            }
        }
        return new Tensor<bool>(outShape, buf);
    }

    /// <summary>
    /// ONNX <c>Trilu</c>: keeps the upper (<c>upper=1</c>, default) or lower triangle of the trailing 2-D
    /// matrices, zeroing the rest; the optional scalar <c>k</c> input shifts the diagonal. Batched over leading
    /// dims, dtype-preserving (float re-uploads to the device, int/bool host-side). Mirrors the CPU <c>TriluKernel</c>.
    /// </summary>
    private void RunTrilu(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        DeviceValue dataV = values[node.Inputs[0]];
        Tensor data = dataV.ToHost();
        bool upper = AttrInt(node, "upper", 1) != 0;
        long k = 0;
        if (node.Inputs.Count > 1 && !string.IsNullOrEmpty(node.Inputs[1]))
            k = ReadInts(values, node.Inputs[1])[0];

        Tensor outT = data.Dtype switch
        {
            ElementType.Int64 => TriluHost(data.AsInt64(), k, upper),
            ElementType.Int32 => TriluHost(data.AsInt32(), k, upper),
            ElementType.Boolean => TriluHost(data.AsBool(), k, upper),
            _ => TriluHost(data.AsFloat(), k, upper),
        };

        // Preserve the device/host placement of the original dtype (float → device).
        values[node.Outputs[0]] = dataV.IsFloat ? Load(outT) : DeviceValue.Host(outT);
    }

    private static Tensor<T> TriluHost<T>(Tensor<T> x, long k, bool upper) where T : unmanaged
    {
        ReadOnlySpan<int> dims = x.Shape.Dimensions;
        int rank = dims.Length;
        if (rank < 2) throw new ModelSharpException("Trilu requires a tensor of rank >= 2.");
        int rows = dims[rank - 2], cols = dims[rank - 1];
        int batch = 1; for (int i = 0; i < rank - 2; i++) batch *= dims[i];

        var y = new Tensor<T>(x.Shape);
        ReadOnlySpan<T> xs = x.Span;
        Span<T> ys = y.Span;
        int mat = rows * cols;
        T zero = default;
        for (int b = 0; b < batch; b++)
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            int idx = b * mat + i * cols + j;
            bool keep = upper ? (long)j >= (long)i + k : (long)j <= (long)i + k;
            ys[idx] = keep ? xs[idx] : zero;
        }
        return y;
    }

    /// <summary>
    /// ONNX <c>ScatterND</c> (<c>batch_dims=0</c>, <c>reduction=none</c>): copies <c>data</c>, then writes each
    /// <c>updates</c> slice into the location addressed by the matching length-<c>k</c> index tuple in
    /// <c>indices</c> (negative-normalized). Dtype-preserving (float → device, int/bool host-side). Mirrors the
    /// CPU <c>ScatterNDKernel</c> none-reduction path (the only one distilgpt2 needs).
    /// </summary>
    private void RunScatterND(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        DeviceValue dataV = values[node.Inputs[0]];
        Tensor data = dataV.ToHost();
        long[] idx = ReadInts(values, node.Inputs[1]);
        int[] iDims = values[node.Inputs[1]].Shape.Dimensions.ToArray();
        Tensor updates = values[node.Inputs[2]].ToHost();
        string reduction = AttrStr(node, "reduction", "none");
        if (reduction != "none")
            throw new ModelSharpException($"ScatterND on the GPU engine supports reduction=none only, got '{reduction}'.");

        Tensor outT = data.Dtype switch
        {
            ElementType.Int64 => ScatterNDHost(data.AsInt64(), idx, iDims, updates.AsInt64()),
            ElementType.Int32 => ScatterNDHost(data.AsInt32(), idx, iDims, updates.AsInt32()),
            ElementType.Boolean => ScatterNDHost(data.AsBool(), idx, iDims, updates.AsBool()),
            _ => ScatterNDHost(data.AsFloat(), idx, iDims, updates.AsFloat()),
        };
        values[node.Outputs[0]] = dataV.IsFloat ? Load(outT) : DeviceValue.Host(outT);
    }

    private static Tensor<T> ScatterNDHost<T>(Tensor<T> data, long[] idx, int[] iDims, Tensor<T> updates)
        where T : unmanaged
    {
        ReadOnlySpan<int> dDims = data.Shape.Dimensions;
        int r = dDims.Length;
        int q = iDims.Length;
        int k = iDims[q - 1];
        int[] dStrides = Strides(dDims);

        int sliceLen = 1;
        for (int i = k; i < r; i++) sliceLen *= dDims[i];
        int numTuples = k == 0 ? 0 : idx.Length / k;

        var y = new Tensor<T>(data.Shape);
        data.Span.CopyTo(y.Span);
        Span<T> ys = y.Span;
        ReadOnlySpan<T> us = updates.Span;
        for (int t = 0; t < numTuples; t++)
        {
            int baseOff = 0;
            for (int j = 0; j < k; j++)
            {
                int g = (int)idx[t * k + j];
                if (g < 0) g += dDims[j];
                baseOff += g * dStrides[j];
            }
            int uBase = t * sliceLen;
            for (int s = 0; s < sliceLen; s++) ys[baseOff + s] = us[uBase + s];
        }
        return y;
    }

    /// <summary>
    /// ONNX <c>Gemm</c>: <c>Y = α·op(A)·op(B) + β·C</c> with optional transpose flags and broadcastable C, all on
    /// the device. A/B are pre-transposed via the existing Transpose kernel when needed, then multiplied with the
    /// MatMul kernel; α/β scaling and the C add are applied with elementwise kernels. Mirrors the CPU <c>GemmKernel</c>.
    /// </summary>
    private void RunGemm(GraphNode node, Dictionary<string, DeviceValue> values, List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        MemoryBuffer1D<float, Stride1D.Dense> b = FloatBuf(values, node.Inputs[1]);
        TensorShape sa = values[node.Inputs[0]].Shape;
        TensorShape sb = values[node.Inputs[1]].Shape;
        float alpha = AttrFloat(node, "alpha", 1f);
        float beta = AttrFloat(node, "beta", 1f);
        bool transA = AttrInt(node, "transA", 0) != 0;
        bool transB = AttrInt(node, "transB", 0) != 0;

        int M = transA ? sa[1] : sa[0];
        int K = transA ? sa[0] : sa[1];
        int Kb = transB ? sb[1] : sb[0];
        int N = transB ? sb[0] : sb[1];
        if (K != Kb) throw new ModelSharpException($"Gemm inner dimensions disagree: {K} vs {Kb}.");

        ArrayView<float> aView = transA ? Transpose2D(a, sa[0], sa[1], temps) : a.View;
        ArrayView<float> bView = transB ? Transpose2D(b, sb[0], sb[1], temps) : b.View;

        // A·B via the batched MatMul kernel with a single (trivial) batch.
        int total = M * N;
        MemoryBuffer1D<float, Stride1D.Dense> ab = AllocFloat(total);
        ArrayView<int> vAOff = Upload(new[] { 0 }, temps);
        ArrayView<int> vBOff = Upload(new[] { 0 }, temps);
        if (total > 0) _matmul(total, aView, bView, ab.View, vAOff, vBOff, M, K, N);

        // y = alpha*ab (+ beta*C, broadcast). Fold alpha/beta and the optional C with one Where-free pass:
        // scale ab by alpha, then add beta*C broadcast. Done with small device buffers for the scalars.
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(total);
        MemoryBuffer1D<float, Stride1D.Dense> alphaBuf = _accelerator.Allocate1D(new[] { alpha });
        temps.Add(alphaBuf);
        // y = ab * alpha  (broadcast scalar)
        {
            int[] outStrides = Strides(new[] { M, N });
            int[] sA = BroadcastStrides(new[] { M, N }, 2);
            int[] sB = BroadcastStrides(Array.Empty<int>(), 2);
            ArrayView<int> vOut = Upload(outStrides, temps);
            ArrayView<int> vAs = Upload(sA, temps);
            ArrayView<int> vBs = Upload(sB, temps);
            if (total > 0) _broadcast(total, ab.View, alphaBuf.View, y.View, vOut, vAs, vBs, 2, GpuKernels.OpMul);
        }

        bool hasC = node.Inputs.Count > 2 && node.Inputs[2].Length > 0;
        if (hasC)
        {
            MemoryBuffer1D<float, Stride1D.Dense> c = FloatBuf(values, node.Inputs[2]);
            TensorShape sc = values[node.Inputs[2]].Shape;
            // betaC = C * beta
            MemoryBuffer1D<float, Stride1D.Dense> betaC = AllocFloat(sc.Length);
            temps.Add(betaC);
            MemoryBuffer1D<float, Stride1D.Dense> betaBuf = _accelerator.Allocate1D(new[] { beta });
            temps.Add(betaBuf);
            {
                int[] outStrides = Strides(sc.Dimensions);
                int rankC = sc.Rank;
                int[] sCc = BroadcastStrides(sc.Dimensions, rankC);
                int[] sBeta = BroadcastStrides(Array.Empty<int>(), rankC);
                ArrayView<int> vOut = Upload(outStrides, temps);
                ArrayView<int> vCc = Upload(sCc, temps);
                ArrayView<int> vBeta = Upload(sBeta, temps);
                if (sc.Length > 0) _broadcast((int)sc.Length, c.View, betaBuf.View, betaC.View, vOut, vCc, vBeta, rankC, GpuKernels.OpMul);
            }
            // y = y + betaC (broadcast betaC over [M,N])
            MemoryBuffer1D<float, Stride1D.Dense> y2 = AllocFloat(total);
            int[] outStridesY = Strides(new[] { M, N });
            int[] sYstr = BroadcastStrides(new[] { M, N }, 2);
            int[] sCstr = BroadcastStrides(sc.Dimensions, 2);
            ArrayView<int> vOutY = Upload(outStridesY, temps);
            ArrayView<int> vYs = Upload(sYstr, temps);
            ArrayView<int> vCs = Upload(sCstr, temps);
            if (total > 0) _broadcast(total, y.View, betaC.View, y2.View, vOutY, vYs, vCs, 2, GpuKernels.OpAdd);
            temps.Add(ab);
            temps.Add(y);
            values[node.Outputs[0]] = DeviceValue.Device(y2, new TensorShape(M, N));
            return;
        }

        temps.Add(ab);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(M, N));
    }

    /// <summary>Transposes a 2-D [r,c] device buffer to [c,r] via the Transpose kernel; returns a temp view.</summary>
    private ArrayView<float> Transpose2D(MemoryBuffer1D<float, Stride1D.Dense> x, int r, int c, List<MemoryBuffer> temps)
    {
        int[] inStrides = Strides(new[] { r, c });
        int[] outDims = { c, r };
        int[] srcStrides = { inStrides[1], inStrides[0] };
        int[] outStrides = Strides(outDims);
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat((long)r * c);
        temps.Add(y);
        ArrayView<int> vOut = Upload(outStrides, temps);
        ArrayView<int> vSrc = Upload(srcStrides, temps);
        if (r * c > 0) _transpose(r * c, x.View, y.View, vOut, vSrc, 2);
        return y.View;
    }

    /// <summary>Uploads a host int array to the accelerator, tracking the buffer for disposal, and returns its view.</summary>
    private ArrayView<int> Upload(int[] data, List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<int, Stride1D.Dense> buf = _accelerator.Allocate1D(data);
        temps.Add(buf);
        return buf.View;
    }

    // --- Host-side row-major shape/stride and broadcasting math (mirrors the CPU Nd helpers) ---

    /// <summary>Row-major strides for a shape.</summary>
    private static int[] Strides(ReadOnlySpan<int> dims)
    {
        var s = new int[dims.Length];
        int acc = 1;
        for (int i = dims.Length - 1; i >= 0; i--) { s[i] = acc; acc *= dims[i]; }
        return s;
    }

    /// <summary>Strides over <paramref name="rank"/> axes for broadcasting; 0 where the axis is size-1 or padded.</summary>
    private static int[] BroadcastStrides(ReadOnlySpan<int> dims, int rank)
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
    private static int[] BroadcastShape(ReadOnlySpan<int> a, ReadOnlySpan<int> b)
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
    private static void SamePad(int inSize, int k, int stride, int dilation, bool upper, out int begin, out int end)
    {
        int outSize = (inSize + stride - 1) / stride;
        int needed = (outSize - 1) * stride + ((k - 1) * dilation + 1);
        int total = Math.Max(0, needed - inSize);
        begin = upper ? total / 2 : total - total / 2;
        end = total - begin;
    }

    // --- Typed attribute readers (mirror the CPU Attr helpers) ---

    private static long AttrInt(GraphNode n, string name, long dflt)
        => n.Attributes.TryGetValue(name, out object? v) ? Convert.ToInt64(v) : dflt;

    private static float AttrFloat(GraphNode n, string name, float dflt)
        => n.Attributes.TryGetValue(name, out object? v) ? Convert.ToSingle(v) : dflt;

    private static string AttrStr(GraphNode n, string name, string dflt)
        => n.Attributes.TryGetValue(name, out object? v) && v is string s ? s : dflt;

    private static int[] AttrInts(GraphNode n, string name, int[] dflt)
    {
        if (!n.Attributes.TryGetValue(name, out object? v)) return dflt;
        return v switch
        {
            long[] la => Array.ConvertAll(la, x => (int)x),
            int[] ia => ia,
            long l => new[] { (int)l },
            _ => dflt,
        };
    }

    // --- B5: on-device KV-cache + stateful attention decode seam ---

    /// <summary>
    /// Allocates a device-resident key/value cache for an autoregressive decoder with <paramref name="numHeads"/>
    /// heads of dimension <paramref name="headDim"/>, holding up to <paramref name="maxSeq"/> past tokens. The K
    /// and V buffers live on this engine's accelerator and persist across <see cref="DecodeStepAttention"/> calls
    /// so multi-step attention reads the past K/V from GPU memory with no host round-trip. Dispose the cache when
    /// the sequence is finished (or call <see cref="GpuKvCache.Reset"/> to reuse it for a new sequence).
    /// </summary>
    public GpuKvCache CreateKvCache(int numHeads, int maxSeq, int headDim)
    {
        var kBuf = _accelerator.Allocate1D<float>((long)numHeads * maxSeq * headDim);
        var vBuf = _accelerator.Allocate1D<float>((long)numHeads * maxSeq * headDim);
        return new GpuKvCache(kBuf, vBuf, numHeads, maxSeq, headDim);
    }

    /// <summary>
    /// One autoregressive attention decode step, run entirely on the GPU against a persistent on-device KV-cache.
    /// Appends the step's per-head key/value (<paramref name="stepK"/>/<paramref name="stepV"/>, each
    /// <c>[numHeads, stepLen, headDim]</c>) to <paramref name="cache"/> at its current sequence offset (a device→
    /// device copy, no realloc, no host copy), then computes scaled-dot-product attention of the step's query
    /// (<paramref name="stepQ"/>, <c>[numHeads, stepLen, headDim]</c>) against the <em>entire</em> cached
    /// key/value sequence and returns the context <c>[numHeads, stepLen, headDim]</c>. The softmax scale defaults
    /// to <c>1/√headDim</c>. All intermediates (scores, softmax, context) stay on the device; only the final
    /// context is downloaded. Mirrors a single transformer self-attention block so results match the CPU engine.
    /// </summary>
    public Tensor<float> DecodeStepAttention(
        GpuKvCache cache,
        Tensor<float> stepQ,
        Tensor<float> stepK,
        Tensor<float> stepV,
        float? scale = null)
    {
        if (cache.NumHeads != stepQ.Shape[0])
            throw new ModelSharpException($"KV-cache head count {cache.NumHeads} != query heads {stepQ.Shape[0]}.");
        int H = cache.NumHeads, D = cache.HeadDim;
        int stepLen = stepK.Shape[1];
        if (cache.SeqLen + stepLen > cache.MaxSeq)
            throw new ModelSharpException($"KV-cache overflow: {cache.SeqLen}+{stepLen} > maxSeq {cache.MaxSeq}.");
        float invScale = scale ?? (1f / MathF.Sqrt(D));

        // Upload the step's Q/K/V to the device.
        using MemoryBuffer1D<float, Stride1D.Dense> qBuf = _accelerator.Allocate1D(stepQ.Span.ToArray());
        using MemoryBuffer1D<float, Stride1D.Dense> kStep = _accelerator.Allocate1D(stepK.Span.ToArray());
        using MemoryBuffer1D<float, Stride1D.Dense> vStep = _accelerator.Allocate1D(stepV.Span.ToArray());

        // Append K/V into the persistent cache at [head, seqLen .. seqLen+stepLen) — device→device, per head
        // (the cache is laid out [H, maxSeq, D] so each head's region is contiguous).
        int prev = cache.SeqLen;
        for (int h = 0; h < H; h++)
        {
            long dstK = (long)h * cache.MaxSeq * D + (long)prev * D;
            long dstV = dstK;
            long src = (long)h * stepLen * D;
            long len = (long)stepLen * D;
            kStep.View.SubView(src, len).CopyTo(_accelerator.DefaultStream, cache.KBuffer.View.SubView(dstK, len));
            vStep.View.SubView(src, len).CopyTo(_accelerator.DefaultStream, cache.VBuffer.View.SubView(dstV, len));
        }
        int total = prev + stepLen;
        cache.SeqLen = total;

        // scores[h, i, j] = sum_d q[h,i,d] * K[h,j,d] * invScale, for j in [0,total).
        // Compute per head with the batched MatMul kernel: q_h [stepLen,D] · kᵀ_h [D,total].
        // We assemble a contiguous Kᵀ per head on the device via Transpose, then MatMul, Softmax, MatMul with V.
        var ctxHost = new float[(long)H * stepLen * D];
        var temps = new List<MemoryBuffer>();
        try
        {
            for (int h = 0; h < H; h++)
            {
                // Views into the cache for this head: K_h, V_h are [total, D] (contiguous prefix of [maxSeq,D]).
                ArrayView<float> kH = cache.KBuffer.View.SubView((long)h * cache.MaxSeq * D, (long)total * D);
                ArrayView<float> vH = cache.VBuffer.View.SubView((long)h * cache.MaxSeq * D, (long)total * D);
                ArrayView<float> qH = qBuf.View.SubView((long)h * stepLen * D, (long)stepLen * D);

                // Kᵀ_h : [D, total] via Transpose kernel (perm [1,0] over [total, D]).
                MemoryBuffer1D<float, Stride1D.Dense> ktBuf = _accelerator.Allocate1D<float>((long)D * total);
                temps.Add(ktBuf);
                {
                    int[] inStrides = Strides(new[] { total, D });
                    int[] outStrides = Strides(new[] { D, total });
                    int[] srcStrides = { inStrides[1], inStrides[0] };
                    ArrayView<int> vOut = Upload(outStrides, temps);
                    ArrayView<int> vSrc = Upload(srcStrides, temps);
                    _transpose(D * total, kH, ktBuf.View, vOut, vSrc, 2);
                }

                // scores = q_h [stepLen,D] · Kᵀ_h [D,total]  -> [stepLen, total]
                MemoryBuffer1D<float, Stride1D.Dense> scores = _accelerator.Allocate1D<float>((long)stepLen * total);
                temps.Add(scores);
                ArrayView<int> z0 = Upload(new[] { 0 }, temps);
                _matmul(stepLen * total, qH, ktBuf.View, scores.View, z0, z0, stepLen, D, total);

                // scaled = scores * invScale  (in place via a scalar-broadcast Mul)
                MemoryBuffer1D<float, Stride1D.Dense> scaled = _accelerator.Allocate1D<float>((long)stepLen * total);
                temps.Add(scaled);
                MemoryBuffer1D<float, Stride1D.Dense> scaleBuf = _accelerator.Allocate1D(new[] { invScale });
                temps.Add(scaleBuf);
                {
                    int[] outStrides = Strides(new[] { stepLen, total });
                    int[] sA = BroadcastStrides(new[] { stepLen, total }, 2);
                    int[] sB = BroadcastStrides(Array.Empty<int>(), 2);
                    ArrayView<int> vOut = Upload(outStrides, temps);
                    ArrayView<int> vA = Upload(sA, temps);
                    ArrayView<int> vB = Upload(sB, temps);
                    _broadcast(stepLen * total, scores.View, scaleBuf.View, scaled.View, vOut, vA, vB, 2, GpuKernels.OpMul);
                }

                // attn = softmax(scaled) over the last axis (total).
                MemoryBuffer1D<float, Stride1D.Dense> attn = _accelerator.Allocate1D<float>((long)stepLen * total);
                temps.Add(attn);
                _softmax(stepLen /* outer*inner with inner=1 */, scaled.View, attn.View, total, 1);

                // ctx_h = attn [stepLen,total] · V_h [total, D] -> [stepLen, D]
                MemoryBuffer1D<float, Stride1D.Dense> ctx = _accelerator.Allocate1D<float>((long)stepLen * D);
                temps.Add(ctx);
                _matmul(stepLen * D, attn.View, vH, ctx.View, z0, z0, stepLen, total, D);

                _accelerator.Synchronize();
                float[] ctxArr = ctx.GetAsArray1D();
                Array.Copy(ctxArr, 0, ctxHost, (long)h * stepLen * D, (long)stepLen * D);
            }
        }
        finally
        {
            foreach (MemoryBuffer b in temps) b.Dispose();
        }

        return new Tensor<float>(new TensorShape(H, stepLen, D), ctxHost);
    }

    /// <summary>
    /// Weights for one GPT-2-style (pre-LayerNorm) transformer decoder layer, in the conventional HF/GPT-2
    /// layout: the attention QKV and the two MLP linears use the <c>Conv1D</c> convention <c>y = x·W + b</c>
    /// with <c>W</c> shaped <c>[in, out]</c> (NOT transposed). <see cref="IlgpuEngine.DecodeLayerStep"/> consumes
    /// these to run a whole decoder layer on-device through the persistent KV-cache.
    /// </summary>
    public sealed class DecoderLayerWeights
    {
        /// <summary>ln_1 scale/bias, each <c>[embed]</c>.</summary>
        public Tensor<float> Ln1Scale = null!, Ln1Bias = null!;
        /// <summary>c_attn weight <c>[embed, 3*embed]</c> and bias <c>[3*embed]</c> (produces Q|K|V concatenated).</summary>
        public Tensor<float> QkvWeight = null!, QkvBias = null!;
        /// <summary>c_proj (attention output) weight <c>[embed, embed]</c> and bias <c>[embed]</c>.</summary>
        public Tensor<float> OutWeight = null!, OutBias = null!;
        /// <summary>ln_2 scale/bias, each <c>[embed]</c>.</summary>
        public Tensor<float> Ln2Scale = null!, Ln2Bias = null!;
        /// <summary>mlp.c_fc weight <c>[embed, hidden]</c> and bias <c>[hidden]</c>.</summary>
        public Tensor<float> FcWeight = null!, FcBias = null!;
        /// <summary>mlp.c_proj weight <c>[hidden, embed]</c> and bias <c>[embed]</c>.</summary>
        public Tensor<float> ProjWeight = null!, ProjBias = null!;
        /// <summary>LayerNorm epsilon (GPT-2 default 1e-5).</summary>
        public float Epsilon = 1e-5f;
    }

    /// <summary>
    /// Runs ONE full GPT-2-style decoder layer for a step's hidden state entirely on the GPU, threaded through
    /// the persistent on-device <paramref name="cache"/>. This closes the B5 gap where only the bare
    /// self-attention block (<see cref="DecodeStepAttention"/>) was wired into the cache seam — here the whole
    /// layer is composed on-device:
    /// <list type="number">
    /// <item>pre-attention LayerNorm of <paramref name="hidden"/> (<c>[stepLen, embed]</c>);</item>
    /// <item>fused QKV projection (<c>ln·Wqkv + bqkv</c>) and a split into Q/K/V, each <c>[stepLen, embed]</c>;</item>
    /// <item>reshape to per-head <c>[numHeads, stepLen, headDim]</c>, append K/V into the persistent cache, and
    ///       scaled-dot-product attention over the <em>entire</em> cached prefix (the K/V stay on-device across
    ///       steps);</item>
    /// <item>output projection (<c>ctx·Wo + bo</c>) and the attention residual add;</item>
    /// <item>pre-MLP LayerNorm, the two-matmul GELU MLP, and the MLP residual add.</item>
    /// </list>
    /// Every op runs on the GPU kernels (LayerNorm/MatMul/Softmax/Transpose/Gelu/Add); only the final hidden
    /// state <c>[stepLen, embed]</c> is downloaded. Validates against a CPU reference to 1e-3 in the tests.
    /// </summary>
    public Tensor<float> DecodeLayerStep(GpuKvCache cache, Tensor<float> hidden, DecoderLayerWeights w)
    {
        if (cache is null) throw new ArgumentNullException(nameof(cache));
        if (hidden is null) throw new ArgumentNullException(nameof(hidden));
        if (w is null) throw new ArgumentNullException(nameof(w));
        if (hidden.Shape.Rank != 2)
            throw new ModelSharpException($"DecodeLayerStep expects hidden [stepLen, embed]; got rank {hidden.Shape.Rank}.");

        int S = hidden.Shape[0];      // step length (new tokens this step)
        int E = hidden.Shape[1];      // embedding dim
        int H = cache.NumHeads, D = cache.HeadDim;
        if (H * D != E)
            throw new ModelSharpException($"numHeads*headDim ({H}*{D}) != embed ({E}).");
        if (cache.SeqLen + S > cache.MaxSeq)
            throw new ModelSharpException($"KV-cache overflow: {cache.SeqLen}+{S} > maxSeq {cache.MaxSeq}.");

        var temps = new List<MemoryBuffer>();
        try
        {
            MemoryBuffer1D<float, Stride1D.Dense> hid = UploadF(hidden.Span.ToArray(), temps); // [S,E]

            // 1) ln1 = LayerNorm(hidden)
            MemoryBuffer1D<float, Stride1D.Dense> ln1 = LayerNormDev(hid, S, E, w.Ln1Scale, w.Ln1Bias, w.Epsilon, temps);

            // 2) qkv = ln1 [S,E] · Wqkv [E,3E] + bqkv  -> [S, 3E]; split columns into Q|K|V [S,E] each.
            MemoryBuffer1D<float, Stride1D.Dense> wqkv = UploadF(w.QkvWeight.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> bqkv = UploadF(w.QkvBias.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> qkv = LinearDev(ln1, S, E, wqkv, 3 * E, bqkv, temps); // [S,3E]
            MemoryBuffer1D<float, Stride1D.Dense> qHeads = SplitColsToHeads(qkv, S, 3 * E, 0 * E, H, D, temps);
            MemoryBuffer1D<float, Stride1D.Dense> kHeads = SplitColsToHeads(qkv, S, 3 * E, 1 * E, H, D, temps);
            MemoryBuffer1D<float, Stride1D.Dense> vHeads = SplitColsToHeads(qkv, S, 3 * E, 2 * E, H, D, temps);

            // 3) append K/V into the persistent cache and attend over the whole prefix -> ctx [H,S,D].
            MemoryBuffer1D<float, Stride1D.Dense> ctxHeads = AttendOverCache(cache, qHeads, kHeads, vHeads, S, temps);

            // ctx [H,S,D] -> [S,E] (head-major D contiguous): out[s, h*D+d] = ctx[h, s, d].
            MemoryBuffer1D<float, Stride1D.Dense> ctxSE = HeadsToCols(ctxHeads, S, H, D, temps); // [S,E]

            // 4) attnOut = ctx [S,E] · Wo [E,E] + bo ; residual h2 = hidden + attnOut.
            MemoryBuffer1D<float, Stride1D.Dense> wo = UploadF(w.OutWeight.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> bo = UploadF(w.OutBias.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> attnOut = LinearDev(ctxSE, S, E, wo, E, bo, temps); // [S,E]
            MemoryBuffer1D<float, Stride1D.Dense> h2 = AddDev(hid, attnOut, S * E, temps);

            // 5) ln2 -> MLP (GELU) -> residual.
            MemoryBuffer1D<float, Stride1D.Dense> ln2 = LayerNormDev(h2, S, E, w.Ln2Scale, w.Ln2Bias, w.Epsilon, temps);
            int Hid = w.FcBias.Shape[0]; // MLP hidden width
            MemoryBuffer1D<float, Stride1D.Dense> wfc = UploadF(w.FcWeight.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> bfc = UploadF(w.FcBias.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> fc = LinearDev(ln2, S, E, wfc, Hid, bfc, temps);   // [S,Hid]
            MemoryBuffer1D<float, Stride1D.Dense> act = AllocFloat((long)S * Hid); temps.Add(act);
            if (S * Hid > 0) _gelu(S * Hid, fc.View, act.View);
            MemoryBuffer1D<float, Stride1D.Dense> wproj = UploadF(w.ProjWeight.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> bproj = UploadF(w.ProjBias.Span.ToArray(), temps);
            MemoryBuffer1D<float, Stride1D.Dense> mlp = LinearDev(act, S, Hid, wproj, E, bproj, temps); // [S,E]
            MemoryBuffer1D<float, Stride1D.Dense> outBuf = AddDev(h2, mlp, S * E, temps);

            _accelerator.Synchronize();
            float[] raw = outBuf.GetAsArray1D();   // buffer may be a size-1 sentinel; trim to S*E.
            long len = (long)S * E;
            float[] host;
            if (raw.LongLength == len) host = raw;
            else { host = new float[len]; Array.Copy(raw, host, len); }
            return new Tensor<float>(new TensorShape(S, E), host);
        }
        finally
        {
            foreach (MemoryBuffer b in temps) b.Dispose();
        }
    }

    // --- Device building blocks for DecodeLayerStep (operate directly on buffers; all tracked in temps) ---

    /// <summary>Uploads a host float array to the device, tracking it for disposal (empty → sentinel buffer).</summary>
    private MemoryBuffer1D<float, Stride1D.Dense> UploadF(float[] data, List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> buf = data.Length > 0 ? _accelerator.Allocate1D(data) : AllocFloat(0);
        temps.Add(buf);
        return buf;
    }

    /// <summary>LayerNorm over the trailing <paramref name="norm"/> dim of an [outer, norm] buffer, on-device.</summary>
    private MemoryBuffer1D<float, Stride1D.Dense> LayerNormDev(
        MemoryBuffer1D<float, Stride1D.Dense> x, int outer, int norm,
        Tensor<float> scale, Tensor<float> bias, float eps, List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> sc = UploadF(scale.Span.ToArray(), temps);
        MemoryBuffer1D<float, Stride1D.Dense> bi = UploadF(bias.Span.ToArray(), temps);
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat((long)outer * norm); temps.Add(y);
        if (outer > 0 && norm > 0) _layerNorm(outer, x.View, sc.View, bi.View, y.View, norm, eps, 1);
        return y;
    }

    /// <summary>Linear <c>y = x[rows,K]·W[K,N] + b[N]</c> on-device (W in [in,out] Conv1D layout); returns [rows,N].</summary>
    private MemoryBuffer1D<float, Stride1D.Dense> LinearDev(
        MemoryBuffer1D<float, Stride1D.Dense> x, int rows, int K,
        MemoryBuffer1D<float, Stride1D.Dense> wBuf, int N,
        MemoryBuffer1D<float, Stride1D.Dense> bBuf, List<MemoryBuffer> temps)
    {
        int total = rows * N;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(total); temps.Add(y);
        ArrayView<int> z0 = Upload(new[] { 0 }, temps);
        if (total > 0) _matmul(total, x.View, wBuf.View, y.View, z0, z0, rows, K, N);
        // y = y + b  (broadcast bias [N] over rows)
        if (total > 0)
        {
            int[] outStrides = Strides(new[] { rows, N });
            int[] sY = BroadcastStrides(new[] { rows, N }, 2);
            int[] sB = BroadcastStrides(new[] { N }, 2);
            ArrayView<int> vOut = Upload(outStrides, temps);
            ArrayView<int> vY = Upload(sY, temps);
            ArrayView<int> vB = Upload(sB, temps);
            MemoryBuffer1D<float, Stride1D.Dense> y2 = AllocFloat(total); temps.Add(y2);
            _broadcast(total, y.View, bBuf.View, y2.View, vOut, vY, vB, 2, GpuKernels.OpAdd);
            return y2;
        }
        return y;
    }

    /// <summary>Elementwise add of two equal-length [.. ] buffers; returns a fresh buffer.</summary>
    private MemoryBuffer1D<float, Stride1D.Dense> AddDev(
        MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b, int n, List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(n); temps.Add(y);
        if (n > 0) _add(n, a.View, b.View, y.View);
        return y;
    }

    /// <summary>
    /// Extracts the <c>[S, E]</c> column block starting at <paramref name="colOffset"/> of a row-major
    /// <c>[S, rowWidth]</c> buffer and re-lays it as per-head <c>[H, S, D]</c> (E = H·D, head-major D contiguous):
    /// <c>out[h, s, d] = x[s, colOffset + h*D + d]</c>. Uses the Gather kernel with host-precomputed offsets.
    /// </summary>
    private MemoryBuffer1D<float, Stride1D.Dense> SplitColsToHeads(
        MemoryBuffer1D<float, Stride1D.Dense> x, int S, int rowWidth, int colOffset, int H, int D, List<MemoryBuffer> temps)
    {
        int total = H * S * D;
        var off = new int[total == 0 ? 1 : total];
        int p = 0;
        for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++)
        for (int d = 0; d < D; d++)
            off[p++] = s * rowWidth + colOffset + h * D + d;
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(total); temps.Add(y);
        ArrayView<int> vOff = Upload(off, temps);
        if (total > 0) _gather(total, x.View, y.View, vOff);
        return y;
    }

    /// <summary>Inverse of <see cref="SplitColsToHeads"/> for a single block: <c>[H,S,D] -> [S,E]</c>, <c>out[s, h*D+d] = x[h,s,d]</c>.</summary>
    private MemoryBuffer1D<float, Stride1D.Dense> HeadsToCols(
        MemoryBuffer1D<float, Stride1D.Dense> x, int S, int H, int D, List<MemoryBuffer> temps)
    {
        int E = H * D;
        int total = S * E;
        var off = new int[total == 0 ? 1 : total];
        int p = 0;
        for (int s = 0; s < S; s++)
        for (int h = 0; h < H; h++)
        for (int d = 0; d < D; d++)
            off[p++] = (h * S + s) * D + d; // source index into [H,S,D]
        MemoryBuffer1D<float, Stride1D.Dense> y = AllocFloat(total); temps.Add(y);
        ArrayView<int> vOff = Upload(off, temps);
        if (total > 0) _gather(total, x.View, y.View, vOff);
        return y;
    }

    /// <summary>
    /// Appends per-head K/V (<c>[H,S,D]</c>) into the persistent cache at its current offset and computes
    /// scaled-dot-product attention of Q (<c>[H,S,D]</c>) over the whole cached prefix, returning ctx
    /// <c>[H,S,D]</c> on-device. Same math as <see cref="DecodeStepAttention"/> but keeps everything on the
    /// device (no per-head host download) so it can chain into the rest of <see cref="DecodeLayerStep"/>.
    /// </summary>
    private MemoryBuffer1D<float, Stride1D.Dense> AttendOverCache(
        GpuKvCache cache,
        MemoryBuffer1D<float, Stride1D.Dense> qHeads,
        MemoryBuffer1D<float, Stride1D.Dense> kHeads,
        MemoryBuffer1D<float, Stride1D.Dense> vHeads,
        int S, List<MemoryBuffer> temps)
    {
        int H = cache.NumHeads, D = cache.HeadDim;
        int prev = cache.SeqLen;
        for (int h = 0; h < H; h++)
        {
            long dst = (long)h * cache.MaxSeq * D + (long)prev * D;
            long src = (long)h * S * D;
            long len = (long)S * D;
            if (len > 0)
            {
                kHeads.View.SubView(src, len).CopyTo(_accelerator.DefaultStream, cache.KBuffer.View.SubView(dst, len));
                vHeads.View.SubView(src, len).CopyTo(_accelerator.DefaultStream, cache.VBuffer.View.SubView(dst, len));
            }
        }
        int total = prev + S;
        cache.SeqLen = total;
        float invScale = 1f / MathF.Sqrt(D);

        MemoryBuffer1D<float, Stride1D.Dense> ctx = AllocFloat((long)H * S * D); temps.Add(ctx);
        ArrayView<int> z0 = Upload(new[] { 0 }, temps);
        for (int h = 0; h < H; h++)
        {
            ArrayView<float> kH = cache.KBuffer.View.SubView((long)h * cache.MaxSeq * D, (long)total * D);
            ArrayView<float> vH = cache.VBuffer.View.SubView((long)h * cache.MaxSeq * D, (long)total * D);
            ArrayView<float> qH = qHeads.View.SubView((long)h * S * D, (long)S * D);

            // Kᵀ [D,total]
            MemoryBuffer1D<float, Stride1D.Dense> kt = AllocFloat((long)D * total); temps.Add(kt);
            int[] outStr = Strides(new[] { D, total });
            int[] inStr = Strides(new[] { total, D });
            int[] srcStr = { inStr[1], inStr[0] };
            ArrayView<int> vOut = Upload(outStr, temps);
            ArrayView<int> vSrc = Upload(srcStr, temps);
            if (D * total > 0) _transpose(D * total, kH, kt.View, vOut, vSrc, 2);

            // scores = q[S,D]·Kᵀ[D,total] -> [S,total], scaled, softmax, ·V -> [S,D]
            MemoryBuffer1D<float, Stride1D.Dense> scores = AllocFloat((long)S * total); temps.Add(scores);
            if (S * total > 0) _matmul(S * total, qH, kt.View, scores.View, z0, z0, S, D, total);

            MemoryBuffer1D<float, Stride1D.Dense> scaled = AllocFloat((long)S * total); temps.Add(scaled);
            MemoryBuffer1D<float, Stride1D.Dense> scaleBuf = _accelerator.Allocate1D(new[] { invScale }); temps.Add(scaleBuf);
            {
                int[] os = Strides(new[] { S, total });
                int[] sA = BroadcastStrides(new[] { S, total }, 2);
                int[] sB = BroadcastStrides(Array.Empty<int>(), 2);
                ArrayView<int> vO = Upload(os, temps);
                ArrayView<int> vA = Upload(sA, temps);
                ArrayView<int> vB = Upload(sB, temps);
                if (S * total > 0) _broadcast(S * total, scores.View, scaleBuf.View, scaled.View, vO, vA, vB, 2, GpuKernels.OpMul);
            }

            MemoryBuffer1D<float, Stride1D.Dense> attn = AllocFloat((long)S * total); temps.Add(attn);
            if (S > 0 && total > 0) _softmax(S, scaled.View, attn.View, total, 1);

            ArrayView<float> ctxH = ctx.View.SubView((long)h * S * D, (long)S * D);
            if (S * D > 0) _matmul(S * D, attn.View, vH, ctxH, z0, z0, S, total, D);
        }
        return ctx;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}
