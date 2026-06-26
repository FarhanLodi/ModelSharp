using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
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
        _context = Context.CreateDefault();
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
                    case "Conv": RunConv(node, values, temps); break;
                    default:
                        throw new UnsupportedOperatorException(node.OpType,
                            $"node '{node.Name}' — the GPU engine covers elementwise/activations, Softmax, " +
                            "Transpose, ReduceSum/ReduceMean, LayerNormalization, Gather/Concat/Slice/Cast, " +
                            "MatMul and Conv ops");
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
            foreach (DeviceValue v in values.Values) v.FloatBuf?.Dispose();
            foreach (MemoryBuffer buf in temps) buf.Dispose();
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
            return DeviceValue.Device(_accelerator.Allocate1D(t.AsFloat().Span.ToArray()), t.Shape);
        return DeviceValue.Host(t);
    }

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
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        MemoryBuffer1D<float, Stride1D.Dense> b = FloatBuf(values, node.Inputs[1]);
        TensorShape sa = values[node.Inputs[0]].Shape;
        TensorShape sb = values[node.Inputs[1]].Shape;

        if (sa.Equals(sb))
        {
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> fast = op switch
            {
                GpuKernels.OpAdd => _add,
                GpuKernels.OpSub => _sub,
                GpuKernels.OpMul => _mul,
                _ => _div,
            };
            MemoryBuffer1D<float, Stride1D.Dense> yEqual = _accelerator.Allocate1D<float>(a.Length);
            fast((int)a.Length, a.View, b.View, yEqual.View);
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

        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(n);
        ArrayView<int> vOut = Upload(outStrides, temps);
        ArrayView<int> vA = Upload(strideA, temps);
        ArrayView<int> vB = Upload(strideB, temps);

        _broadcast(n, a.View, b.View, y.View, vOut, vA, vB, rank, op);
        values[node.Outputs[0]] = DeviceValue.Device(y, new TensorShape(outd));
    }

    /// <summary>Runs a unary elementwise op (currently Relu) over the input buffer.</summary>
    private void RunUnary(
        Action<Index1D, ArrayView<float>, ArrayView<float>> kernel,
        GraphNode node,
        Dictionary<string, DeviceValue> values)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(a.Length);
        kernel((int)a.Length, a.View, y.View);
        values[node.Outputs[0]] = DeviceValue.Device(y, values[node.Inputs[0]].Shape);
    }

    /// <summary>Leaky ReLU (<c>x</c> if x ≥ 0 else <c>α·x</c>); <c>alpha</c> defaults to 0.01. Mirrors the CPU <c>LeakyReluKernel</c>.</summary>
    private void RunLeakyRelu(GraphNode node, Dictionary<string, DeviceValue> values)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = FloatBuf(values, node.Inputs[0]);
        float alpha = AttrFloat(node, "alpha", 0.01f);
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(a.Length);
        _leakyRelu((int)a.Length, a.View, y.View, alpha);
        values[node.Outputs[0]] = DeviceValue.Device(y, values[node.Inputs[0]].Shape);
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
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(n);
        ArrayView<int> vOut = Upload(outStrides, temps);
        ArrayView<int> vSrc = Upload(srcStrides, temps);

        _transpose(n, x.View, y.View, vOut, vSrc, rank);
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
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(total);

        _softmax(outer * inner, x.View, y.View, axisSize, inner);
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
            // Identity: copy the input through unchanged.
            MemoryBuffer1D<float, Stride1D.Dense> copy = _accelerator.Allocate1D(x.GetAsArray1D());
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
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(outLen);
        ArrayView<int> vBase = Upload(outBase, temps);
        ArrayView<int> vRedOut = Upload(redOutStrides, temps);
        ArrayView<int> vRedStr = Upload(redStrides, temps);

        _reduce(outLen, x.View, y.View, vBase, vRedOut, vRedStr, numRed, redCount, divisor);

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
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(total);
        _layerNorm(outer, x.View, scale.View, biasView, y.View, norm, eps, hasBias ? 1 : 0);
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
        MemoryBuffer1D<float, Stride1D.Dense> data = FloatBuf(values, node.Inputs[0]);
        TensorShape dS = values[node.Inputs[0]].Shape;
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
            values[node.Outputs[0]] = DeviceValue.Device(_accelerator.Allocate1D(y.Span.ToArray()), y.Shape);
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
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(total);
        ArrayView<int> vAOff = Upload(aOffArr, temps);
        ArrayView<int> vBOff = Upload(bOffArr, temps);

        _matmul(total, a.View, b.View, y.View, vAOff, vBOff, M, K, N);
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

    /// <inheritdoc />
    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}
