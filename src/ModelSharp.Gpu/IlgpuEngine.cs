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
/// Covers the elementwise ops (Add/Sub/Mul/Div, with NumPy-style broadcasting, plus Relu) and the
/// two heavyweight tensor ops: batched MatMul (NumPy matmul semantics) and Conv2D (NCHW, with
/// stride/padding/dilation/groups/bias). The engine is float32-only; the device kernels live in
/// <see cref="GpuKernels"/> and the host-side stride/offset precomputation lives here.
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

    /// <inheritdoc />
    public IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds)
    {
        var buffers = new Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>>();
        var shapes = new Dictionary<string, TensorShape>();
        var temps = new List<MemoryBuffer>();   // auxiliary buffers (strides, batch offsets, dummy bias)
        try
        {
            foreach (KeyValuePair<string, Tensor> init in _graph.Initializers)
            {
                Tensor<float> initF = init.Value.AsFloat();
                buffers[init.Key] = _accelerator.Allocate1D(initF.Span.ToArray());
                shapes[init.Key] = initF.Shape;
            }
            foreach (string input in _graph.Inputs)
            {
                if (!feeds.TryGetValue(input, out NamedTensor? nt))
                    throw new ModelSharpException($"Missing feed for required input '{input}'.");
                buffers[input] = _accelerator.Allocate1D(nt.Data.Span.ToArray());
                shapes[input] = nt.Data.Shape;
            }

            foreach (GraphNode node in _graph.Nodes)
            {
                switch (node.OpType)
                {
                    case "Add": RunBinary(GpuKernels.OpAdd, node, buffers, shapes, temps); break;
                    case "Sub": RunBinary(GpuKernels.OpSub, node, buffers, shapes, temps); break;
                    case "Mul": RunBinary(GpuKernels.OpMul, node, buffers, shapes, temps); break;
                    case "Div": RunBinary(GpuKernels.OpDiv, node, buffers, shapes, temps); break;
                    case "Relu": RunUnary(_relu, node, buffers, shapes); break;
                    case "MatMul": RunMatMul(node, buffers, shapes, temps); break;
                    case "Conv": RunConv(node, buffers, shapes, temps); break;
                    default:
                        throw new UnsupportedOperatorException(node.OpType,
                            $"node '{node.Name}' — the GPU engine covers elementwise, MatMul and Conv ops");
                }
            }
            _accelerator.Synchronize();

            var result = new Dictionary<string, NamedTensor>();
            foreach (string outName in _graph.Outputs)
            {
                float[] data = buffers[outName].GetAsArray1D();
                result[outName] = new NamedTensor(outName, new Tensor<float>(shapes[outName], data));
            }
            return result;
        }
        finally
        {
            foreach (MemoryBuffer1D<float, Stride1D.Dense> buf in buffers.Values) buf.Dispose();
            foreach (MemoryBuffer buf in temps) buf.Dispose();
        }
    }

    /// <summary>
    /// Runs a binary elementwise op (Add/Sub/Mul/Div), selected by <paramref name="op"/>. Equal shapes
    /// take the contiguous fast path; differing shapes use NumPy-style broadcasting where the per-axis
    /// broadcast strides are precomputed on the host and passed to the kernel as device arrays.
    /// </summary>
    private void RunBinary(
        int op,
        GraphNode node,
        Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> buffers,
        Dictionary<string, TensorShape> shapes,
        List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = buffers[node.Inputs[0]];
        MemoryBuffer1D<float, Stride1D.Dense> b = buffers[node.Inputs[1]];
        TensorShape sa = shapes[node.Inputs[0]];
        TensorShape sb = shapes[node.Inputs[1]];

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
            buffers[node.Outputs[0]] = yEqual;
            shapes[node.Outputs[0]] = sa;
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
        buffers[node.Outputs[0]] = y;
        shapes[node.Outputs[0]] = new TensorShape(outd);
    }

    /// <summary>Runs a unary elementwise op (currently Relu) over the input buffer.</summary>
    private void RunUnary(
        Action<Index1D, ArrayView<float>, ArrayView<float>> kernel,
        GraphNode node,
        Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> buffers,
        Dictionary<string, TensorShape> shapes)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = buffers[node.Inputs[0]];
        MemoryBuffer1D<float, Stride1D.Dense> y = _accelerator.Allocate1D<float>(a.Length);
        kernel((int)a.Length, a.View, y.View);
        buffers[node.Outputs[0]] = y;
        shapes[node.Outputs[0]] = shapes[node.Inputs[0]];
    }

    /// <summary>
    /// Matrix multiply with full NumPy semantics: 2-D, stacked, and batched n-D × n-D with broadcasting
    /// of the leading (batch) dimensions; 1-D operands are promoted per NumPy rules. The batch is flattened
    /// on the host and each flattened batch index's operand base offsets are precomputed and uploaded, so the
    /// kernel is a simple one-thread-per-output-element dot product. Mirrors the CPU <c>MatMulKernel</c>.
    /// </summary>
    private void RunMatMul(
        GraphNode node,
        Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> buffers,
        Dictionary<string, TensorShape> shapes,
        List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> a = buffers[node.Inputs[0]];
        MemoryBuffer1D<float, Stride1D.Dense> b = buffers[node.Inputs[1]];
        TensorShape saS = shapes[node.Inputs[0]];
        TensorShape sbS = shapes[node.Inputs[1]];

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
        buffers[node.Outputs[0]] = y;
        shapes[node.Outputs[0]] = new TensorShape(outDims.ToArray());
    }

    /// <summary>
    /// 2-D convolution (NCHW) with strides, pads/auto_pad, dilations, group and optional bias. A direct
    /// convolution: one GPU thread per output element. Mirrors the CPU <c>ConvKernel</c>.
    /// </summary>
    private void RunConv(
        GraphNode node,
        Dictionary<string, MemoryBuffer1D<float, Stride1D.Dense>> buffers,
        Dictionary<string, TensorShape> shapes,
        List<MemoryBuffer> temps)
    {
        MemoryBuffer1D<float, Stride1D.Dense> x = buffers[node.Inputs[0]];
        MemoryBuffer1D<float, Stride1D.Dense> w = buffers[node.Inputs[1]];
        TensorShape xS = shapes[node.Inputs[0]];
        TensorShape wS = shapes[node.Inputs[1]];
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
            biasView = buffers[node.Inputs[2]].View;
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
        buffers[node.Outputs[0]] = y;
        shapes[node.Outputs[0]] = new TensorShape(N, cout, outH, outW);
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
