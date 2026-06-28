using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ModelSharp.Cpu;
using ModelSharp.Engine;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;

namespace ModelSharp.Bench;

/// <summary>
/// Head-to-head latency: ModelSharp's ManagedCpuEngine (with the native AVX-512 GEMM seam)
/// vs Microsoft ONNX Runtime, on the SAME .onnx model and the SAME random input.
///
/// Goal: is ModelSharp competitive? Outputs are parity-checked (max relative error) so the
/// latency comparison is apples-to-apples. Also reports ModelSharp managed-only (native off)
/// when run with MODELSHARP_NATIVE=0 in the environment for the managed column.
///
/// Run:  dotnet run -c Release --project bench/ModelSharp.Bench -- ort [modelPath]
/// </summary>
internal static class OrtCompare
{
    private const int Warmup = 5;
    private const int Iters = 30;

    public static void Run(string[] args)
    {
        string? modelArg = args.Length > 1 ? args[1] : null;
        string model = ResolveModel(modelArg);
        if (model is null)
        {
            Console.WriteLine("No usable .onnx model found. Pass a path: -- ort /path/to/model.onnx");
            return;
        }

        Console.WriteLine($"Model: {model}");
        Console.WriteLine($"ORT version: {OrtVersion()}");
        Console.WriteLine($"Cores: {Environment.ProcessorCount}");
        Console.WriteLine($"MODELSHARP_NATIVE = {Environment.GetEnvironmentVariable("MODELSHARP_NATIVE") ?? "(default=on)"}");
        Console.WriteLine($"NativeGemm.Available = {ModelSharp.Native.NativeGemm.Available}, Enabled = {ModelSharp.Native.NativeGemm.Enabled}");
        Console.WriteLine();

        // --- Load ModelSharp graph + engine once; it describes the inputs we must feed. ---
        ModelGraph graph = OnnxModelLoader.LoadModel(model);
        using var msEngine = new ManagedCpuEngine(graph);

        // Open the ORT session once; its InputMetadata gives authoritative shapes/dtypes,
        // including for models whose ONNX value_info leaves the input rank unspecified.
        using var so = new SessionOptions();
        using var session = new InferenceSession(model, so);

        // Decide batch sizes to try: only when the first input axis is dynamic (<=0/symbolic).
        var firstDims = session.InputMetadata.First().Value.Dimensions;
        bool batchDynamic = firstDims.Length > 0 && firstDims[0] <= 0;
        int[] batches = batchDynamic ? new[] { 1, 4 } : new[] { 1 };

        Console.WriteLine($"{"batch",6} {"ModelSharp ms",14} {"ORT ms",10} {"ratio ORT/MS",13} {"max rel err",13} {"match",7}");
        Console.WriteLine(new string('-', 70));

        foreach (int batch in batches)
        {
            try
            {
                BenchOne(msEngine, session, batch);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{batch,6}  FAILED: {ex.GetType().Name}: {ex.Message}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("ratio > 1 => ModelSharp faster than ORT; < 1 => ORT faster.");
        Console.WriteLine("Compare managed vs native: rerun with MODELSHARP_NATIVE=0 dotnet run ... -- ort " + (modelArg ?? ""));
    }

    private static void BenchOne(ManagedCpuEngine msEngine, InferenceSession session, int batch)
    {
        var rng = new Random(1234);

        // Build feeds for every input using ORT's authoritative shape/dtype metadata,
        // replacing dynamic axes (axis 0 = batch, others = a small default).
        var msFeeds = new Dictionary<string, NamedTensor>();
        var ortInputs = new List<NamedOnnxValue>();

        foreach (var kv in session.InputMetadata)
        {
            string name = kv.Key;
            int[] shape = ResolveShape(name, kv.Value.Dimensions, batch);
            long count = 1;
            foreach (int d in shape) count *= d;
            Type t = kv.Value.ElementType;

            if (t == typeof(float))
            {
                var data = new float[count];
                for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
                msFeeds[name] = new NamedTensor(name,
                    ModelSharp.Tensors.Tensor<float>.FromArray(new TensorShape(shape), data));
                ortInputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, shape)));
            }
            else if (t == typeof(long))
            {
                var data = new long[count];
                for (int i = 0; i < data.Length; i++) data[i] = rng.Next(0, 100);
                msFeeds[name] = new NamedTensor(name,
                    ModelSharp.Tensors.Tensor<long>.FromArray(new TensorShape(shape), data));
                ortInputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, shape)));
            }
            else if (t == typeof(int))
            {
                var data = new int[count];
                for (int i = 0; i < data.Length; i++) data[i] = rng.Next(0, 100);
                msFeeds[name] = new NamedTensor(name,
                    ModelSharp.Tensors.Tensor<int>.FromArray(new TensorShape(shape), data));
                ortInputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(data, shape)));
            }
            else
            {
                throw new NotSupportedException($"input '{name}' dtype {t.Name} not handled by bench");
            }
        }

        // --- ModelSharp run ---
        IReadOnlyDictionary<string, NamedTensor> msOut = msEngine.Run(msFeeds); // warmup-ish
        for (int i = 0; i < Warmup; i++) msOut = msEngine.Run(msFeeds);
        double msMs = Median(Iters, () => msEngine.Run(msFeeds));

        // --- ORT run (CPU EP, default settings) ---
        string ortOutName = session.OutputMetadata
            .First(kv => kv.Value.ElementType == typeof(float)).Key;
        var outNames = new[] { ortOutName };

        // Pick the same-named ModelSharp output for parity (fall back to first float output).
        string msOutName = msOut.ContainsKey(ortOutName)
            ? ortOutName
            : msEngine.Outputs.First(o => o.ElementType == ElementType.Float32).Name;
        float[] msResult = msOut[msOutName].Data.Span.ToArray();

        float[] ortResult = Array.Empty<float>();
        using (var r = session.Run(ortInputs, outNames)) // warmup
            ortResult = r.First().AsEnumerable<float>().ToArray();
        for (int i = 0; i < Warmup; i++)
            using (var r = session.Run(ortInputs, outNames)) { }

        double ortMs = Median(Iters, () =>
        {
            using var r = session.Run(ortInputs, outNames);
        });

        // Capture a fresh ORT result for parity after warmups.
        using (var r = session.Run(ortInputs, outNames))
            ortResult = r.First().AsEnumerable<float>().ToArray();

        double maxRel = MaxRelErr(ortResult, msResult);
        bool match = maxRel < 1e-2 && ortResult.Length == msResult.Length;
        double ratio = ortMs / msMs;

        Console.WriteLine($"{batch,6} {msMs,14:0.000} {ortMs,10:0.000} {ratio,13:0.00} {maxRel,13:0.0e+0} {(match ? "yes" : "NO"),7}");
    }

    // Known concrete shapes for famous models whose ONNX exports leave dims symbolic
    // (resnet50/yolo export every NCHW axis as a named symbol -> ORT reports them all dynamic).
    private static readonly Dictionary<string, int[]> ShapeOverrides = new()
    {
        ["pixel_values"] = new[] { -1, 3, 224, 224 }, // resnet50 (HF export)
        ["images"]       = new[] { -1, 3, 640, 640 }, // yolov8
    };

    private static int[] ResolveShape(string name, IReadOnlyList<int> dims, int batch)
    {
        IReadOnlyList<int> src = ShapeOverrides.TryGetValue(name, out var ov) ? ov : dims;
        if (src.Count == 0) return new[] { batch };
        var shape = new int[src.Count];
        for (int i = 0; i < src.Count; i++)
        {
            int d = src[i];
            if (d > 0) { shape[i] = d; continue; }
            // dynamic axis: batch on axis 0, otherwise a sensible default (sequence length, etc.)
            shape[i] = i == 0 ? batch : 8;
        }
        return shape;
    }

    private static string ResolveModel(string? arg)
    {
        if (arg != null && File.Exists(arg)) return arg;

        // Priority list of likely-present models, smallest/most-reliable first.
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "assets", "mnist-12.onnx"),
            "/home/x16/ModelSharp/tests/ModelSharp.Tests/assets/mnist-12.onnx",
            "/home/x16/models/resnet50.onnx",
            "/home/x16/models/yolo.onnx",
        };
        foreach (var c in candidates)
            if (File.Exists(c)) return c;
        return null!;
    }

    private static string OrtVersion()
    {
        try
        {
            var v = typeof(InferenceSession).Assembly.GetName().Version;
            return v?.ToString() ?? "unknown";
        }
        catch { return "unknown"; }
    }

    private static double Median(int n, Action body)
    {
        var samples = new double[n];
        for (int i = 0; i < n; i++)
        {
            var sw = Stopwatch.StartNew();
            body();
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        return samples[n / 2];
    }

    private static double MaxRelErr(float[] reference, float[] test)
    {
        if (reference.Length != test.Length) return double.PositiveInfinity;
        double sumSq = 0;
        for (int i = 0; i < reference.Length; i++) sumSq += (double)reference[i] * reference[i];
        double rms = Math.Sqrt(sumSq / Math.Max(1, reference.Length)) + 1e-9;
        double max = 0;
        for (int i = 0; i < reference.Length; i++)
            max = Math.Max(max, Math.Abs(reference[i] - test[i]) / rms);
        return max;
    }
}
