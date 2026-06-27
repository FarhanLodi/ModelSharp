using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// Thread-safety / concurrency hardening for the engine and the newly-multithreaded CPU
/// kernels (MatMul, MultiHeadAttention, LayerNormalization, Conv all use
/// <c>System.Threading.Tasks.Parallel.For</c> over disjoint output regions).
///
/// <para><b>How these prove the absence of data races:</b> the CPU forward pass is fully
/// deterministic — the same feeds always yield bit-identical outputs (the SIMD/parallel
/// kernels partition <i>output</i> rows across threads, so accumulation order within any
/// single output element never changes regardless of thread count). We therefore:</para>
/// <list type="number">
///   <item>compute a <b>serial reference</b> output once (single-threaded call), then</item>
///   <item>hammer the SAME <see cref="ManagedCpuEngine"/> instance from many threads at once
///   with the same feeds and assert every result is <b>bit-exact</b> to the reference. Any
///   shared-state corruption (a kernel scratch buffer, the engine env dictionary, a registry
///   entry) would surface as a diverged value, a thrown exception, or a hang.</item>
///   <item>also interleave <b>different</b> feeds across threads on the one engine and assert
///   each matches that feed's own serial reference — catching cross-talk where one call's
///   intermediate state leaks into another's.</item>
///   <item>run <b>N independent engines/models</b> concurrently, each verified against its own
///   serial reference, to exercise per-model state (registry/kernel construction) under load.</item>
/// </list>
///
/// <para>Each parallel section is wrapped so a hang (deadlock / livelock) fails the test via a
/// wall-clock timeout rather than blocking the suite forever.</para>
///
/// <para>The graph is sized <b>above</b> the kernels' parallelization thresholds
/// (<c>1&lt;&lt;16</c> for MatMul/Conv/LayerNorm, <c>1&lt;&lt;18</c> for attention) so worker
/// threads actually spin up — otherwise the serial fallback would hide a real race.</para>
/// </summary>
public class ConcurrencyTests
{
    private readonly ITestOutputHelper _out;
    public ConcurrencyTests(ITestOutputHelper output) => _out = output;

    // Wall-clock budget for each parallel fan-out. Generous (the work itself is small); its job
    // is purely to convert a deadlock/livelock into a test failure instead of an infinite hang.
    private static readonly TimeSpan HangTimeout = TimeSpan.FromSeconds(60);

    private static Tensor<float> Rand(Random rng, params int[] dims)
    {
        long n = 1; foreach (int d in dims) n *= d;
        var data = new float[n];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return Tensor<float>.FromArray(new TensorShape(dims), data);
    }

    // ---------------------------------------------------------------------------------------
    // A non-trivial graph that drives EVERY parallelized kernel above its threshold:
    //   x:[B,S,H]  --LayerNorm-->  ln
    //   ln @ Wqkv:[H, H] (MatMul, M=B*S=128 rows, K=256, N=256 -> 128*256*256 >> 1<<16)
    //   the projection feeds a MultiHeadAttention (num_heads heads, inner MACs >> 1<<18)
    //   attention out @ Wo:[H,H] (MatMul)
    //   plus a Conv branch on a separate 4-D input (output MACs >> 1<<16)
    // Outputs are concatenated-by-name so a race in any kernel perturbs the final dictionary.
    // ---------------------------------------------------------------------------------------
    private const int Batch = 2, Seq = 64, Hidden = 256, Heads = 8; // B*S = 128 rows

    private static ModelGraph BuildKernelStressGraph(Random rng)
    {
        // Initializers (shared, read-only weights — exactly the kind of state a race could corrupt).
        var wQkv = Rand(rng, Hidden, Hidden);
        var wO = Rand(rng, Hidden, Hidden);
        var lnScale = Rand(rng, Hidden);
        var lnBias = Rand(rng, Hidden);

        // Conv branch weights: out=16, in=8, 3x3; input below is [1,8,24,24] -> ~ 16*22*22*8*9 MACs.
        var convW = Rand(rng, 16, 8, 3, 3);
        var convB = Rand(rng, 16);

        var inits = new Dictionary<string, Tensor>
        {
            ["Wqkv"] = wQkv,
            ["Wo"] = wO,
            ["ln_scale"] = lnScale,
            ["ln_bias"] = lnBias,
            ["conv_w"] = convW,
            ["conv_b"] = convB,
        };

        var nodes = new[]
        {
            // LayerNorm over the last (hidden) axis: outer = B*S = 128 rows, norm = 256 -> 32768.
            new GraphNode("LayerNormalization", "ln", new[] { "x", "ln_scale", "ln_bias" },
                new[] { "lned" }, new Dictionary<string, object> { ["axis"] = -1L }),

            // Projection MatMul: [B,S,H] @ [H,H] -> [B,S,H]. 128*256*256 MACs, well above 1<<16.
            new GraphNode("MatMul", "proj", new[] { "lned", "Wqkv" }, new[] { "qkv" }),

            // Self-attention on the projection (q=k=v=qkv). Inner MACs = B*heads*S*S*headDim huge.
            new GraphNode("MultiHeadAttention", "mha", new[] { "qkv", "qkv", "qkv" },
                new[] { "attn" }, new Dictionary<string, object> { ["num_heads"] = (long)Heads }),

            // Output projection MatMul.
            new GraphNode("MatMul", "out_proj", new[] { "attn", "Wo" }, new[] { "y_attn" }),

            // Independent Conv branch on a 4-D feed.
            new GraphNode("Conv", "conv", new[] { "img", "conv_w", "conv_b" }, new[] { "y_conv" },
                new Dictionary<string, object> { ["kernel_shape"] = new long[] { 3, 3 } }),
        };

        return new ModelGraph
        {
            Inputs = new[] { "x", "img" },
            Outputs = new[] { "y_attn", "y_conv" },
            Nodes = nodes,
            Initializers = inits,
        };
    }

    private static Dictionary<string, NamedTensor> MakeFeeds(Random rng)
        => new()
        {
            ["x"] = new NamedTensor("x", Rand(rng, Batch, Seq, Hidden)),
            ["img"] = new NamedTensor("img", Rand(rng, 1, 8, 24, 24)),
        };

    private static (float[] yAttn, float[] yConv) RunOnce(
        ManagedCpuEngine engine, IReadOnlyDictionary<string, NamedTensor> feeds)
    {
        IReadOnlyDictionary<string, NamedTensor> outs = engine.Run(feeds);
        return (outs["y_attn"].Data.Span.ToArray(), outs["y_conv"].Data.Span.ToArray());
    }

    private static void AssertBitExact(string what, (float[] a, float[] c) expected, (float[] a, float[] c) got)
    {
        Assert.Equal(expected.a.Length, got.a.Length);
        Assert.Equal(expected.c.Length, got.c.Length);
        // Bit-exact: a data race would almost certainly perturb at least one value; we demand
        // EXACT equality (the forward pass is deterministic) so even a 1-ULP corruption fails.
        for (int i = 0; i < expected.a.Length; i++)
            Assert.True(expected.a[i].Equals(got.a[i]),
                $"{what}: y_attn[{i}] expected {expected.a[i]:R} got {got.a[i]:R}");
        for (int i = 0; i < expected.c.Length; i++)
            Assert.True(expected.c[i].Equals(got.c[i]),
                $"{what}: y_conv[{i}] expected {expected.c[i]:R} got {got.c[i]:R}");
    }

    /// <summary>Runs <paramref name="body"/> and fails (instead of hanging) if it overruns the budget.</summary>
    private void WithHangGuard(string what, Action body)
    {
        Exception? captured = null;
        var t = new Thread(() =>
        {
            try { body(); }
            catch (Exception ex) { captured = ex; }
        }) { IsBackground = true, Name = what };

        t.Start();
        bool finished = t.Join(HangTimeout);
        Assert.True(finished, $"{what}: did not complete within {HangTimeout.TotalSeconds:F0}s — " +
            "likely a deadlock/livelock (thread-safety bug).");
        if (captured is not null)
            throw new Xunit.Sdk.XunitException($"{what} threw: {captured}");
    }

    // ---------------------------------------------------------------------------------------

    [Fact]
    public void SameEngine_SameFeeds_ManyThreads_AllBitExactToSerialReference()
    {
        var rng = new Random(20260627);
        ModelGraph graph = BuildKernelStressGraph(rng);
        using var engine = new ManagedCpuEngine(graph);

        var feeds = MakeFeeds(new Random(1));

        // Serial reference (single thread).
        (float[], float[]) reference = RunOnce(engine, feeds);

        const int Threads = 64;
        var results = new (float[], float[])[Threads];

        WithHangGuard("SameEngine/SameFeeds x64", () =>
            Parallel.For(0, Threads,
                new ParallelOptions { MaxDegreeOfParallelism = Threads },
                i => results[i] = RunOnce(engine, feeds)));

        for (int i = 0; i < Threads; i++)
            AssertBitExact($"thread {i}", reference, results[i]);

        _out.WriteLine($"64 concurrent Run() calls on one engine: all bit-exact to serial reference " +
            $"(y_attn={reference.Item1.Length} floats, y_conv={reference.Item2.Length} floats).");
    }

    [Fact]
    public void SameEngine_DifferentFeeds_Interleaved_EachMatchesOwnSerialReference()
    {
        var rng = new Random(7);
        ModelGraph graph = BuildKernelStressGraph(rng);
        using var engine = new ManagedCpuEngine(graph);

        // A pool of distinct feed sets; each gets its own serial reference computed up front
        // (serially, before any concurrency), so cross-talk between concurrent calls is detectable.
        const int Variants = 8;
        var feedSets = new Dictionary<string, NamedTensor>[Variants];
        var refs = new (float[], float[])[Variants];
        for (int v = 0; v < Variants; v++)
        {
            feedSets[v] = MakeFeeds(new Random(1000 + v));
            refs[v] = RunOnce(engine, feedSets[v]);
        }

        const int Threads = 96;
        var got = new (float[], float[])[Threads];
        var pickedVariant = new int[Threads];

        WithHangGuard("SameEngine/DifferentFeeds interleaved", () =>
            Parallel.For(0, Threads,
                new ParallelOptions { MaxDegreeOfParallelism = Threads },
                i =>
                {
                    int v = i % Variants;
                    pickedVariant[i] = v;
                    got[i] = RunOnce(engine, feedSets[v]);
                }));

        for (int i = 0; i < Threads; i++)
            AssertBitExact($"thread {i} (variant {pickedVariant[i]})", refs[pickedVariant[i]], got[i]);

        _out.WriteLine($"96 concurrent Run() calls interleaving {Variants} distinct feed sets: " +
            "each matched its own serial reference (no cross-talk).");
    }

    [Fact]
    public void NDifferentEngines_RunConcurrently_EachMatchesItsOwnSerialReference()
    {
        const int Models = 16;
        var engines = new ManagedCpuEngine[Models];
        var feeds = new Dictionary<string, NamedTensor>[Models];
        var refs = new (float[], float[])[Models];

        try
        {
            for (int m = 0; m < Models; m++)
            {
                // Distinct weights + feeds per model -> distinct reference outputs.
                var rng = new Random(500 + m);
                engines[m] = new ManagedCpuEngine(BuildKernelStressGraph(rng));
                feeds[m] = MakeFeeds(new Random(9000 + m));
                refs[m] = RunOnce(engines[m], feeds[m]);
            }

            var got = new (float[], float[])[Models];
            WithHangGuard("N different engines concurrently", () =>
                Parallel.For(0, Models,
                    new ParallelOptions { MaxDegreeOfParallelism = Models },
                    m => got[m] = RunOnce(engines[m], feeds[m])));

            for (int m = 0; m < Models; m++)
                AssertBitExact($"model {m}", refs[m], got[m]);

            // Sanity: the models genuinely differ (otherwise the test would be vacuous).
            Assert.False(refs[0].Item1.SequenceEqual(refs[1].Item1),
                "expected distinct models to produce distinct outputs");

            _out.WriteLine($"{Models} distinct engines run concurrently: each bit-exact to its own reference.");
        }
        finally
        {
            foreach (ManagedCpuEngine? e in engines) e?.Dispose();
        }
    }

    [Fact]
    public void RepeatedConcurrentRuns_AreStable_AcrossManyRounds()
    {
        // Stress repetition: a race may only manifest occasionally, so run many rounds. Each
        // round fans out and every result must stay bit-exact to the single serial reference.
        var rng = new Random(42);
        ModelGraph graph = BuildKernelStressGraph(rng);
        using var engine = new ManagedCpuEngine(graph);
        var feeds = MakeFeeds(new Random(3));
        (float[], float[]) reference = RunOnce(engine, feeds);

        const int Rounds = 20, Fanout = 32;
        WithHangGuard("repeated concurrent rounds", () =>
        {
            for (int round = 0; round < Rounds; round++)
            {
                var got = new (float[], float[])[Fanout];
                Parallel.For(0, Fanout,
                    new ParallelOptions { MaxDegreeOfParallelism = Fanout },
                    i => got[i] = RunOnce(engine, feeds));
                for (int i = 0; i < Fanout; i++)
                    AssertBitExact($"round {round} thread {i}", reference, got[i]);
            }
        });

        _out.WriteLine($"{Rounds} rounds x {Fanout} concurrent runs: stable / bit-exact throughout.");
    }
}
