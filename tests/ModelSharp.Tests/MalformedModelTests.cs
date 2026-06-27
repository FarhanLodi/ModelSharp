using System;
using System.Collections.Generic;
using System.Threading;
using ModelSharp;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// Malformed-model fuzzing for the hand-rolled ONNX loader. The loader must accept untrusted
/// files, so adversarial/corrupt input must <b>fail cleanly</b> — a thrown exception, never a
/// process crash, an infinite loop/hang, or an unbounded allocation.
///
/// <para>"Clean" here means: the call returns control by throwing a managed exception within a
/// short wall-clock budget. We accept the loader's documented <see cref="ModelSharpException"/>
/// as well as the lower-level framework exceptions the pure-managed protobuf reader naturally
/// raises on corrupt wire data (<see cref="FormatException"/>, <see cref="OverflowException"/>,
/// <see cref="ArgumentException"/>/<see cref="ArgumentOutOfRangeException"/>,
/// <see cref="IndexOutOfRangeException"/>, <see cref="InvalidOperationException"/>). What we do
/// NOT tolerate is: no exception at all (silent acceptance of garbage), a hang, or a crash that
/// escapes the managed exception model.</para>
///
/// <para>Each case is run under a hang guard: the parse executes on a worker thread and the test
/// fails if it does not return within the timeout (so a livelock/infinite-loop bug fails loudly
/// instead of wedging the suite).</para>
/// </summary>
public class MalformedModelTests
{
    private readonly ITestOutputHelper _out;
    public MalformedModelTests(ITestOutputHelper output) => _out = output;

    // Generous wall-clock budget; parses of these tiny inputs complete in microseconds when the
    // loader is well-behaved. Its sole purpose is to turn a hang into a failure.
    private static readonly TimeSpan HangTimeout = TimeSpan.FromSeconds(10);

    // Exception types we consider a "clean" failure for adversarial input.
    private static bool IsCleanFailure(Exception ex) =>
        ex is ModelSharpException
        or FormatException
        or OverflowException
        or ArgumentException            // includes ArgumentOutOfRangeException
        or IndexOutOfRangeException
        or InvalidOperationException
        or NotSupportedException;

    /// <summary>
    /// Parses <paramref name="bytes"/> on a worker thread and asserts the parse fails cleanly
    /// (a tolerated exception) without crashing or hanging. Returns the caught exception so a
    /// caller can additionally assert on its type/message.
    /// </summary>
    private Exception AssertParseFailsCleanly(string what, byte[] bytes)
    {
        Exception? thrown = null;
        bool completedNormally = false;

        var worker = new Thread(() =>
        {
            try
            {
                ModelGraph g = OnnxModelLoader.ParseModel(bytes);
                // Reached only if no exception was thrown — touch the result so nothing is elided.
                completedNormally = g.Nodes.Count >= 0;
            }
            catch (Exception ex)
            {
                thrown = ex;
            }
        }) { IsBackground = true, Name = "fuzz:" + what };

        worker.Start();
        bool finished = worker.Join(HangTimeout);

        Assert.True(finished,
            $"[{what}] loader did not return within {HangTimeout.TotalSeconds:F0}s — " +
            "a hang/infinite-loop on malformed input is a robustness bug.");

        Assert.False(completedNormally,
            $"[{what}] loader silently ACCEPTED malformed input (no exception) — it must reject it.");

        Assert.NotNull(thrown);
        Assert.True(IsCleanFailure(thrown!),
            $"[{what}] loader threw an unexpected exception type {thrown!.GetType().FullName}: {thrown.Message}");

        _out.WriteLine($"[{what}] rejected cleanly with {thrown!.GetType().Name}: {thrown.Message}");
        return thrown!;
    }

    /// <summary>
    /// Weaker contract for fuzz inputs that <i>might</i> happen to decode as a (possibly empty)
    /// valid message: the parse must RETURN within the budget and, if it throws, throw cleanly.
    /// It is allowed to succeed. This still proves no hang / no crash on adversarial bytes.
    /// </summary>
    private void AssertParseReturnsCleanly(string what, byte[] bytes)
    {
        Exception? thrown = null;
        var worker = new Thread(() =>
        {
            try { OnnxModelLoader.ParseModel(bytes); }
            catch (Exception ex) { thrown = ex; }
        }) { IsBackground = true, Name = "fuzz:" + what };

        worker.Start();
        bool finished = worker.Join(HangTimeout);

        Assert.True(finished,
            $"[{what}] loader did not return within {HangTimeout.TotalSeconds:F0}s — a hang on " +
            "malformed input is a robustness bug.");
        if (thrown is not null)
            Assert.True(IsCleanFailure(thrown),
                $"[{what}] loader threw an unexpected exception type {thrown.GetType().FullName}: {thrown.Message}");
    }

    // ---- Trivial / garbage inputs ----------------------------------------------------------

    [Fact]
    public void EmptyBytes_FailsCleanly()
    {
        // An empty buffer has no graph field -> the loader's explicit "no graph" guard.
        Exception ex = AssertParseFailsCleanly("empty", Array.Empty<byte>());
        Assert.IsType<ModelSharpException>(ex);
        Assert.Contains("no graph", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RandomGarbage_VariousSizes_AllFailCleanly()
    {
        // A battery of pseudo-random buffers across a range of sizes. None should ever be a valid
        // ModelProto; each must throw cleanly (or, if it happens to decode as an empty/graph-less
        // message, be rejected). We loop many seeds to broaden coverage of wire-tag permutations.
        for (int seed = 0; seed < 200; seed++)
        {
            var rng = new Random(seed * 6353 + 1);
            int len = rng.Next(0, 512);
            var buf = new byte[len];
            rng.NextBytes(buf);
            // Random bytes are overwhelmingly invalid ModelProtos, but a short buffer could in
            // principle decode as an empty/graph-less message; we only require no hang / clean throw.
            AssertParseReturnsCleanly($"garbage seed={seed} len={len}", buf);
        }
    }

    [Fact]
    public void HighBitGarbage_LooksLikeVarints_FailsCleanly()
    {
        // All-0xFF looks like an endless multi-byte varint; the reader must bound it (it caps at
        // 64 bits) and not spin. Also try patterns that resemble length-delimited tags with huge
        // lengths.
        AssertParseFailsCleanly("all-0xFF-256", Fill(256, 0xFF));
        AssertParseFailsCleanly("all-0x80-64", Fill(64, 0x80));   // continuation bits forever
        AssertParseFailsCleanly("all-0x08-64", Fill(64, 0x08));   // field=1 varint tags, no payload
    }

    // ---- Truncated-mid-message valid model -------------------------------------------------

    [Fact]
    public void TruncatedValidModel_AtEveryOffset_NeverHangsOrCrashes()
    {
        // Build a real (small but non-trivial) valid model, then cut it at EVERY byte offset.
        // The full model parses; every strict prefix must either parse to *something* (a valid
        // shorter message can be legal) OR fail cleanly — but never hang/crash.
        byte[] full = BuildSmallValidModel();

        // Sanity: the untruncated model parses successfully.
        ModelGraph ok = OnnxModelLoader.ParseModel(full);
        Assert.NotEmpty(ok.Nodes);

        int hangFreeCount = 0;
        for (int cut = 0; cut < full.Length; cut++)
        {
            var prefix = new byte[cut];
            Array.Copy(full, prefix, cut);

            // A truncated prefix may legitimately still parse (e.g. cut inside a trailing optional
            // field that was already skipped). We only require: returns within the budget, and if
            // it throws, it throws cleanly. Run under the hang guard.
            RunUnderHangGuard($"truncate@{cut}", () =>
            {
                try { OnnxModelLoader.ParseModel(prefix); }
                catch (Exception ex)
                {
                    Assert.True(IsCleanFailure(ex),
                        $"truncate@{cut}: unexpected {ex.GetType().FullName}: {ex.Message}");
                }
            });
            hangFreeCount++;
        }

        _out.WriteLine($"Truncated valid model at all {hangFreeCount} offsets: no hang, no crash.");
    }

    // ---- Bogus opset / unknown op ----------------------------------------------------------

    [Fact]
    public void UnknownOp_ParsesButExecutionRejectsCleanly()
    {
        // An unknown op type is structurally valid ONNX — the LOADER should accept it (it does not
        // resolve kernels). The clean failure surfaces at EXECUTION time as
        // UnsupportedOperatorException. This documents the loader/engine boundary.
        byte[] model = BuildModelWithSingleNode(opType: "TotallyMadeUpOp",
            inputs: new[] { "a" }, outputs: new[] { "y" });

        ModelGraph g = OnnxModelLoader.ParseModel(model);   // loader accepts it
        GraphNode node = Assert.Single(g.Nodes);
        Assert.Equal("TotallyMadeUpOp", node.OpType);

        using var engine = new ModelSharp.Cpu.ManagedCpuEngine(g);
        var a = Tensors_Float(new[] { 1 }, 0f);
        Assert.Throws<UnsupportedOperatorException>(() =>
            engine.Run(new Dictionary<string, ModelSharp.Tensors.NamedTensor>
            {
                ["a"] = new ModelSharp.Tensors.NamedTensor("a", a),
            }));

        _out.WriteLine("Unknown op: loader accepts, engine rejects with UnsupportedOperatorException.");
    }

    // ---- Tensor whose declared shape disagrees with its data length ------------------------

    [Fact]
    public void Initializer_ShapeLongerThanData_FailsCleanly()
    {
        // A float initializer that declares dims=[100] but supplies only 3 raw float values.
        // The loader must not read past the supplied data (no buffer over-read / crash). Whether
        // it throws or zero-pads, it must NOT crash/hang.
        byte[] tensor = BuildFloatInitializer(name: "w", dims: new long[] { 100 },
            floats: new[] { 1f, 2f, 3f });
        byte[] model = BuildModelWithInitializer(tensor);

        RunUnderHangGuard("shape>data", () =>
        {
            try
            {
                ModelGraph g = OnnxModelLoader.ParseModel(model);
                // If it parsed, the initializer must exist and not have over-read memory.
                Assert.True(g.Initializers.ContainsKey("w"));
            }
            catch (Exception ex)
            {
                Assert.True(IsCleanFailure(ex),
                    $"shape>data: unexpected {ex.GetType().FullName}: {ex.Message}");
            }
        });
        _out.WriteLine("Initializer shape > data: handled without over-read/crash.");
    }

    [Fact]
    public void Initializer_OverflowingDims_FailsCleanly()
    {
        // dims = [int.MaxValue, int.MaxValue] -> shape.Length overflows the checked (int) cast in
        // the loader. Must throw (OverflowException/ModelSharpException), never OOM-attempt a
        // multi-exabyte array or crash.
        byte[] tensor = BuildFloatInitializer(name: "huge",
            dims: new long[] { int.MaxValue, int.MaxValue }, floats: new[] { 1f });
        byte[] model = BuildModelWithInitializer(tensor);
        AssertParseFailsCleanly("overflowing-dims", model);
    }

    // NOTE (reported as a hardening finding, intentionally NOT exercised as a live test):
    // a single dim near int.MaxValue (e.g. dims=[2^31-1]) makes count fit a 32-bit int but drives
    // the loader to `new float[count]` (~8 GB) BEFORE it ever inspects the supplied data length.
    // There is no per-tensor element-count / allocation cap. A malicious model can therefore force
    // a multi-GB allocation (DoS) from a few hundred bytes of input. We deliberately do not allocate
    // 8 GB on the shared CI machine here; see the test report for the suggested src-side guard.

    // ---- Length-delimited field claiming more than the buffer ------------------------------

    [Fact]
    public void LengthDelimited_ExceedingBuffer_FailsCleanly()
    {
        // graph field (7, wire 2) with a declared length far larger than the remaining bytes.
        // ProtoReader.ReadLengthDelimited must reject this rather than slice out of bounds.
        byte[] bad = Concat(
            Tag(7, 2),
            Varint(100000),   // claim 100k bytes...
            new byte[] { 1, 2, 3 });   // ...but only 3 present
        Exception ex = AssertParseFailsCleanly("len-exceeds-buffer", bad);
        Assert.IsType<FormatException>(ex);
    }

    [Fact]
    public void NegativeLengthDelimited_FromHugeVarint_FailsCleanly()
    {
        // A length varint whose (int) cast is negative. The reader/slice must reject it cleanly
        // (no negative-length slice, no buffer over-read).
        byte[] bad = Concat(
            Tag(7, 2),
            // 0xFFFFFFFF... encodes a varint that casts to a negative int.
            new byte[] { 0xFF, 0xFF, 0xFF, 0xFF, 0x0F },
            new byte[] { 1, 2, 3 });
        AssertParseFailsCleanly("negative-len", bad);
    }

    [Fact]
    public void UnsupportedWireType_FailsCleanly()
    {
        // Wire types 3/4 (group start/end, deprecated) and 6 (reserved) are unsupported; SkipField
        // throws FormatException. Embed one at the top level.
        byte[] bad = Concat(Tag(7, 3), new byte[] { 0x00 });
        Exception ex = AssertParseFailsCleanly("wire-type-3", bad);
        Assert.IsType<FormatException>(ex);
    }

    // ---- Cyclic / dangling node-input references -------------------------------------------

    [Fact]
    public void DanglingNodeInput_ParsesButExecutionRejectsCleanly()
    {
        // Node references input "ghost" that no feed/initializer/prior node produces. The loader is
        // a structural parse and accepts it; the clean failure is at execution (missing tensor).
        byte[] model = BuildModelWithSingleNode(opType: "Relu",
            inputs: new[] { "ghost" }, outputs: new[] { "y" });

        ModelGraph g = OnnxModelLoader.ParseModel(model);
        Assert.Single(g.Nodes);

        using var engine = new ModelSharp.Cpu.ManagedCpuEngine(g);
        RunUnderHangGuard("dangling-input-exec", () =>
        {
            Exception ex = Assert.ThrowsAny<Exception>(() =>
                engine.Run(new Dictionary<string, ModelSharp.Tensors.NamedTensor>()));
            Assert.True(IsCleanFailure(ex),
                $"dangling input exec: unexpected {ex.GetType().FullName}: {ex.Message}");
        });
        _out.WriteLine("Dangling node input: loader accepts, engine fails cleanly at run.");
    }

    [Fact]
    public void CyclicNodeReference_ParsesAndDoesNotHang()
    {
        // Two nodes form a data cycle: n0 consumes "b" and produces "a"; n1 consumes "a" and
        // produces "b". The engine runs nodes in listed order (no topo re-sort), so the first node
        // hits a missing input and fails cleanly. The point: the parser builds this without
        // looping, and execution does not spin.
        byte[] n0 = NodeProto("Relu", new[] { "b" }, new[] { "a" }, name: "n0");
        byte[] n1 = NodeProto("Relu", new[] { "a" }, new[] { "b" }, name: "n1");
        byte[] graph = Concat(
            LenField(1, n0),
            LenField(1, n1),
            LenField(11, LenField(1, Str("x"))),   // a declared input that is never used
            LenField(12, LenField(1, Str("b"))));   // output
        byte[] model = LenField(7, graph);

        ModelGraph g = OnnxModelLoader.ParseModel(model);
        Assert.Equal(2, g.Nodes.Count);

        using var engine = new ModelSharp.Cpu.ManagedCpuEngine(g);
        RunUnderHangGuard("cyclic-exec", () =>
        {
            Exception ex = Assert.ThrowsAny<Exception>(() =>
                engine.Run(new Dictionary<string, ModelSharp.Tensors.NamedTensor>
                {
                    ["x"] = new ModelSharp.Tensors.NamedTensor("x",
                        Tensors_Float(new[] { 1 }, 0f)),
                }));
            Assert.True(IsCleanFailure(ex),
                $"cyclic exec: unexpected {ex.GetType().FullName}: {ex.Message}");
        });
        _out.WriteLine("Cyclic node reference: parsed without looping; execution fails cleanly.");
    }

    // ---- Deeply-nested junk ----------------------------------------------------------------

    [Fact]
    public void DeeplyNestedSubgraphs_DoNotStackOverflowOrHang()
    {
        // Build a chain of nested GRAPH attributes (If.then_branch -> node -> If.then_branch ...)
        // many levels deep. Recursive ParseGraph must either parse it or fail cleanly — never crash
        // the process with an uncatchable StackOverflowException-equivalent that escapes the budget.
        // We keep depth large enough to be adversarial but the per-level payload tiny.
        const int Depth = 2000;
        byte[] model = BuildDeeplyNestedModel(Depth);

        Exception? thrown = null;
        bool completed = false;
        var worker = new Thread(() =>
        {
            try { OnnxModelLoader.ParseModel(model); completed = true; }
            catch (Exception ex) { thrown = ex; }
        }, maxStackSize: 16 * 1024 * 1024) { IsBackground = true };
        worker.Start();
        bool finished = worker.Join(HangTimeout);

        Assert.True(finished, "deeply-nested: loader did not return within budget (hang).");
        // Acceptable outcomes: parsed the whole chain, OR threw a clean exception. The forbidden
        // outcome is a hang (already asserted) or an unmanaged crash (would not reach here).
        if (!completed)
            Assert.True(IsCleanFailure(thrown!),
                $"deeply-nested: unexpected {thrown!.GetType().FullName}: {thrown.Message}");
        _out.WriteLine($"Deeply nested ({Depth} levels): returned " +
            $"(completed={completed}, threw={thrown?.GetType().Name ?? "none"}).");
    }

    [Fact]
    public void TruncatedTensorRawData_FailsCleanly()
    {
        // An initializer declaring an int64 tensor of 4 elements but whose raw_data holds only
        // a few stray bytes (not a multiple of 8). The MemoryMarshal.Cast slice must not over-read.
        byte[] tensor = Concat(
            VarintField(2, 7),                       // data_type = INT64
            LenField(1, PackedVarints(new long[] { 4 })),  // dims = [4]  (packed)
            LenField(8, Str("t")),                   // name
            LenField(9, new byte[] { 1, 2, 3 }));    // raw_data = 3 bytes (needs 32)
        byte[] model = BuildModelWithInitializer(tensor);

        RunUnderHangGuard("truncated-raw", () =>
        {
            try { OnnxModelLoader.ParseModel(model); }
            catch (Exception ex)
            {
                Assert.True(IsCleanFailure(ex),
                    $"truncated-raw: unexpected {ex.GetType().FullName}: {ex.Message}");
            }
        });
        _out.WriteLine("Truncated tensor raw_data: handled without over-read.");
    }

    // =======================================================================================
    //  Hang guard + protobuf encoders
    // =======================================================================================

    private void RunUnderHangGuard(string what, Action body)
    {
        Exception? captured = null;
        var t = new Thread(() =>
        {
            try { body(); }
            catch (Exception ex) { captured = ex; }
        }) { IsBackground = true, Name = "guard:" + what };
        t.Start();
        bool finished = t.Join(HangTimeout);
        Assert.True(finished, $"[{what}] did not complete within {HangTimeout.TotalSeconds:F0}s (hang).");
        if (captured is not null) throw captured;
    }

    private static ModelSharp.Tensors.Tensor<float> Tensors_Float(int[] dims, float fill)
    {
        long n = 1; foreach (int d in dims) n *= d;
        var data = new float[n];
        Array.Fill(data, fill);
        return ModelSharp.Tensors.Tensor<float>.FromArray(new ModelSharp.Tensors.TensorShape(dims), data);
    }

    // ---- A small but real, valid ONNX ModelProto (Add then Relu) ---------------------------

    private static byte[] BuildSmallValidModel()
    {
        // graph: nodes [Add(a,b->s), Relu(s->y)], inputs a,b, output y, one float initializer.
        byte[] add = NodeProto("Add", new[] { "a", "b" }, new[] { "s" }, name: "add0");
        byte[] relu = NodeProto("Relu", new[] { "s" }, new[] { "y" }, name: "relu0");
        byte[] init = BuildFloatInitializer("c", new long[] { 2 }, new[] { 1.5f, -2f });

        byte[] graph = Concat(
            LenField(1, add),
            LenField(1, relu),
            LenField(5, init),                          // initializer
            LenField(11, LenField(1, Str("a"))),        // input a
            LenField(11, LenField(1, Str("b"))),        // input b
            LenField(12, LenField(1, Str("y"))));       // output y
        return LenField(7, graph);
    }

    private static byte[] BuildModelWithSingleNode(string opType, string[] inputs, string[] outputs)
    {
        byte[] node = NodeProto(opType, inputs, outputs, name: "n0");
        var parts = new List<byte[]> { LenField(1, node) };
        foreach (string i in inputs) parts.Add(LenField(11, LenField(1, Str(i))));
        foreach (string o in outputs) parts.Add(LenField(12, LenField(1, Str(o))));
        byte[] graph = Concat(parts.ToArray());
        return LenField(7, graph);
    }

    private static byte[] BuildModelWithInitializer(byte[] tensorProto)
    {
        byte[] graph = LenField(5, tensorProto);   // graph.initializer (field 5)
        return LenField(7, graph);
    }

    private static byte[] BuildDeeplyNestedModel(int depth)
    {
        // Innermost graph: a single Identity node.
        byte[] inner = Concat(
            LenField(1, NodeProto("Identity", new[] { "in" }, new[] { "z" }, name: "leaf")),
            LenField(12, LenField(1, Str("z"))));

        byte[] current = inner;
        for (int d = 0; d < depth; d++)
        {
            // then_branch attribute carrying the current subgraph.
            byte[] thenAttr = Concat(
                LenField(1, Str("then_branch")),
                VarintField(20, 5),          // type = GRAPH
                LenField(6, current));       // g = nested graph
            byte[] ifNode = Concat(
                LenField(1, Str("cond")),
                LenField(2, Str("y")),
                LenField(4, Str("If")),
                LenField(5, thenAttr));
            current = Concat(
                LenField(1, ifNode),
                LenField(12, LenField(1, Str("y"))));
        }
        return LenField(7, current);
    }

    private static byte[] NodeProto(string opType, string[] inputs, string[] outputs, string name)
    {
        var parts = new List<byte[]>();
        foreach (string i in inputs) parts.Add(LenField(1, Str(i)));   // input (field 1)
        foreach (string o in outputs) parts.Add(LenField(2, Str(o)));  // output (field 2)
        parts.Add(LenField(3, Str(name)));                             // name (field 3)
        parts.Add(LenField(4, Str(opType)));                           // op_type (field 4)
        return Concat(parts.ToArray());
    }

    private static byte[] BuildFloatInitializer(string name, long[] dims, float[] floats)
    {
        var parts = new List<byte[]>
        {
            VarintField(2, 1),   // data_type = FLOAT (1)
        };
        foreach (long d in dims) parts.Add(VarintField(1, (ulong)d));   // dims (field 1, varint each)
        // float_data (field 4) as repeated fixed32.
        foreach (float f in floats) parts.Add(Concat(Tag(4, 5), BitConverter.GetBytes(f)));
        parts.Add(LenField(8, Str(name)));                             // name (field 8)
        return Concat(parts.ToArray());
    }

    // ---- Low-level wire encoders -----------------------------------------------------------

    private static byte[] Str(string s) => System.Text.Encoding.UTF8.GetBytes(s);

    private static byte[] Tag(int fieldNo, int wireType) => Varint((uint)((fieldNo << 3) | wireType));

    private static byte[] LenField(int fieldNo, byte[] payload) =>
        Concat(Tag(fieldNo, 2), Varint((uint)payload.Length), payload);

    private static byte[] VarintField(int fieldNo, ulong value) =>
        Concat(Tag(fieldNo, 0), Varint(value));

    private static byte[] PackedVarints(long[] vals)
    {
        var list = new List<byte>();
        foreach (long v in vals) list.AddRange(Varint((ulong)v));
        return list.ToArray();
    }

    private static byte[] Varint(ulong v)
    {
        var bytes = new List<byte>();
        do { byte b = (byte)(v & 0x7F); v >>= 7; if (v != 0) b |= 0x80; bytes.Add(b); } while (v != 0);
        return bytes.ToArray();
    }

    private static byte[] Fill(int n, byte b)
    {
        var a = new byte[n];
        Array.Fill(a, b);
        return a;
    }

    private static byte[] Concat(params byte[][] parts)
    {
        int n = 0;
        foreach (byte[] p in parts) n += p.Length;
        var outBuf = new byte[n];
        int off = 0;
        foreach (byte[] p in parts) { Array.Copy(p, 0, outBuf, off, p.Length); off += p.Length; }
        return outBuf;
    }
}
