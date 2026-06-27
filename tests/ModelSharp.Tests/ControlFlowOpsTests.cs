using System.Collections.Generic;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Tests for the ONNX control-flow ops (If / Loop / Scan) and the subgraph executor seam.
/// Graphs and their nested subgraphs are built in-memory as <see cref="ModelGraph"/>s; a
/// subgraph is stored as a GRAPH attribute (a <c>ModelGraph</c> value) on the control-flow node.
/// </summary>
public class ControlFlowOpsTests
{
    private static Tensor<float> Scalar(float v) => new(new TensorShape(), new[] { v });
    private static Tensor<float> Vec(params float[] v) => new(new TensorShape(v.Length), v);
    private static Tensor<bool> BoolScalar(bool v) => new(new TensorShape(), new[] { v });
    private static Tensor<long> LongScalar(long v) => new(new TensorShape(), new[] { v });

    private static IReadOnlyDictionary<string, NamedTensor> Run(
        ModelGraph g, Dictionary<string, NamedTensor> feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        return engine.Run(feeds);
    }

    // ---- If --------------------------------------------------------------------------------

    [Theory]
    [InlineData(true, 5f)]   // then = a + b = 2 + 3
    [InlineData(false, 6f)]  // else = a * b = 2 * 3
    public void If_Selects_Branch_By_Cond(bool cond, float expected)
    {
        // then_branch: out = Add(a, b);  else_branch: out = Mul(a, b)
        // Both branches capture the outer-scope tensors "a" and "b" (no formal inputs).
        var thenBranch = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "t_out" },
            Nodes = new[] { new GraphNode("Add", "t_add", new[] { "a", "b" }, new[] { "t_out" }) },
        };
        var elseBranch = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "e_out" },
            Nodes = new[] { new GraphNode("Mul", "e_mul", new[] { "a", "b" }, new[] { "e_out" }) },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "cond", "a", "b" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("If", "if0", new[] { "cond" }, new[] { "y" },
                    new Dictionary<string, object>
                    {
                        ["then_branch"] = thenBranch,
                        ["else_branch"] = elseBranch,
                    }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["cond"] = new NamedTensor("cond", BoolScalar(cond)),
            ["a"] = new NamedTensor("a", Scalar(2f)),
            ["b"] = new NamedTensor("b", Scalar(3f)),
        });

        Assert.Equal(expected, outputs["y"].Data.Span[0]);
    }

    // ---- Loop ------------------------------------------------------------------------------

    [Fact]
    public void Loop_Sums_1_To_N_Via_Carried_Value()
    {
        // Carried accumulator "acc" starts at 0. Each iteration:
        //   one = 1 (initializer), step = iter + one ; acc_next = acc + step ; cond_out = true
        // After N iterations acc = sum(1..N).
        var body = new ModelGraph
        {
            Inputs = new[] { "iter", "cond_in", "acc" },
            // Re-emit the incoming condition as the loop's cond_out (a subgraph output may name
            // one of its inputs directly), avoiding an Identity on a Boolean tensor.
            Outputs = new[] { "cond_in", "acc_out" },
            Initializers = new Dictionary<string, Tensor> { ["one"] = LongScalar(1) },
            Nodes = new[]
            {
                // iter is i64; step = iter + 1
                new GraphNode("Add", "b_step", new[] { "iter", "one" }, new[] { "step_i64" }),
                new GraphNode("Cast", "b_cast", new[] { "step_i64" }, new[] { "step" },
                    new Dictionary<string, object> { ["to"] = (long)1 /* FLOAT */ }),
                new GraphNode("Add", "b_acc", new[] { "acc", "step" }, new[] { "acc_out" }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "M", "cond", "acc_init" },
            Outputs = new[] { "acc_final" },
            Nodes = new[]
            {
                new GraphNode("Loop", "loop0",
                    new[] { "M", "cond", "acc_init" }, new[] { "acc_final" },
                    new Dictionary<string, object> { ["body"] = body }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["M"] = new NamedTensor("M", LongScalar(5)),
            ["cond"] = new NamedTensor("cond", BoolScalar(true)),
            ["acc_init"] = new NamedTensor("acc_init", Scalar(0f)),
        });

        // sum(1..5) = 15
        Assert.Equal(15f, outputs["acc_final"].Data.Span[0]);
    }

    [Fact]
    public void Loop_Accumulates_Scan_Output_Stacked()
    {
        // Body emits a scan output = iter (as float) each step; with no carried values the
        // node produces a single stacked scan output of shape [N].
        var body = new ModelGraph
        {
            Inputs = new[] { "iter", "cond_in" },
            // cond_out re-emits the incoming condition (no Identity on a Boolean tensor).
            Outputs = new[] { "cond_in", "scan_val" },
            Nodes = new[]
            {
                new GraphNode("Cast", "b_cast", new[] { "iter" }, new[] { "scan_val" },
                    new Dictionary<string, object> { ["to"] = (long)1 /* FLOAT */ }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "M", "cond" },
            Outputs = new[] { "scan_out" },
            Nodes = new[]
            {
                new GraphNode("Loop", "loop0", new[] { "M", "cond" }, new[] { "scan_out" },
                    new Dictionary<string, object> { ["body"] = body }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["M"] = new NamedTensor("M", LongScalar(4)),
            ["cond"] = new NamedTensor("cond", BoolScalar(true)),
        });

        // iters 0,1,2,3 stacked along a new leading axis.
        Tensor<float> scan = outputs["scan_out"].Data;
        Assert.Equal(new[] { 4 }, scan.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 0f, 1f, 2f, 3f }, scan.Span.ToArray());
    }

    [Fact]
    public void Loop_Terminates_Early_On_Cond_False()
    {
        // Stop once acc >= 3 (cond_out = acc_out < threshold). Counts iterations into a scan out.
        var body = new ModelGraph
        {
            Inputs = new[] { "iter", "cond_in", "acc" },
            Outputs = new[] { "cond_out", "acc_out" },
            Initializers = new Dictionary<string, Tensor>
            {
                ["one"] = Scalar(1f),
                ["limit"] = Scalar(3f),
            },
            Nodes = new[]
            {
                new GraphNode("Add", "b_acc", new[] { "acc", "one" }, new[] { "acc_out" }),
                new GraphNode("Less", "b_cmp", new[] { "acc_out", "limit" }, new[] { "cond_out" }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "M", "cond", "acc_init" },
            Outputs = new[] { "acc_final" },
            Nodes = new[]
            {
                new GraphNode("Loop", "loop0",
                    new[] { "M", "cond", "acc_init" }, new[] { "acc_final" },
                    new Dictionary<string, object> { ["body"] = body }),
            },
        };

        // M is large; cond termination should kick in first (acc reaches 3).
        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["M"] = new NamedTensor("M", LongScalar(100)),
            ["cond"] = new NamedTensor("cond", BoolScalar(true)),
            ["acc_init"] = new NamedTensor("acc_init", Scalar(0f)),
        });

        Assert.Equal(3f, outputs["acc_final"].Data.Span[0]);
    }

    // ---- Scan ------------------------------------------------------------------------------

    [Fact]
    public void Scan_Cumulative_Sum_Over_Sequence()
    {
        // State "sum" threads the running total; scan input "x" is sliced per step.
        // Body: sum_out = sum + x_t ; scan_out = sum_out  -> stacked cumulative sums.
        var body = new ModelGraph
        {
            Inputs = new[] { "sum_in", "x_t" },
            Outputs = new[] { "sum_out", "y_t" },
            Nodes = new[]
            {
                new GraphNode("Add", "s_add", new[] { "sum_in", "x_t" }, new[] { "sum_out" }),
                new GraphNode("Identity", "s_id", new[] { "sum_out" }, new[] { "y_t" }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "init", "x" },
            Outputs = new[] { "final_sum", "cumsum" },
            Nodes = new[]
            {
                new GraphNode("Scan", "scan0",
                    new[] { "init", "x" }, new[] { "final_sum", "cumsum" },
                    new Dictionary<string, object> { ["num_scan_inputs"] = (long)1, ["body"] = body }),
            },
        };

        // x = [1, 2, 3, 4] scanned along axis 0; init = 0.
        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["init"] = new NamedTensor("init", Scalar(0f)),
            ["x"] = new NamedTensor("x", Vec(1f, 2f, 3f, 4f)),
        });

        Assert.Equal(10f, outputs["final_sum"].Data.Span[0]);
        Tensor<float> cum = outputs["cumsum"].Data;
        Assert.Equal(new[] { 4 }, cum.Shape.Dimensions.ToArray());
        Assert.Equal(new[] { 1f, 3f, 6f, 10f }, cum.Span.ToArray());
    }

    [Fact]
    public void Scan_Honors_Reverse_Input_Direction()
    {
        // Scan x = [1,2,3] in reverse; cumulative sum then runs 3, 3+2=5, 5+1=6.
        var body = new ModelGraph
        {
            Inputs = new[] { "sum_in", "x_t" },
            Outputs = new[] { "sum_out", "y_t" },
            Nodes = new[]
            {
                new GraphNode("Add", "s_add", new[] { "sum_in", "x_t" }, new[] { "sum_out" }),
                new GraphNode("Identity", "s_id", new[] { "sum_out" }, new[] { "y_t" }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "init", "x" },
            Outputs = new[] { "final_sum", "cumsum" },
            Nodes = new[]
            {
                new GraphNode("Scan", "scan0",
                    new[] { "init", "x" }, new[] { "final_sum", "cumsum" },
                    new Dictionary<string, object>
                    {
                        ["num_scan_inputs"] = (long)1,
                        ["scan_input_directions"] = new long[] { 1 },
                        ["body"] = body,
                    }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["init"] = new NamedTensor("init", Scalar(0f)),
            ["x"] = new NamedTensor("x", Vec(1f, 2f, 3f)),
        });

        Assert.Equal(6f, outputs["final_sum"].Data.Span[0]);
        // Forward-stacked outputs of the reverse walk: 3, 5, 6.
        Assert.Equal(new[] { 3f, 5f, 6f }, outputs["cumsum"].Data.Span.ToArray());
    }

    // ---- Parse-level (subgraph as GRAPH attribute round-trips through the loader) -----------

    [Fact]
    public void Loader_Parses_Nested_Subgraph_From_Attribute()
    {
        // Hand-encode a minimal ONNX ModelProto whose graph holds an "If" node with a
        // then_branch GRAPH attribute (field 6). Assert the loader materializes the nested
        // subgraph as a ModelGraph value on the node's attribute dictionary.
        byte[] model = BuildModelWithIfSubgraph();
        ModelGraph parsed = ModelSharp.Onnx.OnnxModelLoader.ParseModel(model);

        GraphNode ifNode = Assert.Single(parsed.Nodes);
        Assert.Equal("If", ifNode.OpType);
        var sub = Assert.IsType<ModelGraph>(ifNode.Attributes["then_branch"]);
        Assert.Single(sub.Nodes);
        Assert.Equal("Identity", sub.Nodes[0].OpType);
        Assert.Equal(new[] { "z" }, sub.Outputs.ToArray());
    }

    // -- minimal protobuf encoders for the parse-level test --

    private static byte[] BuildModelWithIfSubgraph()
    {
        // Subgraph GraphProto: node(1) Identity(in -> z); output(12) ValueInfo name="z".
        byte[] subNode = Concat(
            LenField(1, Str("in")),
            LenField(2, Str("z")),
            LenField(4, Str("Identity")));
        byte[] subGraph = Concat(
            LenField(1, subNode),
            LenField(12, LenField(1, Str("z"))));

        // then_branch AttributeProto: name(1), type(20)=GRAPH(5), g(6)=subGraph.
        byte[] thenAttr = Concat(
            LenField(1, Str("then_branch")),
            VarintField(20, 5),
            LenField(6, subGraph));

        // If NodeProto: input(1)="cond", output(2)="y", op_type(4)="If", attribute(5)=thenAttr.
        byte[] ifNode = Concat(
            LenField(1, Str("cond")),
            LenField(2, Str("y")),
            LenField(4, Str("If")),
            LenField(5, thenAttr));

        // Top GraphProto: node(1)=ifNode, input(11) "cond", output(12) "y".
        byte[] topGraph = Concat(
            LenField(1, ifNode),
            LenField(11, LenField(1, Str("cond"))),
            LenField(12, LenField(1, Str("y"))));

        // ModelProto: graph(7)=topGraph.
        return LenField(7, topGraph);
    }

    private static byte[] Str(string s) => System.Text.Encoding.UTF8.GetBytes(s);

    private static byte[] LenField(int fieldNo, byte[] payload)
    {
        var tag = Varint((uint)((fieldNo << 3) | 2));
        var len = Varint((uint)payload.Length);
        return Concat(tag, len, payload);
    }

    private static byte[] VarintField(int fieldNo, ulong value)
        => Concat(Varint((uint)((fieldNo << 3) | 0)), Varint(value));

    private static byte[] Varint(ulong v)
    {
        var bytes = new List<byte>();
        do { byte b = (byte)(v & 0x7F); v >>= 7; if (v != 0) b |= 0x80; bytes.Add(b); } while (v != 0);
        return bytes.ToArray();
    }

    private static byte[] Concat(params byte[][] parts)
    {
        int n = 0;
        foreach (var p in parts) n += p.Length;
        var outBuf = new byte[n];
        int off = 0;
        foreach (var p in parts) { System.Array.Copy(p, 0, outBuf, off, p.Length); off += p.Length; }
        return outBuf;
    }
}
