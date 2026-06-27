using System.Collections.Generic;
using System.Linq;
using ModelSharp;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Proves that ONNX <c>Sequence*</c>/<c>Optional*</c> (non-tensor) values cross the control-flow
/// subgraph boundary in both directions:
/// <list type="bullet">
/// <item><b>Inward (capture):</b> an If/Loop body reads a sequence built in the OUTER graph
/// (e.g. <c>SequenceAt</c> over an outer sequence).</item>
/// <item><b>Outward (produce):</b> a subgraph body BUILDS a sequence and declares it as an output,
/// which the parent then consumes.</item>
/// </list>
/// Mirrors <see cref="ControlFlowOpsTests"/> + <see cref="SequenceOpsTests"/>: graphs and their
/// nested subgraphs are hand-built in-memory; a subgraph is a GRAPH attribute on the node.
/// </summary>
public class ControlFlowSequenceCrossingTests
{
    private static Tensor<float> Vec(params float[] v) => new(new TensorShape(v.Length), v);
    private static Tensor<bool> BoolScalar(bool v) => new(new TensorShape(), new[] { v });
    private static Tensor<long> LongScalar(long v) => new(new TensorShape(), new[] { v });

    private static IReadOnlyDictionary<string, NamedTensor> Run(
        ModelGraph g, Dictionary<string, NamedTensor> feeds)
    {
        using var engine = new ManagedCpuEngine(g);
        return engine.Run(feeds);
    }

    // ---- Inward: If body reads an OUTER-scope sequence via SequenceAt -----------------------

    [Theory]
    [InlineData(true, new[] { 10f, 11f })]   // then -> seq[0] = a
    [InlineData(false, new[] { 30f, 31f })]  // else -> seq[2] = c
    public void If_Body_Reads_Outer_Sequence_Via_SequenceAt(bool cond, float[] expected)
    {
        // Outer graph builds seq = [a, b, c]; the If branches each SequenceAt into that OUTER
        // sequence (captured, no formal inputs), proving a sequence crosses INTO the subgraph.
        var thenBranch = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "t_out" },
            Initializers = new Dictionary<string, Tensor> { ["i0"] = LongScalar(0) },
            Nodes = new[]
            {
                new GraphNode("SequenceAt", "t_at", new[] { "seq", "i0" }, new[] { "t_out" }),
            },
        };
        var elseBranch = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "e_out" },
            Initializers = new Dictionary<string, Tensor> { ["i2"] = LongScalar(2) },
            Nodes = new[]
            {
                new GraphNode("SequenceAt", "e_at", new[] { "seq", "i2" }, new[] { "e_out" }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "cond", "a", "b", "c" },
            Outputs = new[] { "y" },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "sc", new[] { "a", "b", "c" }, new[] { "seq" }),
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
            ["a"] = new NamedTensor("a", Vec(10f, 11f)),
            ["b"] = new NamedTensor("b", Vec(20f, 21f)),
            ["c"] = new NamedTensor("c", Vec(30f, 31f)),
        });

        Assert.Equal(expected, outputs["y"].Data.Span.ToArray());
    }

    // ---- Outward: If body BUILDS a sequence and declares it as an output --------------------

    [Fact]
    public void If_Body_Produces_Sequence_Output_Consumed_By_Parent()
    {
        // The chosen branch constructs a 2-element sequence "seq" and declares it as its output.
        // The If node's output is ALSO named "seq" (legal: a node output may share the body's
        // output name) so the produced SeqValue lands in the parent scope under "seq". The parent
        // then SequenceAt(seq, 1) to recover a tensor — proving a sequence crosses OUT of the body.
        var thenBranch = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "seq" },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "t_sc", new[] { "a", "b" }, new[] { "seq" }),
            },
        };
        var elseBranch = new ModelGraph
        {
            Inputs = new string[0],
            Outputs = new[] { "seq" },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "e_sc", new[] { "b", "a" }, new[] { "seq" }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "cond", "a", "b" },
            Outputs = new[] { "elem", "len" },
            Initializers = new Dictionary<string, Tensor> { ["one"] = LongScalar(1) },
            Nodes = new[]
            {
                new GraphNode("If", "if0", new[] { "cond" }, new[] { "seq" },
                    new Dictionary<string, object>
                    {
                        ["then_branch"] = thenBranch,
                        ["else_branch"] = elseBranch,
                    }),
                // Consume the sequence that the subgraph produced, back in the parent scope.
                new GraphNode("SequenceAt", "sa", new[] { "seq", "one" }, new[] { "elem" }),
                new GraphNode("SequenceLength", "sl", new[] { "seq" }, new[] { "len" }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["cond"] = new NamedTensor("cond", BoolScalar(true)),  // then: seq = [a, b]; seq[1] = b
            ["a"] = new NamedTensor("a", Vec(1f, 2f)),
            ["b"] = new NamedTensor("b", Vec(7f, 8f)),
        });

        Assert.Equal(2L, outputs["len"].Tensor.AsInt64().Span[0]);
        Assert.Equal(new[] { 7f, 8f }, outputs["elem"].Data.Span.ToArray());
    }

    // ---- Inward + Loop: a Loop body reads an outer sequence each iteration ------------------

    [Fact]
    public void Loop_Body_Reads_Outer_Sequence_Each_Iteration()
    {
        // Outer seq = [v0, v1, v2]. Loop runs M=3 times; each iteration the body does
        // SequenceAt(seq, iter) (reading the OUTER sequence) and accumulates the scalar into acc.
        // acc_final = sum of the three scalars -> proves the outer sequence is visible inside the
        // loop body across iterations.
        var body = new ModelGraph
        {
            Inputs = new[] { "iter", "cond_in", "acc" },
            Outputs = new[] { "cond_in", "acc_out" },
            Nodes = new[]
            {
                new GraphNode("SequenceAt", "b_at", new[] { "seq", "iter" }, new[] { "elem" }),
                new GraphNode("Add", "b_acc", new[] { "acc", "elem" }, new[] { "acc_out" }),
            },
        };

        var graph = new ModelGraph
        {
            Inputs = new[] { "M", "cond", "acc_init", "v0", "v1", "v2" },
            Outputs = new[] { "acc_final" },
            Nodes = new[]
            {
                new GraphNode("SequenceConstruct", "sc", new[] { "v0", "v1", "v2" }, new[] { "seq" }),
                new GraphNode("Loop", "loop0",
                    new[] { "M", "cond", "acc_init" }, new[] { "acc_final" },
                    new Dictionary<string, object> { ["body"] = body }),
            },
        };

        var outputs = Run(graph, new Dictionary<string, NamedTensor>
        {
            ["M"] = new NamedTensor("M", LongScalar(3)),
            ["cond"] = new NamedTensor("cond", BoolScalar(true)),
            ["acc_init"] = new NamedTensor("acc_init", Vec(0f)),
            ["v0"] = new NamedTensor("v0", Vec(2f)),
            ["v1"] = new NamedTensor("v1", Vec(3f)),
            ["v2"] = new NamedTensor("v2", Vec(5f)),
        });

        // 2 + 3 + 5 = 10
        Assert.Equal(new[] { 10f }, outputs["acc_final"].Data.Span.ToArray());
    }

    // ---- Regression: a tensor-only If subgraph still works (no seq plumbing engaged) --------

    [Fact]
    public void Tensor_Only_If_Still_Works()
    {
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
            ["cond"] = new NamedTensor("cond", BoolScalar(true)),
            ["a"] = new NamedTensor("a", Vec(2f)),
            ["b"] = new NamedTensor("b", Vec(3f)),
        });
        Assert.Equal(new[] { 5f }, outputs["y"].Data.Span.ToArray());
    }
}
