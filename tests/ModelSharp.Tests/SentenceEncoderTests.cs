using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Graph;
using ModelSharp.Tensors;
using ModelSharp.Text;
using Xunit;

namespace ModelSharp.Tests;

public class SentenceEncoderTests
{
    private static Tensor<float> T(int[] dims, params float[] data) => Tensor<float>.FromArray(new TensorShape(dims), data);

    /// <summary>
    /// Capstone: a full text → embedding pipeline assembled from the pieces built across the
    /// phases — WordPiece tokenizer → embedding Gather → scaled-dot-product self-attention →
    /// ReduceMean pooling — producing a sentence vector. With every token embedding equal to
    /// e = [3, 5], identity projections + uniform attention + mean pooling return exactly e,
    /// so the result is hand-verifiable end to end.
    /// </summary>
    [Fact]
    public void Tokenize_Embed_Attend_Pool_Produces_Sentence_Vector()
    {
        // 1) Real WordPiece tokenization.
        var vocab = new Dictionary<string, int>
        {
            ["[PAD]"] = 0, ["[UNK]"] = 1, ["[CLS]"] = 2, ["[SEP]"] = 3, ["hello"] = 4, ["world"] = 5,
        };
        var tok = new WordPieceTokenizer(vocab, lowercase: true);
        Encoding enc = tok.Encode("Hello world");
        Assert.Equal(new[] { 2, 4, 5, 3 }, enc.InputIds.ToArray());   // [CLS] hello world [SEP]

        int S = enc.InputIds.Count;
        const int d = 2;
        Tensor<float> ids = T(new[] { S }, enc.InputIds.Select(i => (float)i).ToArray());

        // 2) Embedding table: every token maps to e = [3, 5].
        int V = vocab.Count;
        var embedData = new float[V * d];
        for (int i = 0; i < V; i++) { embedData[i * d] = 3f; embedData[i * d + 1] = 5f; }
        Tensor<float> embed = T(new[] { V, d }, embedData);

        Tensor<float> identity = T(new[] { d, d }, 1, 0, 0, 1);
        float scale = 1f / MathF.Sqrt(d);

        // 3) Graph: Gather embeddings → self-attention → mean-pool over the sequence.
        var graph = new ModelGraph
        {
            Inputs = new[] { "ids" },
            Outputs = new[] { "pooled" },
            Nodes = new[]
            {
                new GraphNode("Gather", "emb", new[] { "embed", "ids" }, new[] { "E" }, new Dictionary<string, object> { ["axis"] = 0L }),
                new GraphNode("MatMul", "q", new[] { "E", "Wq" }, new[] { "Q" }),
                new GraphNode("MatMul", "k", new[] { "E", "Wk" }, new[] { "K" }),
                new GraphNode("MatMul", "v", new[] { "E", "Wv" }, new[] { "Vv" }),
                new GraphNode("Transpose", "kt", new[] { "K" }, new[] { "Kt" }),
                new GraphNode("MatMul", "sc", new[] { "Q", "Kt" }, new[] { "Sc" }),
                new GraphNode("Mul", "scl", new[] { "Sc", "scale" }, new[] { "Scl" }),
                new GraphNode("Softmax", "sm", new[] { "Scl" }, new[] { "A" }),
                new GraphNode("MatMul", "ctx", new[] { "A", "Vv" }, new[] { "Ctx" }),
                new GraphNode("ReduceMean", "pool", new[] { "Ctx" }, new[] { "pooled" },
                    new Dictionary<string, object> { ["axes"] = new long[] { 0 }, ["keepdims"] = 0L }),
            },
            Initializers = new Dictionary<string, Tensor>
            {
                ["embed"] = embed,
                ["Wq"] = identity,
                ["Wk"] = identity,
                ["Wv"] = identity,
                ["scale"] = T(new[] { 1 }, scale),
            },
        };

        using var engine = new ManagedCpuEngine(graph);
        Tensor<float> pooled = engine.Run(new Dictionary<string, NamedTensor>
        {
            ["ids"] = new NamedTensor("ids", ids),
        }).Values.Single().Data;

        Assert.Equal(new[] { d }, pooled.Shape.Dimensions.ToArray());
        float[] vec = pooled.Span.ToArray();
        Assert.Equal(3f, vec[0], 4);
        Assert.Equal(5f, vec[1], 4);
    }
}
