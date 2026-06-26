using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ModelSharp.Cpu;
using ModelSharp.Cpu.Kernels;
using ModelSharp.Graph;
using ModelSharp.Onnx;
using ModelSharp.Tensors;
using ModelSharp.Text;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// Opt-in integration tests against the real pretrained all-MiniLM-L6-v2 model.
/// They no-op unless the (gitignored) model + vocab are present at assets/models/.
/// </summary>
public class MiniLmTests
{
    private readonly ITestOutputHelper _out;
    public MiniLmTests(ITestOutputHelper output) => _out = output;

    private static string ModelDir => Path.Combine(AppContext.BaseDirectory, "assets", "models");
    private static string ModelPath => Path.Combine(ModelDir, "all-MiniLM-L6-v2.onnx");
    private static string VocabPath => Path.Combine(ModelDir, "vocab.txt");

    [Fact]
    public void MiniLm_Op_Coverage_Probe()
    {
        if (!File.Exists(ModelPath)) { _out.WriteLine("MiniLM model not present; skipping."); return; }

        ModelGraph g = OnnxModelLoader.LoadModel(ModelPath);
        KernelRegistry registry = KernelRegistry.CreateDefault();
        var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
        var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();

        _out.WriteLine($"nodes={g.Nodes.Count}  distinctOps={distinct.Count}  initializers={g.Initializers.Count}");
        _out.WriteLine("ALL OPS: " + string.Join(", ", distinct));
        _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));
        Assert.True(missing.Count == 0, "Unsupported ops: " + string.Join(", ", missing));
    }

    [Fact]
    public void MiniLm_Produces_Semantic_Embeddings()
    {
        if (!File.Exists(ModelPath) || !File.Exists(VocabPath)) { _out.WriteLine("MiniLM assets not present; skipping."); return; }

        ModelGraph graph = OnnxModelLoader.LoadModel(ModelPath);
        var tok = WordPieceTokenizer.FromVocab(File.ReadLines(VocabPath), lowercase: true);

        float[] Embed(string text)
        {
            Encoding e = tok.Encode(text);
            int s = e.InputIds.Count;
            var ids = new long[s];
            var mask = new long[s];
            var types = new long[s];
            for (int i = 0; i < s; i++) { ids[i] = e.InputIds[i]; mask[i] = e.AttentionMask[i]; types[i] = e.TokenTypeIds[i]; }

            var feeds = new Dictionary<string, NamedTensor>
            {
                ["input_ids"] = new NamedTensor("input_ids", new Tensor<long>(new TensorShape(1, s), ids)),
                ["attention_mask"] = new NamedTensor("attention_mask", new Tensor<long>(new TensorShape(1, s), mask)),
                ["token_type_ids"] = new NamedTensor("token_type_ids", new Tensor<long>(new TensorShape(1, s), types)),
            };

            using var engine = new ManagedCpuEngine(graph);
            Tensor<float> hidden = engine.Run(feeds).Values.Single().Data;   // [1, S, 384]
            int h = hidden.Shape.Dimensions[^1];
            System.Span<float> span = hidden.Span;

            // Mean-pool token vectors using the attention mask, then L2-normalize.
            var pooled = new float[h];
            float maskSum = 0f;
            for (int t = 0; t < s; t++)
            {
                float m = mask[t];
                maskSum += m;
                for (int k = 0; k < h; k++) pooled[k] += span[t * h + k] * m;
            }
            for (int k = 0; k < h; k++) pooled[k] /= MathF.Max(maskSum, 1f);

            double norm = 0;
            foreach (float v in pooled) norm += v * v;
            norm = Math.Sqrt(norm);
            for (int k = 0; k < h; k++) pooled[k] = (float)(pooled[k] / norm);
            return pooled;
        }

        static float Cos(float[] a, float[] b)   // inputs already L2-normalized
        {
            double d = 0;
            for (int i = 0; i < a.Length; i++) d += a[i] * b[i];
            return (float)d;
        }

        float[] a = Embed("A man is playing a guitar.");
        float[] b = Embed("Someone is playing a musical instrument.");
        float[] c = Embed("The stock market fell sharply today.");

        float related = Cos(a, b);
        float unrelated = Cos(a, c);
        _out.WriteLine($"dim={a.Length}  sim(related)={related:F4}  sim(unrelated)={unrelated:F4}");

        Assert.Equal(384, a.Length);
        Assert.True(related > unrelated + 0.1f, $"related {related:F3} should clearly exceed unrelated {unrelated:F3}");
        Assert.True(related > 0.5f, $"related similarity {related:F3} is implausibly low — likely a numerical bug");
    }
}
