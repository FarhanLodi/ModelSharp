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

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Opt-in integration test against a real <c>bert-base-uncased</c> ONNX export (Xenova/optimum form):
/// a masked-language-model head that outputs <c>logits</c> of shape <c>[1, seq, vocab]</c>. No-ops
/// (green) unless the model + WordPiece vocab are present in the resolved models dir.
///
/// <para>Expected files in <see cref="RealModelAssets.ModelsDir"/>:
/// <c>bert.onnx</c> (BERT-for-MaskedLM with inputs <c>input_ids</c> / <c>attention_mask</c> /
/// <c>token_type_ids</c> and output <c>logits</c>) and <c>bert-vocab.txt</c> (one WordPiece token per line).
/// Source: https://huggingface.co/Xenova/bert-base-uncased (onnx/model.onnx + vocab.txt).</para>
/// </summary>
public class BertMaskedLmTests
{
    private readonly ITestOutputHelper _out;
    public BertMaskedLmTests(ITestOutputHelper output) => _out = output;

    private const string ModelFile = "bert.onnx";
    private const string VocabFile = "bert-vocab.txt";

    [Fact]
    public void Bert_Op_Coverage_Probe()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath))
        {
            _out.WriteLine($"BERT model not present ({modelPath}); skipping.");
            return;
        }

        ModelGraph g = OnnxModelLoader.LoadModel(modelPath);
        KernelRegistry registry = KernelRegistry.CreateDefault();
        var distinct = g.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToList();
        var missing = distinct.Where(op => !registry.TryGet(op, out _)).OrderBy(s => s).ToList();

        _out.WriteLine($"nodes={g.Nodes.Count}  distinctOps={distinct.Count}  initializers={g.Initializers.Count}");
        _out.WriteLine("ALL OPS: " + string.Join(", ", distinct));
        _out.WriteLine("MISSING OPS: " + (missing.Count == 0 ? "(none)" : string.Join(", ", missing)));
        Assert.True(missing.Count == 0, "Unsupported ops: " + string.Join(", ", missing));
    }

    [Fact]
    public void Bert_Masked_Token_Predicts_Plausible_Word()
    {
        if (!RealModelAssets.TryPath(ModelFile, out string modelPath)
            || !RealModelAssets.TryPath(VocabFile, out string vocabPath))
        {
            _out.WriteLine("BERT assets not present; skipping.");
            return;
        }

        ModelGraph graph = OnnxModelLoader.LoadModel(modelPath);
        var tok = WordPieceTokenizer.FromVocab(File.ReadLines(vocabPath), lowercase: true);

        Assert.True(tok.TryGetId("[MASK]", out int maskId), "vocab is missing the [MASK] token.");
        Assert.True(tok.TryGetId("[CLS]", out int clsId), "vocab is missing the [CLS] token.");
        Assert.True(tok.TryGetId("[SEP]", out int sepId), "vocab is missing the [SEP] token.");

        // Assemble "[CLS] the capital of france is [MASK] . [SEP]" by tokenizing the prefix/suffix
        // (no specials) around a single [MASK] slot, so the mask position is unambiguous regardless
        // of how WordPiece splits the surrounding words.
        List<string> prefix = tok.TokenizeToPieces("the capital of france is");
        List<string> suffix = tok.TokenizeToPieces(".");

        var idList = new List<long> { clsId };
        foreach (string p in prefix) idList.Add(tok.TryGetId(p, out int pid) ? pid : maskId);
        int maskPos = idList.Count;
        idList.Add(maskId);
        foreach (string p in suffix) idList.Add(tok.TryGetId(p, out int sid) ? sid : maskId);
        idList.Add(sepId);

        long[] ids = idList.ToArray();
        var mask = ids.Select(_ => 1L).ToArray();
        var types = ids.Select(_ => 0L).ToArray();

        _out.WriteLine($"seq={ids.Length} maskPos={maskPos}");

        int s = ids.Length;
        var feeds = new Dictionary<string, NamedTensor>
        {
            ["input_ids"] = new NamedTensor("input_ids", new Tensor<long>(new TensorShape(1, s), ids)),
            ["attention_mask"] = new NamedTensor("attention_mask", new Tensor<long>(new TensorShape(1, s), mask)),
            ["token_type_ids"] = new NamedTensor("token_type_ids", new Tensor<long>(new TensorShape(1, s), types)),
        };

        using var engine = new ManagedCpuEngine(graph);
        Tensor<float> logits = engine.Run(feeds).Values.Single().Data;   // [1, S, vocab]

        int vocab = logits.Shape.Dimensions[^1];
        Assert.Equal(s, logits.Shape.Dimensions[^2]);
        Span<float> span = logits.Span;

        // Argmax over the vocab at the masked position.
        int baseOff = maskPos * vocab;
        int best = 0;
        float bestVal = span[baseOff];
        for (int v = 1; v < vocab; v++)
        {
            float val = span[baseOff + v];
            if (float.IsNaN(val) || float.IsInfinity(val))
                Assert.Fail($"non-finite logit at vocab index {v}.");
            if (val > bestVal) { bestVal = val; best = v; }
        }

        // Map the predicted id back to a token string via the vocab file (line index == id).
        string[] vocabLines = File.ReadAllLines(vocabPath);
        string predicted = best < vocabLines.Length ? vocabLines[best] : $"<id {best}>";

        // Top-5 for the log.
        float[] logitRow = logits.Buffer.Slice(baseOff, vocab).ToArray();
        var top5 = Enumerable.Range(0, vocab)
            .Select(v => (id: v, score: logitRow[v]))
            .OrderByDescending(t => t.score).Take(5)
            .Select(t => $"{(t.id < vocabLines.Length ? vocabLines[t.id] : t.id.ToString())}({t.score:F2})");
        _out.WriteLine($"predicted='{predicted}'  top5=[{string.Join(", ", top5)}]");

        Assert.False(string.IsNullOrWhiteSpace(predicted), "MLM produced an empty prediction.");
        // "the capital of france is [MASK]" — a correctly wired BERT overwhelmingly predicts "paris".
        Assert.Equal("paris", predicted);
    }
}
