using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using ModelSharp.Graph;
using ModelSharp.Manifest;
using ModelSharp.Onnx;
using ModelSharp.Pipeline;
using ModelSharp.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// Covers the manifest → pipeline wiring: <see cref="ManifestResolver"/> (sidecar JSON,
/// embedded ONNX metadata, built-in heuristic), <see cref="ProcessorRegistry"/> round-trip
/// and errors, the built-in text-embedding processors, and an opt-in end-to-end run against
/// the real all-MiniLM-L6-v2 model (skipped when the assets are absent).
/// </summary>
public class ManifestPipelineTests
{
    private readonly ITestOutputHelper _out;
    public ManifestPipelineTests(ITestOutputHelper output) => _out = output;

    private static string ModelDir => Path.Combine(AppContext.BaseDirectory, "assets", "models");
    private static string ModelPath => Path.Combine(ModelDir, "all-MiniLM-L6-v2.onnx");
    private static string VocabPath => Path.Combine(ModelDir, "vocab.txt");

    private static string NewTempDir()
    {
        string d = Path.Combine(Path.GetTempPath(), "modelsharp-mp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    // ---------------------------------------------------------------- sidecar JSON

    [Fact]
    public void Resolver_Reads_Sidecar_Json()
    {
        string dir = NewTempDir();
        try
        {
            string modelPath = Path.Combine(dir, "img.onnx");
            string json = """
            {
              "task": "ImageClassification",
              "layout": "Nhwc",
              "width": 320,
              "height": 240,
              "mean": [0.5, 0.5, 0.5],
              "std": [0.25, 0.25, 0.25],
              "color": "Bgr",
              "labels": ["cat", "dog", "fish"],
              "extra": { "vocab": "vocab.txt", "foo": "bar" }
            }
            """;
            File.WriteAllText(modelPath + ".manifest.json", json);

            ModelManifest m = ManifestResolver.Resolve(modelPath, new ModelGraph());

            Assert.Equal(ModelTask.ImageClassification, m.Task);
            Assert.Equal(TensorLayout.Nhwc, m.Layout);
            Assert.Equal(320, m.Width);
            Assert.Equal(240, m.Height);
            Assert.Equal(0.5f, m.Mean[0]);
            Assert.Equal(0.25f, m.Std[1]);
            Assert.Equal(ColorOrder.Bgr, m.Color);
            Assert.Equal(new[] { "cat", "dog", "fish" }, m.Labels!.ToArray());
            Assert.Equal("bar", m.Extra["foo"]);
            // Relative vocab is resolved against the model directory.
            Assert.True(Path.IsPathRooted(m.Extra["vocab"]));
            Assert.EndsWith("vocab.txt", m.Extra["vocab"]);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Resolver_Reads_Sidecar_With_Swapped_Extension()
    {
        string dir = NewTempDir();
        try
        {
            string modelPath = Path.Combine(dir, "model2.onnx");
            // "<model>.manifest.json" (extension swapped), not "<model>.onnx.manifest.json".
            File.WriteAllText(Path.Combine(dir, "model2.manifest.json"), """{ "task": "Embedding" }""");

            ModelManifest m = ManifestResolver.Resolve(modelPath, new ModelGraph());
            Assert.Equal(ModelTask.Embedding, m.Task);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Resolver_Sidecar_Tolerates_Missing_Fields()
    {
        string dir = NewTempDir();
        try
        {
            string modelPath = Path.Combine(dir, "sparse.onnx");
            File.WriteAllText(modelPath + ".manifest.json", """{ "task": "Embedding" }""");

            ModelManifest m = ManifestResolver.Resolve(modelPath, new ModelGraph());
            Assert.Equal(ModelTask.Embedding, m.Task);
            Assert.Equal(TensorLayout.Nchw, m.Layout);   // default
            Assert.Equal(0, m.Width);                    // default
            Assert.Equal(new[] { 0f, 0f, 0f }, m.Mean.ToArray());
            Assert.Equal(new[] { 1f, 1f, 1f }, m.Std.ToArray());
            Assert.Null(m.Labels);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---------------------------------------------------------------- embedded ONNX metadata

    [Fact]
    public void Loader_Surfaces_Embedded_Metadata_Props()
    {
        byte[] model = BuildModelProto(
            ("task", "Embedding"),
            ("vocab", "vocab.txt"),
            ("layout", "Nchw"),
            ("mean", "0.1,0.2,0.3"));

        ModelGraph g = OnnxModelLoader.ParseModel(model);

        Assert.Equal("Embedding", g.MetadataProps["task"]);
        Assert.Equal("vocab.txt", g.MetadataProps["vocab"]);
        Assert.Equal("Nchw", g.MetadataProps["layout"]);
        Assert.Equal("0.1,0.2,0.3", g.MetadataProps["mean"]);
    }

    [Fact]
    public void Resolver_Reads_Embedded_Metadata()
    {
        byte[] model = BuildModelProto(
            ("modelsharp.task", "Embedding"),
            ("vocab", "vocab.txt"),
            ("mean", "0.1,0.2,0.3"));
        ModelGraph g = OnnxModelLoader.ParseModel(model);

        string dir = NewTempDir();
        try
        {
            // A name with no built-in match and no sidecar — metadata must win.
            string modelPath = Path.Combine(dir, "mystery-net.onnx");
            ModelManifest m = ManifestResolver.Resolve(modelPath, g);

            Assert.Equal(ModelTask.Embedding, m.Task);
            Assert.Equal(0.1f, m.Mean[0]);
            Assert.Equal(0.3f, m.Mean[2]);
            Assert.True(Path.IsPathRooted(m.Extra["vocab"]));
            Assert.EndsWith("vocab.txt", m.Extra["vocab"]);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Empty_Metadata_Falls_Through_To_BuiltIn()
    {
        // A graph with no metadata + a non-matching name resolves to Unknown via the built-in path.
        string dir = NewTempDir();
        try
        {
            string modelPath = Path.Combine(dir, "no-metadata-here.onnx");
            ModelManifest m = ManifestResolver.Resolve(modelPath, new ModelGraph());
            Assert.Equal(ModelTask.Unknown, m.Task);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---------------------------------------------------------------- built-in heuristic

    [Fact]
    public void BuiltIn_Heuristic_Recognizes_Embedding_And_Image_Models()
    {
        string dir = NewTempDir();
        try
        {
            ModelManifest mini = ManifestResolver.Resolve(Path.Combine(dir, "all-MiniLM-L6-v2.onnx"), new ModelGraph());
            Assert.Equal(ModelTask.Embedding, mini.Task);
            Assert.True(Path.IsPathRooted(mini.Extra["vocab"]));
            Assert.EndsWith("vocab.txt", mini.Extra["vocab"]);

            ModelManifest img = ManifestResolver.Resolve(Path.Combine(dir, "resnet50-v1-7.onnx"), new ModelGraph());
            Assert.Equal(ModelTask.ImageClassification, img.Task);
            Assert.Equal(224, img.Width);
            Assert.Equal(224, img.Height);
            Assert.Equal(TensorLayout.Nchw, img.Layout);

            ModelManifest unknown = ManifestResolver.Resolve(Path.Combine(dir, "totally-custom-thing.onnx"), new ModelGraph());
            Assert.Equal(ModelTask.Unknown, unknown.Task);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---------------------------------------------------------------- ProcessorRegistry

    [Fact]
    public void Registry_Register_Create_RoundTrip_And_Last_Wins()
    {
        const ModelTask task = (ModelTask)900;   // isolated, undefined value — never used by adapters
        var pre = new StubPre();
        var post = new StubPost();
        ProcessorRegistry.RegisterPreprocessor(task, _ => pre);
        ProcessorRegistry.RegisterPostprocessor(task, _ => post);

        var ctx = new ProcessorContext(new ModelManifest { Task = task }, new[] { "x" }, new[] { "y" });
        Assert.Same(pre, ProcessorRegistry.CreatePreprocessor(ctx));
        Assert.Same(post, ProcessorRegistry.CreatePostprocessor(ctx));

        // Last registration wins.
        var pre2 = new StubPre();
        ProcessorRegistry.RegisterPreprocessor(task, _ => pre2);
        Assert.Same(pre2, ProcessorRegistry.CreatePreprocessor(ctx));
    }

    [Fact]
    public void Registry_Throws_Clear_Error_When_No_Factory()
    {
        const ModelTask task = (ModelTask)901;   // never registered
        var ctx = new ProcessorContext(new ModelManifest { Task = task }, Array.Empty<string>(), Array.Empty<string>());

        ModelSharpException pre = Assert.Throws<ModelSharpException>(() => ProcessorRegistry.CreatePreprocessor(ctx));
        Assert.Contains("901", pre.Message);

        ModelSharpException post = Assert.Throws<ModelSharpException>(() => ProcessorRegistry.CreatePostprocessor(ctx));
        Assert.Contains("901", post.Message);
    }

    [Fact]
    public void Embedding_Processors_Registered_By_Default()
    {
        string dir = NewTempDir();
        try
        {
            string vocab = Path.Combine(dir, "vocab.txt");
            File.WriteAllLines(vocab, new[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world" });

            var manifest = new ModelManifest
            {
                Task = ModelTask.Embedding,
                Extra = new Dictionary<string, string> { ["vocab"] = vocab },
            };

            // Defaults are wired by the static ctor — CreatePreprocessor must succeed for Embedding.
            var ctx = new ProcessorContext(manifest, new[] { "input_ids", "attention_mask", "token_type_ids" }, new[] { "out" });
            IPreprocessor pre = ProcessorRegistry.CreatePreprocessor(ctx);
            IPostprocessor post = ProcessorRegistry.CreatePostprocessor(ctx);

            Assert.IsType<TextEmbeddingPreprocessor>(pre);
            Assert.IsType<MeanPoolEmbeddingPostprocessor>(post);

            IReadOnlyDictionary<string, NamedTensor> feeds = pre.ToFeeds("hello world");
            Assert.Equal(new[] { 1, 4 }, feeds["input_ids"].Tensor.Shape.Dimensions.ToArray());   // [1, S], S=4
            Assert.Equal(new long[] { 2, 4, 5, 3 }, feeds["input_ids"].Tensor.AsInt64().Span.ToArray());  // [CLS] hello world [SEP]
            Assert.True(feeds.ContainsKey("attention_mask"));
            Assert.True(feeds.ContainsKey("token_type_ids"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void TextEmbeddingPreprocessor_Maps_Only_Declared_Inputs()
    {
        string dir = NewTempDir();
        try
        {
            string vocab = Path.Combine(dir, "vocab.txt");
            File.WriteAllLines(vocab, new[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello" });
            var manifest = new ModelManifest
            {
                Task = ModelTask.Embedding,
                Extra = new Dictionary<string, string> { ["vocab"] = vocab },
            };

            // Model omits token_type_ids — the preprocessor must not feed it.
            var ctx = new ProcessorContext(manifest, new[] { "input_ids", "attention_mask" }, new[] { "out" });
            var pre = new TextEmbeddingPreprocessor(ctx);
            IReadOnlyDictionary<string, NamedTensor> feeds = pre.ToFeeds("hello");

            Assert.True(feeds.ContainsKey("input_ids"));
            Assert.True(feeds.ContainsKey("attention_mask"));
            Assert.False(feeds.ContainsKey("token_type_ids"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void MeanPool_Postprocessor_Pools_And_Normalizes()
    {
        // Two tokens of hidden size 3: mean = [2,4,4]; then L2-normalize.
        var hidden = new Tensor<float>(new TensorShape(1, 2, 3), new float[] { 1, 2, 3, 3, 6, 5 });
        var outputs = new Dictionary<string, NamedTensor> { ["h"] = new NamedTensor("h", hidden) };

        var v = (float[])new MeanPoolEmbeddingPostprocessor().Decode(outputs);

        float[] mean = { 2f, 4f, 4f };
        double norm = Math.Sqrt(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]);
        Assert.Equal(3, v.Length);
        Assert.Equal((float)(mean[0] / norm), v[0], 5);
        Assert.Equal((float)(mean[1] / norm), v[1], 5);
        Assert.Equal((float)(mean[2] / norm), v[2], 5);

        double unit = 0; foreach (float x in v) unit += x * x;
        Assert.Equal(1.0, unit, 4);   // result is unit length
    }

    // ---------------------------------------------------------------- end-to-end (opt-in)

    [Fact]
    public void EndToEnd_Embedding_Pipeline_Produces_Semantic_Embeddings()
    {
        if (!File.Exists(ModelPath) || !File.Exists(VocabPath)) { _out.WriteLine("MiniLM assets not present; skipping."); return; }

        var manifest = new ModelManifest
        {
            Task = ModelTask.Embedding,
            Extra = new Dictionary<string, string> { ["vocab"] = VocabPath },
        };

        using var pipeline = ModelSharpPipeline.Load(ModelPath, manifest);

        float[] a = pipeline.Run<float[]>("A man is playing a guitar.");
        float[] b = pipeline.Run<float[]>("Someone is playing a musical instrument.");
        float[] c = pipeline.Run<float[]>("The stock market fell sharply today.");

        float related = Cos(a, b);
        float unrelated = Cos(a, c);
        _out.WriteLine($"dim={a.Length}  sim(related)={related:F4}  sim(unrelated)={unrelated:F4}");

        Assert.Equal(384, a.Length);
        Assert.True(related > unrelated + 0.1f, $"related {related:F3} should clearly exceed unrelated {unrelated:F3}");
        Assert.True(related > 0.5f, $"related similarity {related:F3} is implausibly low — likely a numerical bug");
    }

    [Fact]
    public void EndToEnd_AutoResolve_Recognizes_MiniLm_From_FileName()
    {
        if (!File.Exists(ModelPath) || !File.Exists(VocabPath)) { _out.WriteLine("MiniLM assets not present; skipping."); return; }

        // No explicit manifest: the built-in heuristic recognizes "minilm" and points vocab
        // at the sibling vocab.txt — the headline "just run" promise.
        using var pipeline = ModelSharpPipeline.Load(ModelPath);
        Assert.Equal(ModelTask.Embedding, pipeline.Manifest.Task);

        float[] a = pipeline.Run<float[]>("A man is playing a guitar.");
        float[] b = pipeline.Run<float[]>("Someone is playing a musical instrument.");
        Assert.Equal(384, a.Length);
        Assert.True(Cos(a, b) > 0.5f);
    }

    private static float Cos(float[] a, float[] b)   // inputs already L2-normalized
    {
        double d = 0;
        for (int i = 0; i < a.Length; i++) d += a[i] * b[i];
        return (float)d;
    }

    // ---------------------------------------------------------------- helpers

    private sealed class StubPre : IPreprocessor
    {
        public IReadOnlyDictionary<string, NamedTensor> ToFeeds(object input) => new Dictionary<string, NamedTensor>();
    }

    private sealed class StubPost : IPostprocessor
    {
        public object Decode(IReadOnlyDictionary<string, NamedTensor> outputs) => "ok";
    }

    /// <summary>Hand-builds a minimal ONNX ModelProto: an empty graph (field 7) + metadata_props (field 14).</summary>
    private static byte[] BuildModelProto(params (string Key, string Value)[] metadata)
    {
        var buf = new List<byte>();
        WriteLenField(buf, 7, Array.Empty<byte>());   // empty GraphProto
        foreach ((string key, string value) in metadata)
        {
            var entry = new List<byte>();
            WriteStr(entry, 1, key);
            WriteStr(entry, 2, value);
            WriteLenField(buf, 14, entry.ToArray());   // StringStringEntryProto
        }
        return buf.ToArray();
    }

    private static void WriteVarint(List<byte> buf, ulong v)
    {
        while (v >= 0x80) { buf.Add((byte)(v | 0x80)); v >>= 7; }
        buf.Add((byte)v);
    }

    private static void WriteLenField(List<byte> buf, int field, byte[] payload)
    {
        WriteVarint(buf, (ulong)((field << 3) | 2));   // wire type 2 = length-delimited
        WriteVarint(buf, (ulong)payload.Length);
        buf.AddRange(payload);
    }

    private static void WriteStr(List<byte> buf, int field, string s)
        => WriteLenField(buf, field, Encoding.UTF8.GetBytes(s));
}
