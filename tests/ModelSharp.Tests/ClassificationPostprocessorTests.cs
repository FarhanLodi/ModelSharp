using System;
using System.Collections.Generic;
using System.Linq;
using ModelSharp.ImageSharp;
using ModelSharp.Manifest;
using ModelSharp.Tensors;
using Xunit;

namespace ModelSharp.Tests;

/// <summary>
/// Synthetic coverage for <see cref="ClassificationPostprocessor"/>: top-K ordering, label
/// mapping, softmax-vs-raw scoring, and Extra-driven configuration (top_k / softmax). Tests feed
/// a hand-built [1, N] logits tensor straight into <see cref="ClassificationPostprocessor.Decode"/>
/// — no model or image required.
/// </summary>
public class ClassificationPostprocessorTests
{
    [Fact]
    public void Decodes_TopK_Ordering_And_Label_Mapping()
    {
        // Logits favor index 2, then 0, then 1. argmax should be class "dog" (index 2).
        var logits = new[] { 2.0f, 0.5f, 3.0f, -1.0f };
        var outputs = Outputs(new TensorShape(1, 4), logits);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ImageClassification,
            Labels = new[] { "cat", "bird", "dog", "fish" },
        };
        // Raw scoring + top-3 so we can assert exact ordering against the input logits.
        var post = new ClassificationPostprocessor(manifest, topK: 3, applySoftmax: false);

        var results = (List<Classification>)post.Decode(outputs);

        Assert.Equal(3, results.Count);
        Assert.Equal(new[] { 2, 0, 1 }, results.Select(c => c.Index).ToArray());
        Assert.Equal(new[] { "dog", "cat", "bird" }, results.Select(c => c.Label).ToArray());

        // Top-1 is the argmax with its raw logit preserved.
        Assert.Equal(2, results[0].Index);
        Assert.Equal(3.0f, results[0].Score, 5);
    }

    [Fact]
    public void Softmax_Normalizes_Scores_To_A_Probability_Distribution()
    {
        var logits = new[] { 1.0f, 2.0f, 3.0f };
        var outputs = Outputs(new TensorShape(1, 3), logits);

        var manifest = new ModelManifest { Task = ModelTask.ImageClassification };
        var post = new ClassificationPostprocessor(manifest, topK: 3, applySoftmax: true);

        var results = (List<Classification>)post.Decode(outputs);

        // Probabilities sum to 1 and rank in the same order as the logits.
        Assert.Equal(1.0f, results.Sum(c => c.Score), 4);
        Assert.Equal(new[] { 2, 1, 0 }, results.Select(c => c.Index).ToArray());

        // Hand-checked softmax of [1,2,3]: e^x normalized => ~[0.0900, 0.2447, 0.6652].
        Assert.Equal(0.6652f, results[0].Score, 3);   // index 2
        Assert.Equal(0.2447f, results[1].Score, 3);   // index 1
        Assert.Equal(0.0900f, results[2].Score, 3);   // index 0
    }

    [Fact]
    public void TopK_Caps_Result_Count()
    {
        var logits = new[] { 0.1f, 0.9f, 0.4f, 0.7f, 0.2f };
        var outputs = Outputs(new TensorShape(1, 5), logits);

        var manifest = new ModelManifest { Task = ModelTask.ImageClassification };
        var post = new ClassificationPostprocessor(manifest, topK: 2, applySoftmax: false);

        var results = (List<Classification>)post.Decode(outputs);

        Assert.Equal(2, results.Count);
        Assert.Equal(new[] { 1, 3 }, results.Select(c => c.Index).ToArray());   // 0.9 then 0.7
    }

    [Fact]
    public void TopK_And_Softmax_Read_From_Manifest_Extra()
    {
        var logits = new[] { 0.1f, 0.9f, 0.4f, 0.7f };
        var outputs = Outputs(new TensorShape(1, 4), logits);

        var manifest = new ModelManifest
        {
            Task = ModelTask.ImageClassification,
            Extra = new Dictionary<string, string>
            {
                ["top_k"] = "1",
                ["softmax"] = "false",
            },
        };
        // No explicit ctor args => the Extra hints drive top_k=1 and softmax=off.
        var post = new ClassificationPostprocessor(manifest);

        var results = (List<Classification>)post.Decode(outputs);

        Classification only = Assert.Single(results);
        Assert.Equal(1, only.Index);
        Assert.Equal(0.9f, only.Score, 5);   // raw logit, softmax disabled via Extra
    }

    [Fact]
    public void Defaults_To_TopK_Five_And_Softmax_On()
    {
        // 8 classes; default top_k is 5 and softmax normalizes to probabilities.
        var logits = Enumerable.Range(0, 8).Select(i => (float)i).ToArray();
        var outputs = Outputs(new TensorShape(1, 8), logits);

        var manifest = new ModelManifest { Task = ModelTask.ImageClassification };
        var post = new ClassificationPostprocessor(manifest);

        var results = (List<Classification>)post.Decode(outputs);

        Assert.Equal(ClassificationPostprocessor.DefaultTopK, results.Count);
        Assert.Equal(7, results[0].Index);           // largest logit
        Assert.True(results[0].Score < 1.0f);        // softmax probability, not the raw logit 7
        Assert.True(results[0].Score > results[1].Score);
    }

    [Fact]
    public void Missing_Labels_Leave_Label_Null()
    {
        var logits = new[] { 0.2f, 0.8f };
        var outputs = Outputs(new TensorShape(1, 2), logits);

        var manifest = new ModelManifest { Task = ModelTask.ImageClassification };   // no Labels
        var post = new ClassificationPostprocessor(manifest, applySoftmax: false);

        var results = (List<Classification>)post.Decode(outputs);

        Assert.All(results, c => Assert.Null(c.Label));
    }

    [Fact]
    public void Bare_Rank1_Logits_Vector_Is_Accepted()
    {
        // Some classifiers emit [N] with no leading batch axis.
        var logits = new[] { 0.1f, 0.5f, 0.3f };
        var outputs = Outputs(new TensorShape(3), logits);

        var manifest = new ModelManifest { Task = ModelTask.ImageClassification };
        var post = new ClassificationPostprocessor(manifest, applySoftmax: false);

        var results = (List<Classification>)post.Decode(outputs);
        Assert.Equal(1, results[0].Index);   // 0.5 is the argmax
    }

    [Fact]
    public void Batch_Greater_Than_One_Throws()
    {
        var outputs = Outputs(new TensorShape(2, 3), new[] { 0f, 1f, 2f, 3f, 4f, 5f });
        var manifest = new ModelManifest { Task = ModelTask.ImageClassification };
        var post = new ClassificationPostprocessor(manifest);

        Assert.Throws<NotSupportedException>(() => post.Decode(outputs));
    }

    [Fact]
    public void NonPositive_TopK_Throws()
    {
        var manifest = new ModelManifest { Task = ModelTask.ImageClassification };
        Assert.Throws<ArgumentOutOfRangeException>(() => new ClassificationPostprocessor(manifest, topK: 0));
    }

    private static IReadOnlyDictionary<string, NamedTensor> Outputs(TensorShape shape, float[] data) =>
        new Dictionary<string, NamedTensor>
        {
            ["logits"] = new NamedTensor("logits", new Tensor<float>(shape, data)),
        };
}
