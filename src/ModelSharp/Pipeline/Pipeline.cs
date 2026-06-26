using System;
using System.Collections.Generic;
using ModelSharp.Engine;
using ModelSharp.Generation;
using ModelSharp.Manifest;
using ModelSharp.Tensors;

namespace ModelSharp.Pipeline;

/// <summary>
/// The universal, manifest-driven entry point: input in, typed result out. Owns an
/// <see cref="IExecutionEngine"/> plus the pre/post processors chosen for the model's
/// task, and is engine-agnostic by construction (CPU today, GPU later — same API).
///
/// <para>Two shapes of model are supported: single-shot tasks (embeddings, vision, …) go through
/// <see cref="Run{TResult}"/>; autoregressive text generation goes through <see cref="Generate"/> /
/// <see cref="GenerateStream"/>, which drive a <see cref="TextGenerator"/> loop rather than a single
/// <c>ToFeeds → Run → Decode</c> pass. A pipeline is built for exactly one of the two paths.</para>
/// </summary>
public sealed class Pipeline : IDisposable
{
    private readonly IExecutionEngine _engine;
    private readonly IPreprocessor? _pre;
    private readonly IPostprocessor? _post;

    // Set only on the TextGeneration path (constructed by ModelSharpPipeline.Build); null otherwise.
    private readonly TextGenerator? _generator;
    private readonly TextGenerationProcessor? _generation;

    /// <summary>The resolved manifest describing this model.</summary>
    public ModelManifest Manifest { get; }

    /// <summary>Constructs a single-shot pipeline (embeddings, vision, …) wired with pre/post processors.</summary>
    public Pipeline(IExecutionEngine engine, ModelManifest manifest, IPreprocessor pre, IPostprocessor post)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        Manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        _pre = pre ?? throw new ArgumentNullException(nameof(pre));
        _post = post ?? throw new ArgumentNullException(nameof(post));
    }

    /// <summary>
    /// Constructs a text-generation pipeline that drives <paramref name="generator"/> with the
    /// tokenizer and default config held by <paramref name="generation"/>. Use the overload taking
    /// pre/post processors for every other task.
    /// </summary>
    public Pipeline(IExecutionEngine engine, ModelManifest manifest, TextGenerator generator, TextGenerationProcessor generation)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        Manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _generation = generation ?? throw new ArgumentNullException(nameof(generation));
    }

    /// <summary>Loads a model file and resolves its manifest automatically (sidecar → metadata → built-in).</summary>
    public static Pipeline Load(string modelPath) => ModelSharpPipeline.Load(modelPath);

    /// <summary>Loads a model file using an explicit manifest, bypassing manifest resolution.</summary>
    public static Pipeline Load(string modelPath, ModelManifest manifest) => ModelSharpPipeline.Load(modelPath, manifest);

    /// <summary>Runs the full input → tensors → graph → typed result path.</summary>
    public TResult Run<TResult>(object input)
    {
        if (_pre is null || _post is null)
            throw new ModelSharpException(
                $"Run<T> is not available for a '{Manifest.Task}' pipeline. " +
                "This is an autoregressive text-generation model; call Generate / GenerateStream instead.");

        IReadOnlyDictionary<string, NamedTensor> feeds = _pre.ToFeeds(input);
        IReadOnlyDictionary<string, NamedTensor> outputs = _engine.Run(feeds);
        return (TResult)_post.Decode(outputs);
    }

    /// <summary>
    /// Generates a completion for <paramref name="prompt"/> and returns the decoded text (the new
    /// tokens only, not the prompt). Only available on a text-generation pipeline.
    /// </summary>
    /// <param name="prompt">The prompt text; tokenized with the model's BPE tokenizer.</param>
    /// <param name="config">Decoding parameters; defaults to the manifest-derived configuration.</param>
    public string Generate(string prompt, GenerationConfig? config = null)
    {
        (TextGenerator generator, TextGenerationProcessor generation) = RequireGeneration();
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));

        IReadOnlyList<long> promptIds = generation.Encode(prompt);
        IReadOnlyList<long> generated = generator.Generate(promptIds, config ?? generation.DefaultConfig);
        return generation.Decode(generated);
    }

    /// <summary>
    /// Streams the completion for <paramref name="prompt"/> as text fragments, one per generated token,
    /// decoded incrementally. Only available on a text-generation pipeline.
    /// </summary>
    /// <param name="prompt">The prompt text; tokenized with the model's BPE tokenizer.</param>
    /// <param name="config">Decoding parameters; defaults to the manifest-derived configuration.</param>
    public IEnumerable<string> GenerateStream(string prompt, GenerationConfig? config = null)
    {
        (TextGenerator generator, TextGenerationProcessor generation) = RequireGeneration();
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));

        IReadOnlyList<long> promptIds = generation.Encode(prompt);
        return StreamDecoded(generator, generation, promptIds, config ?? generation.DefaultConfig);
    }

    /// <summary>
    /// Decodes streamed tokens incrementally. Each step decodes the whole generated prefix and yields
    /// only the newly produced suffix, so multi-byte / multi-token characters surface correctly once
    /// their final byte arrives rather than emitting replacement characters mid-sequence.
    /// </summary>
    private static IEnumerable<string> StreamDecoded(
        TextGenerator generator, TextGenerationProcessor generation, IReadOnlyList<long> promptIds, GenerationConfig config)
    {
        var produced = new List<long>();
        string previous = string.Empty;
        foreach (long token in generator.GenerateStream(promptIds, config))
        {
            produced.Add(token);
            string current = generation.Decode(produced);
            if (current.Length > previous.Length)
            {
                yield return current.Substring(previous.Length);
                previous = current;
            }
        }
    }

    private (TextGenerator Generator, TextGenerationProcessor Generation) RequireGeneration()
    {
        if (_generator is null || _generation is null)
            throw new ModelSharpException(
                $"Generate is only available for TextGeneration models; this pipeline's task is '{Manifest.Task}'. " +
                "Use Run<T> for single-shot tasks.");
        return (_generator, _generation);
    }

    /// <inheritdoc />
    public void Dispose() => _engine.Dispose();
}
