using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using ModelSharp.Manifest;

namespace ModelSharp.Pipeline;

/// <summary>
/// The context handed to a processor factory: the resolved manifest plus the engine's
/// input/output binding names. Factories use this to build a processor that maps inputs
/// onto exactly the feeds the model declares (e.g. some BERT models omit token_type_ids).
/// </summary>
public sealed record ProcessorContext(
    ModelManifest Manifest,
    IReadOnlyList<string> InputNames,
    IReadOnlyList<string> OutputNames);

/// <summary>
/// Selects the pre/post processors for a model from its manifest task. Adapters
/// (ModelSharp.ImageSharp for vision, the built-in text path here for embeddings)
/// register factories keyed by <see cref="ModelTask"/>; the high-level loader then
/// asks the registry for the processors that match a resolved manifest.
/// Registrations are thread-safe and last-registration-wins.
/// </summary>
public static class ProcessorRegistry
{
    private static readonly ConcurrentDictionary<ModelTask, Func<ProcessorContext, IPreprocessor>> Preprocessors = new();
    private static readonly ConcurrentDictionary<ModelTask, Func<ProcessorContext, IPostprocessor>> Postprocessors = new();

    static ProcessorRegistry()
    {
        // Built-in core processors that depend only on the dependency-free core
        // (WordPiece tokenizer + mean-pool). Vision/audio adapters register their own.
        RegisterPreprocessor(ModelTask.Embedding, ctx => new TextEmbeddingPreprocessor(ctx));
        RegisterPostprocessor(ModelTask.Embedding, _ => new MeanPoolEmbeddingPostprocessor());
    }

    /// <summary>Registers (or replaces) the preprocessor factory for a task.</summary>
    public static void RegisterPreprocessor(ModelTask task, Func<ProcessorContext, IPreprocessor> factory)
    {
        if (factory is null) throw new ArgumentNullException(nameof(factory));
        Preprocessors[task] = factory;
    }

    /// <summary>Registers (or replaces) the postprocessor factory for a task.</summary>
    public static void RegisterPostprocessor(ModelTask task, Func<ProcessorContext, IPostprocessor> factory)
    {
        if (factory is null) throw new ArgumentNullException(nameof(factory));
        Postprocessors[task] = factory;
    }

    /// <summary>Creates the preprocessor for the context's manifest task.</summary>
    public static IPreprocessor CreatePreprocessor(ProcessorContext ctx)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));
        if (Preprocessors.TryGetValue(ctx.Manifest.Task, out Func<ProcessorContext, IPreprocessor>? factory))
            return factory(ctx);
        throw new ModelSharpException(
            $"No preprocessor registered for task '{ctx.Manifest.Task}'. " +
            "Register one via ProcessorRegistry.RegisterPreprocessor, or reference the adapter that provides it.");
    }

    /// <summary>Creates the postprocessor for the context's manifest task.</summary>
    public static IPostprocessor CreatePostprocessor(ProcessorContext ctx)
    {
        if (ctx is null) throw new ArgumentNullException(nameof(ctx));
        if (Postprocessors.TryGetValue(ctx.Manifest.Task, out Func<ProcessorContext, IPostprocessor>? factory))
            return factory(ctx);
        throw new ModelSharpException(
            $"No postprocessor registered for task '{ctx.Manifest.Task}'. " +
            "Register one via ProcessorRegistry.RegisterPostprocessor, or reference the adapter that provides it.");
    }
}
