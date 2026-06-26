using System;
using System.Collections.Generic;
using ModelSharp.Engine;
using ModelSharp.Manifest;
using ModelSharp.Tensors;

namespace ModelSharp.Pipeline;

/// <summary>
/// The universal, manifest-driven entry point: input in, typed result out. Owns an
/// <see cref="IExecutionEngine"/> plus the pre/post processors chosen for the model's
/// task, and is engine-agnostic by construction (CPU today, GPU later — same API).
/// </summary>
public sealed class Pipeline : IDisposable
{
    private readonly IExecutionEngine _engine;
    private readonly IPreprocessor _pre;
    private readonly IPostprocessor _post;

    /// <summary>The resolved manifest describing this model.</summary>
    public ModelManifest Manifest { get; }

    public Pipeline(IExecutionEngine engine, ModelManifest manifest, IPreprocessor pre, IPostprocessor post)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        Manifest = manifest ?? throw new ArgumentNullException(nameof(manifest));
        _pre = pre ?? throw new ArgumentNullException(nameof(pre));
        _post = post ?? throw new ArgumentNullException(nameof(post));
    }

    /// <summary>Loads a model file and resolves its manifest automatically (sidecar → metadata → built-in).</summary>
    public static Pipeline Load(string modelPath) => ModelSharpPipeline.Load(modelPath);

    /// <summary>Loads a model file using an explicit manifest, bypassing manifest resolution.</summary>
    public static Pipeline Load(string modelPath, ModelManifest manifest) => ModelSharpPipeline.Load(modelPath, manifest);

    /// <summary>Runs the full input → tensors → graph → typed result path.</summary>
    public TResult Run<TResult>(object input)
    {
        IReadOnlyDictionary<string, NamedTensor> feeds = _pre.ToFeeds(input);
        IReadOnlyDictionary<string, NamedTensor> outputs = _engine.Run(feeds);
        return (TResult)_post.Decode(outputs);
    }

    /// <inheritdoc />
    public void Dispose() => _engine.Dispose();
}
