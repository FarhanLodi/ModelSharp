using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// One-line "download &amp; run" entry point: resolves a model from the hub (or cache) and returns a ready
/// <c>Pipeline</c>. Lives in <c>ModelSharp.Hub</c> so the zero-dependency core stays acquisition-free.
///
/// <example><code>
/// using ModelSharp.Hub;
/// using var pipeline = HubPipeline.Load("all-minilm-l6-v2");
/// float[] e = pipeline.Run&lt;float[]&gt;("A man is playing a guitar.");
/// </code></example>
/// </summary>
public static class HubPipeline
{
    /// <summary>Downloads (or reuses cached) the model for <paramref name="spec"/> and loads it as a pipeline.</summary>
    public static global::ModelSharp.Pipeline.Pipeline Load(string spec, HubOptions? options = null)
    {
        ResolvedModel model = ModelHub.Get(spec, options);
        return global::ModelSharp.Pipeline.Pipeline.Load(model.ModelPath);
    }

    /// <summary>Async variant of <see cref="Load"/>.</summary>
    public static async Task<global::ModelSharp.Pipeline.Pipeline> LoadAsync(
        string spec, HubOptions? options = null, CancellationToken cancellationToken = default)
    {
        ResolvedModel model = await ModelHub.GetAsync(spec, options, cancellationToken).ConfigureAwait(false);
        return global::ModelSharp.Pipeline.Pipeline.Load(model.ModelPath);
    }
}
