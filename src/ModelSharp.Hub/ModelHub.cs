using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// The model-acquisition facade. Resolves a model specifier (a friendly alias, a Hugging Face repo, a
/// <c>gguf:</c>/<c>safetensors:</c> repo, or a direct URL) and downloads it — plus its companion files
/// (ONNX external-data shards, tokenizer, config) — into a local cache, returning the local paths.
///
/// <example>
/// <code>
/// var hub = new ModelHub();
/// ResolvedModel m = await hub.DownloadAsync("onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx");
/// using var pipeline = ModelSharp.Pipeline.Pipeline.Load(m.ModelPath);
/// // …or in one line: using var p = HubPipeline.Load("qwen2.5-0.5b-int4");
/// </code>
/// </example>
/// </summary>
public sealed class ModelHub
{
    private readonly IModelDownloader _downloader;
    private readonly IModelCache _cache;
    private readonly IHubClient _client;
    private readonly Dictionary<string, IModelSource> _sources;

    /// <summary>Creates a hub. <paramref name="options"/> seeds the cache directory and Hub endpoint.</summary>
    public ModelHub(HubOptions? options = null, HttpClient? http = null)
    {
        _downloader = new HttpDownloader(http);
        _cache = new ModelCache(options);
        _client = new HuggingFaceClient(http, options?.Endpoint ?? "https://huggingface.co");

        var hf = new HuggingFaceSource(_client, _downloader, _cache);
        _sources = new Dictionary<string, IModelSource>(StringComparer.OrdinalIgnoreCase)
        {
            [hf.Kind] = hf,
            ["gguf"] = new GgufSource(_client, _downloader, _cache),
            ["safetensors"] = new SafetensorsSource(_client, _downloader, _cache),
            ["url"] = new UrlSource(_downloader, _cache),
        };
    }

    /// <summary>The resolved local cache directory.</summary>
    public string CacheDirectory => _cache.RootDirectory;

    /// <summary>Downloads (or reuses cached) the model bundle for <paramref name="spec"/>.</summary>
    public async Task<ResolvedModel> DownloadAsync(string spec, HubOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new HubOptions();
        ModelRef reference = ModelRefParser.Parse(ModelAliases.Resolve(spec), options.Revision);
        if (!_sources.TryGetValue(reference.Kind, out IModelSource? source))
            throw new HubException($"No download source registered for kind '{reference.Kind}'.");
        return await source.ResolveAsync(reference, options, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Synchronous convenience over <see cref="DownloadAsync"/>.</summary>
    public ResolvedModel Download(string spec, HubOptions? options = null)
        => DownloadAsync(spec, options).GetAwaiter().GetResult();

    /// <summary>One-shot static download using a fresh hub.</summary>
    public static ResolvedModel Get(string spec, HubOptions? options = null)
        => new ModelHub(options).Download(spec, options);

    /// <summary>One-shot static async download using a fresh hub.</summary>
    public static Task<ResolvedModel> GetAsync(string spec, HubOptions? options = null, CancellationToken cancellationToken = default)
        => new ModelHub(options).DownloadAsync(spec, options, cancellationToken);
}
