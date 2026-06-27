using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// <see cref="IModelSource"/> for ONNX bundles hosted on the Hugging Face Hub. Resolves the primary
/// <c>.onnx</c> file (honouring <see cref="ModelRef.FileHint"/>), discovers its companion files
/// (external-data shards, tokenizer, config), and downloads them — bounded by
/// <see cref="HubOptions.MaxConcurrentDownloads"/> — placing every companion in the same local
/// directory as the <c>.onnx</c> so ModelSharp's loader finds the external-data next to the graph.
/// </summary>
public sealed class HuggingFaceSource : IModelSource
{
    private readonly IHubClient _client;
    private readonly IModelDownloader _downloader;
    private readonly IModelCache _cache;

    public HuggingFaceSource(IHubClient client, IModelDownloader downloader, IModelCache cache)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _downloader = downloader ?? throw new ArgumentNullException(nameof(downloader));
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
    }

    /// <inheritdoc />
    public string Kind => "hf";

    /// <inheritdoc />
    public async Task<ResolvedModel> ResolveAsync(
        ModelRef reference, HubOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        var repo = reference.Repo
            ?? throw new HubException("hf source requires a 'owner/repo' reference, but Repo was null");
        var revision = reference.Revision;

        var token = HubCredentials.Resolve(options);

        // 1. List the repo so we can pick the model file and discover companions.
        var files = await _client.ListFilesAsync(repo, revision, token, cancellationToken)
            .ConfigureAwait(false);

        // 2. Pick the primary .onnx (honouring an explicit file hint).
        var onnxPath = OnnxBundleResolver.PickModelFile(files, reference.FileHint)
            ?? throw new HubException($"no .onnx in {repo}");

        // 3. Build the ordered, de-duplicated fetch list: the .onnx first, then its companions.
        var fetchList = new List<string> { onnxPath };
        var seen = new HashSet<string>(StringComparer.Ordinal) { onnxPath };
        foreach (var companion in OnnxBundleResolver.CompanionFiles(files, onnxPath))
        {
            if (seen.Add(companion))
                fetchList.Add(companion);
        }

        // Index the listing so we can look up the RepoFile (size / sha) for each path.
        var byPath = new Dictionary<string, RepoFile>(StringComparer.Ordinal);
        foreach (var f in files)
            byPath[f.Path] = f;

        // The .onnx's local directory: every companion must land here, regardless of its repo subdir,
        // so external-data shards + tokenizer/config sit next to the graph for ModelSharp's loader.
        var onnxLocalPath = _cache.PathFor(repo, revision, onnxPath);
        var bundleDirectory = Path.GetDirectoryName(onnxLocalPath)!;

        // Resolve each repo path -> local path. The .onnx uses the cache's natural path; companions are
        // forced into the .onnx's directory using only their file name (different repo subdirs collapse).
        var count = fetchList.Count;
        var localPaths = new string[count];
        for (var i = 0; i < count; i++)
        {
            var repoPath = fetchList[i];
            localPaths[i] = i == 0
                ? onnxLocalPath
                : Path.Combine(bundleDirectory, Path.GetFileName(repoPath));
        }

        // 4 & 5. Download (or reuse) each file, bounded by MaxConcurrentDownloads.
        var maxConcurrency = Math.Max(1, options.MaxConcurrentDownloads);
        using var gate = new SemaphoreSlim(maxConcurrency, maxConcurrency);

        var tasks = new Task[count];
        for (var i = 0; i < count; i++)
        {
            tasks[i] = DownloadOneAsync(
                repo, revision, fetchList[i], localPaths[i], i, count,
                byPath.TryGetValue(fetchList[i], out var rf) ? rf : null,
                token, options, gate, cancellationToken);
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);

        // 6. The primary file is the .onnx (index 0); Directory is its dir; Files are all local paths.
        return new ResolvedModel(localPaths[0], bundleDirectory, localPaths);
    }

    private async Task DownloadOneAsync(
        string repo, string revision, string repoPath, string localPath,
        int index, int count, RepoFile? expected,
        string? token, HubOptions options, SemaphoreSlim gate, CancellationToken cancellationToken)
    {
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var hasIntegrity = expected is { } e && (e.Size.HasValue || e.Sha256 is not null);

            // Reuse a valid cached copy unless a re-download is forced. When integrity checking is on
            // and we have metadata, the cached copy must also pass verification to be reused.
            if (!options.ForceDownload && _cache.IsCached(localPath, expected))
            {
                if (!options.VerifyIntegrity || !hasIntegrity ||
                    await IntegrityVerifier.VerifyAsync(localPath, expected!.Value, cancellationToken)
                        .ConfigureAwait(false))
                {
                    return;
                }
            }

            var directory = Path.GetDirectoryName(localPath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);

            var size = expected?.Size;
            var url = _client.ResolveUrl(repo, revision, repoPath);
            var template = new DownloadProgress(repoPath, 0, size, index, count);

            await _downloader.DownloadAsync(
                url, localPath, token, options.Progress, template, cancellationToken)
                .ConfigureAwait(false);

            // Post-download integrity check when requested and metadata is available.
            if (options.VerifyIntegrity && hasIntegrity)
            {
                var ok = await IntegrityVerifier.VerifyAsync(localPath, expected!.Value, cancellationToken)
                    .ConfigureAwait(false);
                if (!ok)
                    throw new HubException(
                        $"integrity check failed for '{repoPath}' from {repo}@{revision} (size/sha256 mismatch)");
            }
        }
        finally
        {
            gate.Release();
        }
    }
}
