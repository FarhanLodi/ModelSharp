using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// <see cref="IModelSource"/> for Safetensors-weight repositories. Resolves either a single
/// <c>model.safetensors</c> file or a sharded model described by <c>model.safetensors.index.json</c>
/// (downloading every shard listed in its <c>weight_map</c>), together with the usual companion files
/// (<c>config.json</c> and tokenizer assets). All files are placed in one local cache directory; the
/// returned <see cref="ResolvedModel.ModelPath"/> is the primary file ModelSharp's loader opens — the
/// index file for sharded models, or the single weight file otherwise.
/// </summary>
public sealed class SafetensorsSource : IModelSource
{
    private const string IndexFile = "model.safetensors.index.json";
    private const string SingleFile = "model.safetensors";

    /// <summary>Companion (non-weight) files downloaded alongside the model when present.</summary>
    private static readonly string[] CompanionFiles =
    {
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    };

    private readonly IHubClient _client;
    private readonly IModelDownloader _downloader;
    private readonly IModelCache _cache;

    public SafetensorsSource(IHubClient client, IModelDownloader downloader, IModelCache cache)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _downloader = downloader ?? throw new ArgumentNullException(nameof(downloader));
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
    }

    /// <inheritdoc/>
    public string Kind => "safetensors";

    /// <inheritdoc/>
    public async Task<ResolvedModel> ResolveAsync(
        ModelRef reference, HubOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        var repo = reference.Repo
            ?? throw new HubException("A safetensors model reference must specify a repository (owner/repo).");
        var revision = !string.IsNullOrWhiteSpace(reference.Revision)
            ? reference.Revision
            : options.Revision;

        var token = HubCredentials.Resolve(options);

        IReadOnlyList<RepoFile> files;
        try
        {
            files = await _client.ListFilesAsync(repo, revision, token, cancellationToken).ConfigureAwait(false);
        }
        catch (HubException)
        {
            throw;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new HubException($"Failed to list files for '{repo}@{revision}'.", ex);
        }

        var available = new HashSet<string>(
            files.Select(f => f.Path).Where(p => !string.IsNullOrEmpty(p)),
            StringComparer.Ordinal);

        // The full set of repo-relative paths to download (the primary first, then weights/companions).
        var toDownload = new List<string>();
        string primaryRepoPath;

        if (HasFile(available, IndexFile))
        {
            // --- Sharded case ---------------------------------------------------------------------
            // Download the index first so we can parse its weight_map, then fetch every distinct shard.
            primaryRepoPath = IndexFile;

            var indexLocalPath = _cache.PathFor(repo, revision, IndexFile);
            await DownloadOneAsync(
                repo, revision, IndexFile, indexLocalPath, token, options,
                fileIndex: 0, fileCount: 1, cancellationToken).ConfigureAwait(false);

            var shards = ParseShardFiles(indexLocalPath);
            if (shards.Count == 0)
            {
                throw new HubException(
                    $"'{IndexFile}' in '{repo}@{revision}' did not reference any shard files in its weight_map.");
            }

            toDownload.Add(IndexFile);
            foreach (var shard in shards)
            {
                if (!HasFile(available, shard))
                {
                    throw new HubException(
                        $"Shard '{shard}' referenced by '{IndexFile}' is missing from '{repo}@{revision}'.");
                }

                if (!toDownload.Contains(shard, StringComparer.Ordinal))
                {
                    toDownload.Add(shard);
                }
            }
        }
        else
        {
            // --- Single case ----------------------------------------------------------------------
            var single = ResolveSingleFile(reference, available, repo, revision);
            primaryRepoPath = single;
            toDownload.Add(single);
        }

        // Companion files (config + tokenizer assets) when present in the repo.
        foreach (var companion in CompanionFiles)
        {
            if (HasFile(available, companion) && !toDownload.Contains(companion, StringComparer.Ordinal))
            {
                toDownload.Add(companion);
            }
        }

        // Download the remaining files honouring MaxConcurrentDownloads. All files share the primary's
        // local directory. For the sharded case the index was fetched eagerly above (so we can parse its
        // weight_map); skip re-downloading it here. DownloadOneAsync is idempotent regardless.
        var alreadyFetched = primaryRepoPath == IndexFile ? IndexFile : null;
        var remaining = toDownload
            .Where(p => alreadyFetched is null || !string.Equals(p, alreadyFetched, StringComparison.Ordinal))
            .ToList();

        var localPaths = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var path in toDownload)
        {
            localPaths[path] = _cache.PathFor(repo, revision, path);
        }

        var fileCount = remaining.Count;
        var maxConcurrency = Math.Max(1, options.MaxConcurrentDownloads);
        using var gate = new SemaphoreSlim(maxConcurrency, maxConcurrency);

        var tasks = new List<Task>(fileCount);
        for (var i = 0; i < fileCount; i++)
        {
            var repoPath = remaining[i];
            var localPath = localPaths[repoPath];
            var index = i;
            tasks.Add(Task.Run(async () =>
            {
                await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
                try
                {
                    await DownloadOneAsync(
                        repo, revision, repoPath, localPath, token, options,
                        fileIndex: index, fileCount: fileCount, cancellationToken).ConfigureAwait(false);
                }
                finally
                {
                    gate.Release();
                }
            }, cancellationToken));
        }

        try
        {
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }
        catch (HubException)
        {
            throw;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new HubException($"Failed to download safetensors bundle for '{repo}@{revision}'.", ex);
        }

        var primaryLocalPath = localPaths[primaryRepoPath];
        var directory = Path.GetDirectoryName(primaryLocalPath)
            ?? throw new HubException($"Could not determine the local directory for '{primaryLocalPath}'.");

        var allLocalPaths = toDownload.Select(p => localPaths[p]).ToList();
        return new ResolvedModel(primaryLocalPath, directory, allLocalPaths);
    }

    /// <summary>
    /// Picks the single weight file for a non-sharded repo: an explicit <see cref="ModelRef.FileHint"/>
    /// if given, otherwise <c>model.safetensors</c>, otherwise the sole <c>*.safetensors</c> in the repo.
    /// </summary>
    private static string ResolveSingleFile(
        ModelRef reference, HashSet<string> available, string repo, string revision)
    {
        if (!string.IsNullOrWhiteSpace(reference.FileHint))
        {
            var hint = NormalizePath(reference.FileHint);
            if (!HasFile(available, hint))
            {
                throw new HubException(
                    $"Requested file '{reference.FileHint}' was not found in '{repo}@{revision}'.");
            }

            return hint;
        }

        if (HasFile(available, SingleFile))
        {
            return SingleFile;
        }

        var candidates = available
            .Where(p => p.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (candidates.Count == 1)
        {
            return candidates[0];
        }

        if (candidates.Count == 0)
        {
            throw new HubException($"No '*.safetensors' weight file found in '{repo}@{revision}'.");
        }

        throw new HubException(
            $"'{repo}@{revision}' contains multiple safetensors files and no '{IndexFile}'; " +
            "specify which one via the reference's file hint.");
    }

    /// <summary>
    /// Parses a <c>model.safetensors.index.json</c> file and returns the distinct shard filenames listed
    /// under its <c>weight_map</c> object (<c>{ "weight_map": { "&lt;tensor&gt;": "&lt;shard&gt;.safetensors" } }</c>).
    /// </summary>
    private static IReadOnlyList<string> ParseShardFiles(string indexLocalPath)
    {
        string json;
        try
        {
            json = File.ReadAllText(indexLocalPath);
        }
        catch (Exception ex)
        {
            throw new HubException($"Failed to read '{indexLocalPath}'.", ex);
        }

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new HubException($"'{indexLocalPath}' is not valid JSON.", ex);
        }

        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object ||
                !doc.RootElement.TryGetProperty("weight_map", out var weightMap) ||
                weightMap.ValueKind != JsonValueKind.Object)
            {
                throw new HubException(
                    $"'{indexLocalPath}' does not contain a 'weight_map' object.");
            }

            // Preserve first-seen order while de-duplicating shard names.
            var seen = new HashSet<string>(StringComparer.Ordinal);
            var shards = new List<string>();
            foreach (var entry in weightMap.EnumerateObject())
            {
                if (entry.Value.ValueKind != JsonValueKind.String)
                {
                    continue;
                }

                var shard = entry.Value.GetString();
                if (string.IsNullOrWhiteSpace(shard))
                {
                    continue;
                }

                shard = NormalizePath(shard);
                if (seen.Add(shard))
                {
                    shards.Add(shard);
                }
            }

            return shards;
        }
    }

    private async Task DownloadOneAsync(
        string repo, string revision, string repoRelativePath, string localPath,
        string? token, HubOptions options, int fileIndex, int fileCount, CancellationToken cancellationToken)
    {
        if (!options.ForceDownload && _cache.IsCached(localPath))
        {
            return;
        }

        var directory = Path.GetDirectoryName(localPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var url = _client.ResolveUrl(repo, revision, repoRelativePath);
        var template = new DownloadProgress(repoRelativePath, 0, null, fileIndex, fileCount);

        try
        {
            await _downloader.DownloadAsync(
                url, localPath, token, options.Progress, template, cancellationToken).ConfigureAwait(false);
        }
        catch (HubException)
        {
            throw;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            throw new HubException($"Failed to download '{repoRelativePath}' from '{repo}@{revision}'.", ex);
        }
    }

    private static bool HasFile(HashSet<string> available, string path)
        => available.Contains(NormalizePath(path));

    /// <summary>Normalizes a repo-relative path to forward slashes (the hub's canonical separator).</summary>
    private static string NormalizePath(string path)
        => path.Replace('\\', '/').TrimStart('/');
}
