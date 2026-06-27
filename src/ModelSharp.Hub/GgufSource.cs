using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// Resolves GGUF model references (llama.cpp / Ollama-style single-file weights). A GGUF repo on the
/// Hub typically holds the same model at several quantisation levels (e.g. <c>Q4_K_M</c>,
/// <c>Q8_0</c>) and, for large models, a single quant split across numbered shards
/// (<c>model-00001-of-00003.gguf</c> …). This source selects the right <c>.gguf</c> (disambiguating by
/// file hint or quant substring), pulls in every shard of a split file, and best-effort fetches a
/// sibling <c>tokenizer.json</c> / <c>config.json</c> when present.
/// </summary>
public sealed class GgufSource : IModelSource
{
    // Matches the canonical split-GGUF naming: "<base>-00001-of-00003.gguf".
    // Group "base" is the shared prefix, "index" this shard's number, "total" the shard count.
    private static readonly Regex SplitPattern = new(
        @"^(?<base>.+)-(?<index>\d{5})-of-(?<total>\d{5})\.gguf$",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    private static readonly string[] OptionalCompanions = { "tokenizer.json", "config.json" };

    private readonly IHubClient _client;
    private readonly IModelDownloader _downloader;
    private readonly IModelCache _cache;

    public GgufSource(IHubClient client, IModelDownloader downloader, IModelCache cache)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _downloader = downloader ?? throw new ArgumentNullException(nameof(downloader));
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
    }

    /// <inheritdoc />
    public string Kind => "gguf";

    /// <inheritdoc />
    public async Task<ResolvedModel> ResolveAsync(
        ModelRef reference, HubOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (string.IsNullOrWhiteSpace(reference.Repo))
            throw new HubException("GGUF resolution requires a 'owner/repo' reference.");

        var repo = reference.Repo!;
        var revision = string.IsNullOrWhiteSpace(reference.Revision) ? options.Revision : reference.Revision;
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
            throw new HubException($"Failed to list files for GGUF repo '{repo}' at '{revision}'.", ex);
        }

        var ggufFiles = files
            .Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (ggufFiles.Count == 0)
            throw new HubException($"No '.gguf' files found in repo '{repo}' at '{revision}'.");

        var chosen = SelectGguf(reference.FileHint, ggufFiles, repo, revision);

        // A split GGUF resolves to the full set of shards; a plain file resolves to just itself.
        var toDownload = ExpandShards(chosen, ggufFiles, repo, revision, out var primary);

        // Best-effort companions: GGUF is self-contained, so a missing tokenizer/config is not fatal.
        foreach (var name in OptionalCompanions)
        {
            var companion = files.FirstOrDefault(f =>
                string.Equals(f.Path, name, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrEmpty(companion.Path) &&
                !toDownload.Any(f => string.Equals(f.Path, companion.Path, StringComparison.OrdinalIgnoreCase)))
            {
                toDownload.Add(companion);
            }
        }

        var localPaths = await DownloadAllAsync(repo, revision, toDownload, token, options, cancellationToken)
            .ConfigureAwait(false);

        var primaryLocal = _cache.PathFor(repo, revision, primary.Path);
        var directory = System.IO.Path.GetDirectoryName(primaryLocal) ?? _cache.RootDirectory;

        return new ResolvedModel(primaryLocal, directory, localPaths);
    }

    /// <summary>
    /// Picks the single <c>.gguf</c> the caller meant:
    /// (1) an explicit file-hint that names a <c>.gguf</c> wins; otherwise
    /// (2) if exactly one <c>.gguf</c> exists, use it; otherwise
    /// (3) if the hint matches a quant substring (e.g. <c>Q4_K_M</c>) uniquely, use that;
    /// (4) otherwise throw, listing the candidates so the caller can disambiguate.
    /// </summary>
    private static RepoFile SelectGguf(string? fileHint, List<RepoFile> ggufFiles, string repo, string revision)
    {
        // (1) Explicit .gguf file hint (exact path or bare file name).
        if (!string.IsNullOrWhiteSpace(fileHint) &&
            fileHint!.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            var hit = ggufFiles.FirstOrDefault(f =>
                string.Equals(f.Path, fileHint, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(FileName(f.Path), FileName(fileHint), StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrEmpty(hit.Path))
                return hit;

            throw new HubException(
                $"GGUF file '{fileHint}' not found in repo '{repo}' at '{revision}'. " +
                $"Available: {FormatCandidates(ggufFiles)}.");
        }

        // (2) Unambiguous single file.
        if (ggufFiles.Count == 1)
            return ggufFiles[0];

        // (3) Quant-substring hint (e.g. "Q4_K_M" / "q8_0").
        if (!string.IsNullOrWhiteSpace(fileHint))
        {
            var matches = ggufFiles
                .Where(f => FileName(f.Path).Contains(fileHint!, StringComparison.OrdinalIgnoreCase))
                .ToList();

            // Prefer the primary shard of a split set so we don't pick "...-00002-of-..." by accident.
            var primaries = matches.Where(IsPrimaryShardOrSingle).ToList();
            var effective = primaries.Count > 0 ? primaries : matches;

            if (effective.Count == 1)
                return effective[0];

            if (effective.Count == 0)
                throw new HubException(
                    $"No '.gguf' in repo '{repo}' at '{revision}' matches hint '{fileHint}'. " +
                    $"Available: {FormatCandidates(ggufFiles)}.");

            throw new HubException(
                $"Hint '{fileHint}' is ambiguous in repo '{repo}' at '{revision}': it matches " +
                $"{FormatCandidates(effective)}. Specify a full '.gguf' file name.");
        }

        // (4) Multiple quants, no hint: the caller must choose.
        throw new HubException(
            $"Repo '{repo}' at '{revision}' contains multiple '.gguf' files; specify which one " +
            $"(by file name or quant level). Available: {FormatCandidates(ggufFiles)}.");
    }

    /// <summary>
    /// If <paramref name="chosen"/> is one shard of a split GGUF, returns every shard in the set
    /// (ordered by index) and sets <paramref name="primary"/> to the first shard
    /// (<c>-00001-of-…</c>). Otherwise returns just the chosen file.
    /// </summary>
    private static List<RepoFile> ExpandShards(
        RepoFile chosen, List<RepoFile> ggufFiles, string repo, string revision, out RepoFile primary)
    {
        var match = SplitPattern.Match(FileName(chosen.Path));
        if (!match.Success)
        {
            primary = chosen;
            return new List<RepoFile> { chosen };
        }

        var baseName = match.Groups["base"].Value;
        var total = match.Groups["total"].Value;
        var dir = DirOf(chosen.Path);

        // Collect all shards sharing the same base + total, ordered by their numeric index.
        var shards = ggufFiles
            .Select(f => (File: f, M: SplitPattern.Match(FileName(f.Path))))
            .Where(x => x.M.Success
                && string.Equals(DirOf(x.File.Path), dir, StringComparison.OrdinalIgnoreCase)
                && string.Equals(x.M.Groups["base"].Value, baseName, StringComparison.OrdinalIgnoreCase)
                && string.Equals(x.M.Groups["total"].Value, total, StringComparison.OrdinalIgnoreCase))
            .OrderBy(x => int.Parse(x.M.Groups["index"].Value))
            .Select(x => x.File)
            .ToList();

        var expectedCount = int.Parse(total);
        if (shards.Count != expectedCount)
            throw new HubException(
                $"Split GGUF '{baseName}-*-of-{total}.gguf' in repo '{repo}' at '{revision}' is incomplete: " +
                $"found {shards.Count} of {expectedCount} shards.");

        primary = shards[0];
        return shards;
    }

    private async Task<IReadOnlyList<string>> DownloadAllAsync(
        string repo, string revision, IReadOnlyList<RepoFile> files, string? token,
        HubOptions options, CancellationToken cancellationToken)
    {
        var maxConcurrency = options.MaxConcurrentDownloads > 0 ? options.MaxConcurrentDownloads : 1;
        using var gate = new SemaphoreSlim(maxConcurrency);

        var results = new string[files.Count];
        var tasks = new List<Task>(files.Count);

        for (var i = 0; i < files.Count; i++)
        {
            var index = i;
            var file = files[i];
            tasks.Add(Task.Run(async () =>
            {
                await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
                try
                {
                    results[index] = await DownloadOneAsync(
                        repo, revision, file, index, files.Count, token, options, cancellationToken)
                        .ConfigureAwait(false);
                }
                finally
                {
                    gate.Release();
                }
            }, cancellationToken));
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
        return results;
    }

    private async Task<string> DownloadOneAsync(
        string repo, string revision, RepoFile file, int fileIndex, int fileCount,
        string? token, HubOptions options, CancellationToken cancellationToken)
    {
        var localPath = _cache.PathFor(repo, revision, file.Path);

        if (!options.ForceDownload && _cache.IsCached(localPath, file))
            return localPath;

        var url = _client.ResolveUrl(repo, revision, file.Path);
        var template = new DownloadProgress(file.Path, 0, file.Size, fileIndex, fileCount);

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
            throw new HubException($"Failed to download '{file.Path}' from repo '{repo}' at '{revision}'.", ex);
        }

        return localPath;
    }

    private static bool IsPrimaryShardOrSingle(RepoFile file)
    {
        var m = SplitPattern.Match(FileName(file.Path));
        return !m.Success || int.Parse(m.Groups["index"].Value) == 1;
    }

    private static string FileName(string path)
    {
        var slash = path.LastIndexOf('/');
        return slash >= 0 ? path[(slash + 1)..] : path;
    }

    private static string DirOf(string path)
    {
        var slash = path.LastIndexOf('/');
        return slash >= 0 ? path[..slash] : string.Empty;
    }

    private static string FormatCandidates(IEnumerable<RepoFile> files) =>
        string.Join(", ", files.Select(f => f.Path).OrderBy(p => p, StringComparer.OrdinalIgnoreCase));
}
