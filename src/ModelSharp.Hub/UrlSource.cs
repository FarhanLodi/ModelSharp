using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// <see cref="IModelSource"/> for a model fetched from a direct <c>https://…</c> URL (no hub API).
/// The remote file is downloaded into a stable, URL-addressed cache directory
/// (<c>{root}/url/{sha256(url)}/{filename}</c>). When the URL points at an <c>.onnx</c> graph, sibling
/// external-data files (<c>{url}.data</c> and <c>{url}_data</c>) are best-effort fetched into the same
/// directory so a direct-URL ONNX with external data still loads — missing siblings are ignored.
/// </summary>
public sealed class UrlSource : IModelSource
{
    private readonly IModelDownloader _downloader;
    private readonly IModelCache _cache;

    public UrlSource(IModelDownloader downloader, IModelCache cache)
    {
        _downloader = downloader ?? throw new ArgumentNullException(nameof(downloader));
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
    }

    /// <inheritdoc />
    public string Kind => "url";

    /// <inheritdoc />
    public async Task<ResolvedModel> ResolveAsync(
        ModelRef reference, HubOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        var url = reference.Url
            ?? throw new HubException("url source requires a direct 'https://…' reference, but Url was null");

        var token = HubCredentials.Resolve(options);

        // Derive a stable, collision-resistant local directory from the full URL, and a human-readable
        // file name from the URL's last path segment. Different URLs sharing a file name stay isolated.
        var fileName = DeriveFileName(url);
        var directory = Path.Combine(_cache.RootDirectory, "url", StableHash(url));
        var modelPath = Path.GetFullPath(Path.Combine(directory, fileName));

        Directory.CreateDirectory(directory);

        var localPaths = new List<string>(3);

        // 1. The primary model file.
        await DownloadOneAsync(url, modelPath, fileName, options, token, cancellationToken)
            .ConfigureAwait(false);
        localPaths.Add(modelPath);

        // 2. Best-effort external-data siblings for a direct-URL .onnx. ONNX external data is referenced
        //    by file name next to the graph, so we co-locate any sibling that actually exists; a 404 (or
        //    any other download failure) just means the sibling is absent and is silently skipped.
        if (fileName.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            foreach (var suffix in new[] { ".data", "_data" })
            {
                var siblingUrl = url + suffix;
                var siblingName = fileName + suffix;
                var siblingPath = Path.GetFullPath(Path.Combine(directory, siblingName));

                if (await TryDownloadOptionalAsync(
                        siblingUrl, siblingPath, siblingName, options, token, cancellationToken)
                    .ConfigureAwait(false))
                {
                    localPaths.Add(siblingPath);
                }
            }
        }

        return new ResolvedModel(modelPath, directory, localPaths);
    }

    /// <summary>Downloads <paramref name="url"/> to <paramref name="localPath"/>, reusing a valid cached copy.</summary>
    private async Task DownloadOneAsync(
        string url, string localPath, string fileName,
        HubOptions options, string? token, CancellationToken cancellationToken)
    {
        if (!options.ForceDownload && _cache.IsCached(localPath))
            return;

        var template = new DownloadProgress(fileName, 0, null, 0, 1);
        await _downloader.DownloadAsync(url, localPath, token, options.Progress, template, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Best-effort download of an optional sibling. Returns true when the file ends up present locally;
    /// swallows 404s and any other download failure (the sibling simply does not exist). Cancellation
    /// is always propagated.
    /// </summary>
    private async Task<bool> TryDownloadOptionalAsync(
        string url, string localPath, string fileName,
        HubOptions options, string? token, CancellationToken cancellationToken)
    {
        if (!options.ForceDownload && _cache.IsCached(localPath))
            return true;

        try
        {
            var template = new DownloadProgress(fileName, 0, null, 0, 1);
            await _downloader.DownloadAsync(url, localPath, token, options.Progress, template, cancellationToken)
                .ConfigureAwait(false);
            return File.Exists(localPath);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch
        {
            // Optional sibling: absent or unreachable. Remove any partial artefact and report "not present".
            TryDelete(localPath);
            return false;
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
            // Best-effort cleanup only.
        }
    }

    /// <summary>
    /// Derives a safe local file name from a URL's last path segment, falling back to "model" when the
    /// URL has no usable segment. Strips any query/fragment and replaces path-illegal characters.
    /// </summary>
    private static string DeriveFileName(string url)
    {
        var candidate = url;

        // Prefer the structured path when the URL parses; otherwise fall back to raw string trimming.
        if (Uri.TryCreate(url, UriKind.Absolute, out var uri))
        {
            candidate = uri.AbsolutePath;
        }
        else
        {
            // Drop query/fragment manually for relative or malformed inputs.
            int cut = candidate.IndexOfAny(new[] { '?', '#' });
            if (cut >= 0)
                candidate = candidate.Substring(0, cut);
        }

        candidate = candidate.Replace('\\', '/').TrimEnd('/');
        int slash = candidate.LastIndexOf('/');
        string lastSegment = slash >= 0 ? candidate.Substring(slash + 1) : candidate;

        // URL-decode so percent-encoded names (e.g. "model%20q4.onnx") become readable, then sanitize.
        lastSegment = Uri.UnescapeDataString(lastSegment).Trim();

        string sanitized = Sanitize(lastSegment);
        return sanitized.Length == 0 ? "model" : sanitized;
    }

    /// <summary>Replaces characters illegal in a file name with '_'.</summary>
    private static string Sanitize(string name)
    {
        if (string.IsNullOrEmpty(name))
            return string.Empty;

        char[] invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(name.Length);
        foreach (char c in name)
            sb.Append(Array.IndexOf(invalid, c) >= 0 ? '_' : c);

        return sb.ToString();
    }

    /// <summary>A stable, filesystem-safe directory key for a URL: the first 32 hex chars of SHA-256(url).</summary>
    private static string StableHash(string url)
    {
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(url));
        var sb = new StringBuilder(32);
        // 16 bytes -> 32 hex chars is ample to avoid collisions while keeping the path short.
        for (int i = 0; i < 16; i++)
            sb.Append(hash[i].ToString("x2"));

        return sb.ToString();
    }
}
