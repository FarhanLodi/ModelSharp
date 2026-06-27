using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>Raised for any model-hub acquisition failure (network, auth, integrity, resolution).</summary>
public sealed class HubException : Exception
{
    public HubException(string message) : base(message) { }
    public HubException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>
/// Configuration for hub downloads. All optional with sensible defaults: the cache lives under
/// <c>MODELSHARP_CACHE</c> / <c>HF_HOME</c> / <c>~/.cache/modelsharp</c>; the token is read from
/// <c>HF_TOKEN</c> / the HF CLI token file when not set here.
/// </summary>
public sealed class HubOptions
{
    /// <summary>Root cache directory. When null, resolved by the cache implementation.</summary>
    public string? CacheDirectory { get; set; }

    /// <summary>Hugging Face access token for gated/private repos. When null, resolved from env / token file.</summary>
    public string? Token { get; set; }

    /// <summary>Git revision (branch, tag, or commit SHA). Defaults to "main".</summary>
    public string Revision { get; set; } = "main";

    /// <summary>Re-download even if a cached copy exists.</summary>
    public bool ForceDownload { get; set; }

    /// <summary>Verify file integrity (size / sha256 / ETag) after download when metadata is available.</summary>
    public bool VerifyIntegrity { get; set; } = true;

    /// <summary>Max concurrent file downloads within a bundle.</summary>
    public int MaxConcurrentDownloads { get; set; } = 4;

    /// <summary>Optional progress sink; receives per-file and aggregate progress.</summary>
    public IProgress<DownloadProgress>? Progress { get; set; }

    /// <summary>Base URL for the Hugging Face Hub (override for mirrors / enterprise endpoints).</summary>
    public string Endpoint { get; set; } = "https://huggingface.co";
}

/// <summary>A progress update for a downloading file (and the overall bundle).</summary>
public readonly record struct DownloadProgress(
    string File,
    long BytesDownloaded,
    long? TotalBytes,
    int FileIndex,
    int FileCount)
{
    /// <summary>Fraction of this file complete in [0,1], or null when the total size is unknown.</summary>
    public double? Fraction => TotalBytes is > 0 ? (double)BytesDownloaded / TotalBytes.Value : null;
}

/// <summary>A single file within a repo listing.</summary>
/// <param name="Path">Repo-relative path (e.g. <c>onnx/model.onnx</c>).</param>
/// <param name="Size">Size in bytes when known.</param>
/// <param name="Sha256">Expected SHA-256 (LFS pointer) when known.</param>
public readonly record struct RepoFile(string Path, long? Size = null, string? Sha256 = null);

/// <summary>
/// A parsed model reference. Accepts shorthand like <c>"onnx-community/Qwen2.5-0.5B-Instruct"</c>,
/// <c>"hf:owner/repo@rev"</c>, <c>"gguf:owner/repo/file.gguf"</c>, or a direct <c>https://…</c> URL.
/// </summary>
/// <param name="Kind">The resolved source kind (hf, gguf, safetensors, url).</param>
/// <param name="Repo">owner/repo for hub sources, or null for direct URLs.</param>
/// <param name="Revision">Git revision for hub sources.</param>
/// <param name="FileHint">An optional specific file or sub-path within the repo (e.g. <c>onnx/model_q4.onnx</c>).</param>
/// <param name="Url">The full URL for direct-URL sources.</param>
public readonly record struct ModelRef(
    string Kind, string? Repo, string Revision, string? FileHint, string? Url);

/// <summary>
/// The result of resolving + downloading a model: the local path of the primary model file plus all
/// the companion files (external-data shards, tokenizer, config) in the same local directory.
/// </summary>
/// <param name="ModelPath">Absolute local path to the primary model file (.onnx / .gguf / .safetensors).</param>
/// <param name="Directory">Local directory containing the full bundle.</param>
/// <param name="Files">All downloaded local file paths.</param>
public sealed record ResolvedModel(string ModelPath, string Directory, IReadOnlyList<string> Files);

/// <summary>Downloads a single remote file to a local path, with resume, retry, and progress.</summary>
public interface IModelDownloader
{
    /// <summary>
    /// Downloads <paramref name="url"/> to <paramref name="destinationPath"/>. Resumes a partial file
    /// when the server supports range requests. Reports progress via <paramref name="progress"/>.
    /// </summary>
    Task DownloadAsync(
        string url, string destinationPath, string? token = null,
        IProgress<DownloadProgress>? progress = null, DownloadProgress template = default,
        CancellationToken cancellationToken = default);
}

/// <summary>A content/revision-addressed local cache of downloaded files.</summary>
public interface IModelCache
{
    /// <summary>The resolved root cache directory.</summary>
    string RootDirectory { get; }

    /// <summary>Returns the local path a repo file should occupy in the cache (does not download).</summary>
    string PathFor(string repo, string revision, string repoRelativePath);

    /// <summary>True if a valid cached copy already exists (optionally validated against <paramref name="expected"/>).</summary>
    bool IsCached(string localPath, RepoFile? expected = null);
}

/// <summary>Talks to a model-hub HTTP API: lists repo files and forms file download URLs.</summary>
public interface IHubClient
{
    /// <summary>Lists the files in a repo at a revision.</summary>
    Task<IReadOnlyList<RepoFile>> ListFilesAsync(
        string repo, string revision, string? token = null, CancellationToken cancellationToken = default);

    /// <summary>Builds the resolve URL for a repo-relative file.</summary>
    string ResolveUrl(string repo, string revision, string repoRelativePath);
}

/// <summary>
/// Resolves a <see cref="ModelRef"/> of a particular kind into a downloaded <see cref="ResolvedModel"/>
/// (primary file + all companions). One implementation per source kind (HF/ONNX, GGUF, safetensors, URL).
/// </summary>
public interface IModelSource
{
    /// <summary>The kind string this source handles (matches <see cref="ModelRef.Kind"/>).</summary>
    string Kind { get; }

    /// <summary>Downloads (or reuses cached) all files for the model and returns the resolved bundle.</summary>
    Task<ResolvedModel> ResolveAsync(ModelRef reference, HubOptions options, CancellationToken cancellationToken = default);
}
