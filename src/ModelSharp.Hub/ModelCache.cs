using System;
using System.IO;

namespace ModelSharp.Hub;

/// <summary>
/// A revision-addressed local cache of downloaded files using a Hugging-Face-style on-disk layout:
/// <c>{root}/models--{owner}--{name}/snapshots/{revision}/{repoRelativePath}</c>.
/// </summary>
/// <remarks>
/// The root directory is resolved (and created if missing) in this precedence:
/// <list type="number">
///   <item><see cref="HubOptions.CacheDirectory"/></item>
///   <item>env <c>MODELSHARP_CACHE</c></item>
///   <item>env <c>HF_HOME</c> (uses <c>{HF_HOME}/hub</c>)</item>
///   <item>env <c>XDG_CACHE_HOME</c> (uses <c>{XDG_CACHE_HOME}/modelsharp</c>)</item>
///   <item><c>~/.cache/modelsharp</c></item>
/// </list>
/// </remarks>
public sealed class ModelCache : IModelCache
{
    /// <summary>Creates a cache, resolving the root from <paramref name="options"/>, environment, or the user home.</summary>
    public ModelCache(HubOptions? options = null)
        : this(options?.CacheDirectory)
    {
    }

    /// <summary>Creates a cache rooted at <paramref name="root"/> (or resolved from environment/home when null).</summary>
    public ModelCache(string? root)
    {
        RootDirectory = ResolveRoot(root);
        Directory.CreateDirectory(RootDirectory);
    }

    /// <inheritdoc/>
    public string RootDirectory { get; }

    /// <summary>Resolves the cache root following the documented precedence. Does not create the directory.</summary>
    private static string ResolveRoot(string? explicitRoot)
    {
        if (!string.IsNullOrWhiteSpace(explicitRoot))
            return Path.GetFullPath(explicitRoot);

        string? modelSharpCache = Environment.GetEnvironmentVariable("MODELSHARP_CACHE");
        if (!string.IsNullOrWhiteSpace(modelSharpCache))
            return Path.GetFullPath(modelSharpCache);

        string? hfHome = Environment.GetEnvironmentVariable("HF_HOME");
        if (!string.IsNullOrWhiteSpace(hfHome))
            return Path.GetFullPath(Path.Combine(hfHome, "hub"));

        string? xdgCache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (!string.IsNullOrWhiteSpace(xdgCache))
            return Path.GetFullPath(Path.Combine(xdgCache, "modelsharp"));

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (string.IsNullOrWhiteSpace(home))
            home = Directory.GetCurrentDirectory();

        return Path.GetFullPath(Path.Combine(home, ".cache", "modelsharp"));
    }

    /// <inheritdoc/>
    public string PathFor(string repo, string revision, string repoRelativePath)
    {
        ArgumentNullException.ThrowIfNull(repo);
        ArgumentNullException.ThrowIfNull(repoRelativePath);

        string repoDir = "models--" + SanitizeRepo(repo);
        string rev = SanitizeComponent(string.IsNullOrWhiteSpace(revision) ? "main" : revision);

        // Build the snapshot root, then append the (possibly nested) repo-relative path.
        string snapshotRoot = Path.Combine(RootDirectory, repoDir, "snapshots", rev);

        string relative = NormalizeRelative(repoRelativePath);
        string combined = string.IsNullOrEmpty(relative)
            ? snapshotRoot
            : Path.Combine(snapshotRoot, relative);

        return Path.GetFullPath(combined);
    }

    /// <inheritdoc/>
    public bool IsCached(string localPath, RepoFile? expected = null)
    {
        if (string.IsNullOrWhiteSpace(localPath))
            return false;

        var info = new FileInfo(localPath);
        if (!info.Exists)
            return false;

        if (expected is { Size: { } size } && info.Length != size)
            return false;

        return true;
    }

    /// <summary>
    /// Ensures the parent directory of <paramref name="localPath"/> exists, returning the path unchanged
    /// so callers can write to it immediately.
    /// </summary>
    public static string EnsureDirectoryFor(string localPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(localPath);

        string? parent = Path.GetDirectoryName(Path.GetFullPath(localPath));
        if (!string.IsNullOrEmpty(parent))
            Directory.CreateDirectory(parent);

        return localPath;
    }

    /// <summary>Turns "owner/name" into a single safe directory component "owner--name".</summary>
    private static string SanitizeRepo(string repo)
    {
        // A repo may legitimately contain a single "owner/name" slash; collapse all slashes to "--".
        string trimmed = repo.Trim().Trim('/');
        string replaced = trimmed.Replace('\\', '/').Replace("/", "--");
        return SanitizeComponent(replaced);
    }

    /// <summary>Normalizes a repo-relative path into safe nested subdirectories under the snapshot root.</summary>
    private static string NormalizeRelative(string repoRelativePath)
    {
        string trimmed = repoRelativePath.Replace('\\', '/').Trim().Trim('/');
        if (trimmed.Length == 0)
            return string.Empty;

        string[] parts = trimmed.Split('/', StringSplitOptions.RemoveEmptyEntries);
        for (int i = 0; i < parts.Length; i++)
        {
            // Drop any "." and ".." segments so the path stays inside the snapshot directory.
            if (parts[i] == "." || parts[i] == "..")
            {
                parts[i] = string.Empty;
                continue;
            }
            parts[i] = SanitizeComponent(parts[i]);
        }

        return Path.Combine(Array.FindAll(parts, static p => p.Length > 0));
    }

    /// <summary>Replaces characters illegal in a path component with '_'.</summary>
    private static string SanitizeComponent(string component)
    {
        if (string.IsNullOrEmpty(component))
            return component;

        char[] invalid = Path.GetInvalidFileNameChars();
        char[] buffer = component.ToCharArray();
        bool changed = false;
        for (int i = 0; i < buffer.Length; i++)
        {
            if (Array.IndexOf(invalid, buffer[i]) >= 0)
            {
                buffer[i] = '_';
                changed = true;
            }
        }

        return changed ? new string(buffer) : component;
    }
}
