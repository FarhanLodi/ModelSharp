using System;
using System.IO;

namespace ModelSharp.Tests.RealModels;

/// <summary>
/// Resolves the on-disk directory that holds the (gitignored) real model assets the Phase A
/// opt-in integration tests run against, and offers a <see cref="TryPath"/> helper that mirrors
/// the skip pattern used by <c>MiniLmTests</c>: if the asset is absent the test no-ops instead of
/// failing, so CI stays green on machines without the model files. The tests only become live once
/// a user drops the exported model files into the models directory.
///
/// <para>The directory is taken from the <c>MODELSHARP_MODELS_DIR</c> environment variable when set,
/// otherwise it falls back to a repo-relative <c>models/</c> directory resolved from the test output
/// directory (and finally to <c>assets/models</c> under the test output, matching the MiniLM layout).
/// See <c>docs/REAL_MODELS.md</c> for how to export each model and which files go where.</para>
///
/// <para>For a one-clone-and-run experience, <see cref="TryResolveOrDownload"/> additionally dogfoods
/// the <c>ModelSharp.Hub</c> package: when a model is absent locally AND the user opts in via the
/// <c>MODELSHARP_DOWNLOAD_MODELS</c> environment variable, it self-downloads the model from the Hub
/// (which caches it under the user cache dir). Downloads are OFF by default so a plain <c>dotnet test</c>
/// stays offline and fast, and any network/Hub failure SKIPS the test cleanly rather than hard-failing.</para>
/// </summary>
internal static class RealModelAssets
{
    /// <summary>Environment variable naming the directory that holds the real model assets.</summary>
    public const string EnvVar = "MODELSHARP_MODELS_DIR";

    /// <summary>
    /// Opt-in gate for Hub auto-download. When set to <c>1</c>/<c>true</c>, a model missing locally is
    /// downloaded via <see cref="ModelSharp.Hub.ModelHub"/>. Unset (the default) keeps tests offline → skip.
    /// </summary>
    public const string DownloadEnvVar = "MODELSHARP_DOWNLOAD_MODELS";

    /// <summary>
    /// Extra gate, on top of <see cref="DownloadEnvVar"/>, required before auto-downloading LARGE
    /// (multi-GB) models (e.g. the ~5&#160;GB Mistral-7B INT4 export) so a casual opt-in run does not
    /// silently pull gigabytes. Set to <c>1</c>/<c>true</c> to allow large downloads.
    /// </summary>
    public const string DownloadLargeEnvVar = "MODELSHARP_DOWNLOAD_LARGE";

    /// <summary>
    /// The resolved models directory. Precedence: <c>MODELSHARP_MODELS_DIR</c> →
    /// repo-relative <c>models/</c> → the test output's <c>assets/models</c> directory.
    /// The directory is not required to exist; callers gate on individual asset presence.
    /// </summary>
    public static string ModelsDir { get; } = ResolveModelsDir();

    /// <summary>
    /// Resolves the absolute path of an asset by file name (or relative sub-path) under
    /// <see cref="ModelsDir"/> and reports whether it exists. Returns <c>true</c> only when the file
    /// is present, so callers can write <c>if (!RealModelAssets.TryPath("x.onnx", out var p)) return;</c>.
    /// </summary>
    /// <param name="name">The asset file name or models-dir-relative path.</param>
    /// <param name="path">The resolved absolute path (set whether or not the file exists).</param>
    /// <returns><c>true</c> when the resolved file exists on disk; otherwise <c>false</c>.</returns>
    public static bool TryPath(string name, out string path)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Asset name is required.", nameof(name));
        path = Path.Combine(ModelsDir, name);
        return File.Exists(path);
    }

    /// <summary>True when <paramref name="envVar"/> is set to <c>1</c>/<c>true</c>/<c>yes</c>/<c>on</c>.</summary>
    private static bool IsEnabled(string envVar)
    {
        string? v = Environment.GetEnvironmentVariable(envVar);
        if (string.IsNullOrWhiteSpace(v)) return false;
        v = v.Trim();
        return v.Equals("1", StringComparison.OrdinalIgnoreCase)
            || v.Equals("true", StringComparison.OrdinalIgnoreCase)
            || v.Equals("yes", StringComparison.OrdinalIgnoreCase)
            || v.Equals("on", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Whether Hub auto-download is opted in via <see cref="DownloadEnvVar"/>.</summary>
    public static bool DownloadsEnabled => IsEnabled(DownloadEnvVar);

    /// <summary>Whether LARGE-model auto-download is additionally opted in via <see cref="DownloadLargeEnvVar"/>.</summary>
    public static bool LargeDownloadsEnabled => IsEnabled(DownloadLargeEnvVar);

    /// <summary>
    /// Resolves a model file by (a) the existing local discovery (<see cref="TryPath"/>) and, when the
    /// asset is absent locally, (b) an opt-in Hub auto-download that dogfoods <see cref="ModelSharp.Hub.ModelHub"/>.
    ///
    /// <para>Precedence:
    /// <list type="number">
    ///   <item>Local: <c>MODELSHARP_MODELS_DIR</c> → repo-relative <c>models/</c> (via <see cref="TryPath"/>).</item>
    ///   <item>Hub: only when <paramref name="hubSpec"/> is non-null AND downloads are enabled
    ///   (<see cref="DownloadEnvVar"/>=1; plus <see cref="DownloadLargeEnvVar"/>=1 when
    ///   <paramref name="isLarge"/>). The Hub caches under the user cache dir, so a second run is fast.</item>
    /// </list></para>
    ///
    /// <para>Any Hub failure (no network, repo/file gone, auth, integrity) is caught and logged, and the
    /// method returns <c>false</c> so the test SKIPS cleanly — it never hard-fails on a missing asset or a
    /// network error. With downloads disabled (the default) this behaves exactly like <see cref="TryPath"/>.</para>
    /// </summary>
    /// <param name="localRelPath">The asset file name or models-dir-relative path to look for locally.</param>
    /// <param name="hubSpec">A <see cref="ModelSharp.Hub.ModelHub"/> spec (e.g. <c>owner/repo/onnx/model_q4.onnx</c>),
    /// or <c>null</c> to stay local-only (skip when absent — never an invented URL).</param>
    /// <param name="path">The resolved absolute path on success; otherwise the would-be local path.</param>
    /// <param name="isLarge">When <c>true</c>, gate the download behind the extra
    /// <see cref="DownloadLargeEnvVar"/> opt-in so a casual run does not pull multiple GB.</param>
    /// <param name="log">Optional sink for skip/diagnostic messages (typically the test's output helper).</param>
    /// <returns><c>true</c> when a usable model path is resolved (locally or freshly downloaded).</returns>
    public static bool TryResolveOrDownload(
        string localRelPath, string? hubSpec, out string path, bool isLarge = false, Action<string>? log = null)
    {
        // (a) Existing local discovery — the dev-machine / MODELSHARP_MODELS_DIR fast path.
        if (TryPath(localRelPath, out path))
            return true;

        // (b) Opt-in Hub auto-download. Off by default → behaves exactly like TryPath (skip).
        if (string.IsNullOrWhiteSpace(hubSpec))
            return false;
        if (!DownloadsEnabled)
        {
            log?.Invoke($"'{localRelPath}' not found locally and {DownloadEnvVar} is not set; skipping " +
                        $"(set {DownloadEnvVar}=1 to auto-download '{hubSpec}' via ModelSharp.Hub).");
            return false;
        }
        if (isLarge && !LargeDownloadsEnabled)
        {
            log?.Invoke($"'{localRelPath}' is a LARGE model and {DownloadLargeEnvVar} is not set; skipping " +
                        $"(set {DownloadLargeEnvVar}=1 to allow the multi-GB download of '{hubSpec}').");
            return false;
        }

        try
        {
            log?.Invoke($"'{localRelPath}' not found locally; downloading '{hubSpec}' via ModelSharp.Hub " +
                        "(cached under the user cache dir for subsequent runs)…");
            ModelSharp.Hub.ResolvedModel resolved = ModelSharp.Hub.ModelHub.Get(hubSpec);
            path = resolved.ModelPath;
            if (File.Exists(path))
            {
                log?.Invoke($"Downloaded '{hubSpec}' → {path}.");
                return true;
            }
            log?.Invoke($"Hub returned a path that does not exist ('{path}'); skipping.");
            return false;
        }
        catch (Exception ex)
        {
            // Network/Hub failures must SKIP, never hard-fail the suite.
            log?.Invoke($"Hub download of '{hubSpec}' failed ({ex.GetType().Name}: {ex.Message}); skipping.");
            return false;
        }
    }

    private static string ResolveModelsDir()
    {
        string? fromEnv = Environment.GetEnvironmentVariable(EnvVar);
        if (!string.IsNullOrWhiteSpace(fromEnv))
            return Path.GetFullPath(fromEnv);

        // Walk up from the test output directory looking for a repo-relative `models/` dir.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "models");
            if (Directory.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }

        // Fallback: the same `assets/models` layout MiniLmTests uses, under the test output dir.
        return Path.Combine(AppContext.BaseDirectory, "assets", "models");
    }
}
