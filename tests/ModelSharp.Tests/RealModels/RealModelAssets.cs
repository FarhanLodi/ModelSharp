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
/// </summary>
internal static class RealModelAssets
{
    /// <summary>Environment variable naming the directory that holds the real model assets.</summary>
    public const string EnvVar = "MODELSHARP_MODELS_DIR";

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
