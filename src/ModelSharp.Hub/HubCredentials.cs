using System;
using System.IO;

namespace ModelSharp.Hub;

/// <summary>
/// Resolves a Hugging Face access token from (in order) explicit options, well-known environment
/// variables, and the Hugging Face CLI token file — mirroring the behaviour of the official tooling.
/// </summary>
public static class HubCredentials
{
    /// <summary>
    /// Resolves the Hugging Face token to use for hub requests, honouring
    /// <see cref="HubOptions.Token"/> first and falling back to the environment / CLI token file.
    /// </summary>
    /// <param name="options">Hub options whose <see cref="HubOptions.Token"/> takes precedence.</param>
    /// <returns>The first non-empty token found, or <c>null</c> when none is available.</returns>
    public static string? Resolve(HubOptions options)
        => Resolve(options?.Token);

    /// <summary>
    /// Resolves the Hugging Face token, in order:
    /// <list type="number">
    /// <item><description><paramref name="explicitToken"/> (e.g. <see cref="HubOptions.Token"/>).</description></item>
    /// <item><description>The <c>HF_TOKEN</c> environment variable.</description></item>
    /// <item><description>The <c>HUGGING_FACE_HUB_TOKEN</c> environment variable.</description></item>
    /// <item><description>The <c>HUGGINGFACE_TOKEN</c> environment variable.</description></item>
    /// <item><description>The HF CLI token file: <c>{HF_HOME}/token</c> when <c>HF_HOME</c> is set,
    /// otherwise <c>~/.cache/huggingface/token</c>, otherwise <c>~/.huggingface/token</c>.</description></item>
    /// </list>
    /// All values are trimmed; missing files never throw. Returns <c>null</c> when nothing is found.
    /// </summary>
    /// <param name="explicitToken">An explicitly supplied token that wins over all other sources.</param>
    /// <returns>The first non-empty token found, or <c>null</c> when none is available.</returns>
    public static string? Resolve(string? explicitToken = null)
    {
        if (NormalizeOrNull(explicitToken) is { } token)
        {
            return token;
        }

        foreach (var name in new[] { "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN" })
        {
            if (NormalizeOrNull(Environment.GetEnvironmentVariable(name)) is { } envToken)
            {
                return envToken;
            }
        }

        return ReadTokenFile();
    }

    /// <summary>Reads and trims the first line of the HF CLI token file, if it exists.</summary>
    private static string? ReadTokenFile()
    {
        foreach (var path in EnumerateTokenFilePaths())
        {
            if (path is null)
            {
                continue;
            }

            try
            {
                if (!File.Exists(path))
                {
                    continue;
                }

                using var reader = new StreamReader(path);
                var line = reader.ReadLine();
                if (NormalizeOrNull(line) is { } fileToken)
                {
                    return fileToken;
                }
            }
            catch (IOException)
            {
                // Treat any IO failure as "no token here" and continue to the next candidate.
            }
            catch (UnauthorizedAccessException)
            {
                // Permission problems are non-fatal; keep looking.
            }
        }

        return null;
    }

    /// <summary>
    /// Yields the candidate token-file paths in priority order. When <c>HF_HOME</c> is set only
    /// <c>{HF_HOME}/token</c> is considered; otherwise the two default locations are tried.
    /// </summary>
    private static System.Collections.Generic.IEnumerable<string?> EnumerateTokenFilePaths()
    {
        var hfHome = NormalizeOrNull(Environment.GetEnvironmentVariable("HF_HOME"));
        if (hfHome is not null)
        {
            yield return Path.Combine(hfHome, "token");
            yield break;
        }

        var home = HomeDirectoryOrNull();
        if (home is not null)
        {
            yield return Path.Combine(home, ".cache", "huggingface", "token");
            yield return Path.Combine(home, ".huggingface", "token");
        }
    }

    /// <summary>Best-effort resolution of the current user's home directory.</summary>
    private static string? HomeDirectoryOrNull()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrEmpty(home))
        {
            return home;
        }

        return NormalizeOrNull(Environment.GetEnvironmentVariable("HOME"))
            ?? NormalizeOrNull(Environment.GetEnvironmentVariable("USERPROFILE"));
    }

    /// <summary>Trims the value and returns it, or <c>null</c> when it is null/blank after trimming.</summary>
    private static string? NormalizeOrNull(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        var trimmed = value.Trim();
        return trimmed.Length == 0 ? null : trimmed;
    }
}
