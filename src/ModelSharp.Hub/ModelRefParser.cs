using System;

namespace ModelSharp.Hub;

/// <summary>
/// Parses model specifier strings into a structured <see cref="ModelRef"/>. Pure logic, no I/O.
/// </summary>
/// <remarks>
/// <para>Accepted grammar (whitespace-trimmed):</para>
/// <list type="bullet">
///   <item><description>
///     <b>Direct URL</b> — anything beginning with <c>http://</c> or <c>https://</c> becomes a
///     <c>url</c> reference, e.g. <c>"https://example.com/model.onnx"</c>.
///   </description></item>
///   <item><description>
///     <b>Scheme prefix</b> — an optional leading <c>gguf:</c>, <c>safetensors:</c>, <c>hf:</c>, or
///     <c>url:</c> sets the kind explicitly. <c>url:</c> treats the remainder as a direct URL; the
///     others parse the remainder as a repo spec.
///   </description></item>
///   <item><description>
///     <b>Repo spec</b> — <c>owner/repo</c> optionally followed by an <c>@revision</c> suffix and/or a
///     trailing sub-path that becomes the <see cref="ModelRef.FileHint"/>. The first two path segments
///     form the repo; everything after is the file hint.
///   </description></item>
/// </list>
/// <para>Examples:</para>
/// <list type="bullet">
///   <item><description>
///     <c>"onnx-community/Qwen2.5-0.5B-Instruct"</c> →
///     <c>Kind=hf, Repo=onnx-community/Qwen2.5-0.5B-Instruct, Revision=main, FileHint=null</c>.
///   </description></item>
///   <item><description>
///     <c>"hf:owner/repo@main/onnx/model_q4.onnx"</c> →
///     <c>Kind=hf, Repo=owner/repo, Revision=main, FileHint=onnx/model_q4.onnx</c>.
///   </description></item>
///   <item><description>
///     <c>"gguf:TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf"</c> →
///     <c>Kind=gguf, Repo=TheBloke/Llama-2-7B-GGUF, Revision=main, FileHint=llama-2-7b.Q4_K_M.gguf</c>.
///   </description></item>
///   <item><description>
///     <c>"https://example.com/model.onnx"</c> →
///     <c>Kind=url, Url=https://example.com/model.onnx, Repo=null</c>.
///   </description></item>
/// </list>
/// </remarks>
public static class ModelRefParser
{
    /// <summary>
    /// Parses a model specifier into a <see cref="ModelRef"/>.
    /// </summary>
    /// <param name="spec">The specifier string (URL, scheme-prefixed, or bare repo spec).</param>
    /// <param name="defaultRevision">Revision to use when the spec carries no <c>@revision</c> suffix.</param>
    /// <returns>The parsed <see cref="ModelRef"/>.</returns>
    /// <exception cref="HubException">When <paramref name="spec"/> is empty or malformed for its kind.</exception>
    /// <example>
    /// <code>
    /// ModelRefParser.Parse("onnx-community/Qwen2.5-0.5B-Instruct");
    /// ModelRefParser.Parse("hf:owner/repo@main/onnx/model_q4.onnx");
    /// ModelRefParser.Parse("gguf:TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf");
    /// ModelRefParser.Parse("https://example.com/model.onnx");
    /// </code>
    /// </example>
    public static ModelRef Parse(string spec, string defaultRevision = "main")
    {
        if (string.IsNullOrWhiteSpace(spec))
            throw new HubException("Model spec is empty. Provide a repo (owner/repo), a scheme-prefixed spec (hf:/gguf:/safetensors:/url:), or a direct https:// URL.");

        spec = spec.Trim();

        // Direct URL (no scheme prefix).
        if (IsHttpUrl(spec))
            return new ModelRef(Kind: "url", Repo: null, Revision: defaultRevision, FileHint: null, Url: spec);

        // Optional scheme prefix.
        string? kind = null;
        string remainder = spec;
        if (TryStripScheme(spec, "url:", out var afterUrl))
        {
            var url = afterUrl.Trim();
            if (url.Length == 0)
                throw new HubException("'url:' scheme requires a URL after the prefix, e.g. url:https://example.com/model.onnx.");
            return new ModelRef(Kind: "url", Repo: null, Revision: defaultRevision, FileHint: null, Url: url);
        }
        if (TryStripScheme(spec, "gguf:", out var afterGguf)) { kind = "gguf"; remainder = afterGguf; }
        else if (TryStripScheme(spec, "safetensors:", out var afterSt)) { kind = "safetensors"; remainder = afterSt; }
        else if (TryStripScheme(spec, "hf:", out var afterHf)) { kind = "hf"; remainder = afterHf; }

        remainder = remainder.Trim();

        // A direct URL hiding behind a non-url scheme is not valid here.
        if (IsHttpUrl(remainder))
            throw new HubException($"Spec '{spec}' looks like a URL but uses a non-url scheme. Use 'url:' (or a bare https:// URL) for direct URLs.");

        if (remainder.Length == 0)
            throw new HubException($"Spec '{spec}' is missing a repo after the scheme prefix. Expected scheme:owner/repo[@revision][/sub/path].");

        // Extract an optional @revision suffix. The '@' separates repo+path from the revision,
        // e.g. "owner/repo@v1.0" or "owner/repo@<sha>/onnx/model.onnx".
        // The revision itself may be followed by a further sub-path, which becomes part of FileHint.
        string revision = defaultRevision;
        string pathPart = remainder;
        var atIndex = remainder.IndexOf('@');
        if (atIndex >= 0)
        {
            var beforeAt = remainder.Substring(0, atIndex);
            var afterAt = remainder.Substring(atIndex + 1);

            // afterAt = "<revision>" or "<revision>/<sub/path>".
            var slashAfterAt = afterAt.IndexOf('/');
            string revPart;
            string trailingPath;
            if (slashAfterAt >= 0)
            {
                revPart = afterAt.Substring(0, slashAfterAt);
                trailingPath = afterAt.Substring(slashAfterAt + 1);
            }
            else
            {
                revPart = afterAt;
                trailingPath = string.Empty;
            }

            revPart = revPart.Trim();
            if (revPart.Length == 0)
                throw new HubException($"Spec '{spec}' has an empty revision after '@'. Use owner/repo@<branch|tag|sha>.");

            revision = revPart;
            // Recompose the path without the revision: repo(+leading path) + trailing sub-path.
            pathPart = trailingPath.Length > 0
                ? beforeAt.TrimEnd('/') + "/" + trailingPath
                : beforeAt;
        }

        pathPart = pathPart.Trim().Trim('/');
        if (pathPart.Length == 0)
            throw new HubException($"Spec '{spec}' is missing a repo. Expected owner/repo[@revision][/sub/path].");

        // First two segments are owner/repo; the rest is the file hint sub-path.
        var segments = pathPart.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length < 2)
            throw new HubException(
                $"Spec '{spec}' does not name a repo. Accepted forms: 'owner/repo', 'owner/repo@revision', " +
                "'owner/repo/sub/path/file', 'hf:owner/repo', 'gguf:owner/repo/file.gguf', " +
                "'safetensors:owner/repo/model.safetensors', 'url:https://…', or a direct https:// URL.");

        var repo = segments[0] + "/" + segments[1];
        string? fileHint = segments.Length > 2
            ? string.Join('/', segments, 2, segments.Length - 2)
            : null;

        // Infer kind from a file-hint extension when no scheme set it.
        if (kind is null)
        {
            var hint = fileHint ?? repo;
            if (hint.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                kind = "gguf";
            else if (hint.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                kind = "safetensors";
            else
                kind = "hf";
        }

        // Validate hub kinds: repo must be exactly owner/repo (one slash, two non-empty parts).
        if (kind is "hf" or "gguf" or "safetensors")
        {
            var slashCount = CountChar(repo, '/');
            var parts = repo.Split('/');
            if (slashCount != 1 || parts.Length != 2 || parts[0].Length == 0 || parts[1].Length == 0)
                throw new HubException(
                    $"Spec '{spec}' has an invalid repo '{repo}' for kind '{kind}'. " +
                    "Expected exactly 'owner/repo'. Accepted forms: 'owner/repo', 'owner/repo@revision', " +
                    "'owner/repo/sub/path/file', 'hf:owner/repo', 'gguf:owner/repo/file.gguf', " +
                    "'safetensors:owner/repo/model.safetensors', or a direct https:// URL.");
        }

        return new ModelRef(Kind: kind, Repo: repo, Revision: revision, FileHint: fileHint, Url: null);
    }

    private static bool IsHttpUrl(string s) =>
        s.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
        s.StartsWith("https://", StringComparison.OrdinalIgnoreCase);

    private static bool TryStripScheme(string s, string scheme, out string remainder)
    {
        if (s.StartsWith(scheme, StringComparison.OrdinalIgnoreCase))
        {
            remainder = s.Substring(scheme.Length);
            return true;
        }
        remainder = s;
        return false;
    }

    private static int CountChar(string s, char c)
    {
        var count = 0;
        foreach (var ch in s)
            if (ch == c) count++;
        return count;
    }
}
