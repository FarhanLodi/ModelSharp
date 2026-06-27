using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ModelSharp.Hub;

/// <summary>
/// <see cref="IHubClient"/> backed by the public Hugging Face Hub HTTP API.
/// Lists repo files via the paginated <c>/api/models/{repo}/tree/{revision}</c> endpoint and
/// builds <c>/{repo}/resolve/{revision}/{path}</c> download URLs. Pure BCL: only
/// <see cref="System.Net.Http.HttpClient"/> and <see cref="System.Text.Json"/>.
/// </summary>
public sealed partial class HuggingFaceClient : IHubClient
{
    private static readonly string UserAgentValue =
        "ModelSharp.Hub/0.2 (+https://github.com/FarhanLodi/ModelSharp)";

    private readonly HttpClient _client;
    private readonly string _endpoint;

    /// <summary>
    /// Creates a client for the given Hub endpoint.
    /// </summary>
    /// <param name="client">HTTP client to use; when null an internal one is created.</param>
    /// <param name="endpoint">Base Hub URL (override for mirrors / enterprise endpoints).</param>
    public HuggingFaceClient(HttpClient? client = null, string endpoint = "https://huggingface.co")
    {
        if (string.IsNullOrWhiteSpace(endpoint))
            throw new ArgumentException("Endpoint must be a non-empty URL.", nameof(endpoint));

        _endpoint = endpoint.TrimEnd('/');
        _client = client ?? new HttpClient();
    }

    /// <inheritdoc />
    public string ResolveUrl(string repo, string revision, string repoRelativePath)
    {
        if (string.IsNullOrWhiteSpace(repo))
            throw new ArgumentException("Repo must be non-empty.", nameof(repo));
        if (string.IsNullOrWhiteSpace(revision))
            throw new ArgumentException("Revision must be non-empty.", nameof(revision));
        if (string.IsNullOrWhiteSpace(repoRelativePath))
            throw new ArgumentException("Path must be non-empty.", nameof(repoRelativePath));

        var revSegment = Uri.EscapeDataString(revision);
        var pathSegments = EscapePath(repoRelativePath);
        return $"{_endpoint}/{repo}/resolve/{revSegment}/{pathSegments}";
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<RepoFile>> ListFilesAsync(
        string repo, string revision, string? token = null, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(repo))
            throw new ArgumentException("Repo must be non-empty.", nameof(repo));
        if (string.IsNullOrWhiteSpace(revision))
            throw new ArgumentException("Revision must be non-empty.", nameof(revision));

        var revSegment = Uri.EscapeDataString(revision);
        var nextUrl =
            $"{_endpoint}/api/models/{repo}/tree/{revSegment}?recursive=true";

        var files = new List<RepoFile>();

        while (nextUrl is not null)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, nextUrl);
            request.Headers.UserAgent.TryParseAdd(UserAgentValue);
            if (!string.IsNullOrEmpty(token))
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

            HttpResponseMessage response;
            try
            {
                response = await _client
                    .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) when (ex is not HubException)
            {
                throw new HubException(
                    $"Failed to list files for '{repo}' at '{revision}': {ex.Message}", ex);
            }

            using (response)
            {
                if (!response.IsSuccessStatusCode)
                    throw await BuildErrorAsync(response, repo, revision, cancellationToken)
                        .ConfigureAwait(false);

                await using var stream = await response.Content
                    .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);

                List<TreeEntry>? entries;
                try
                {
                    entries = await JsonSerializer
                        .DeserializeAsync(stream, HubJsonContext.Default.ListTreeEntry, cancellationToken)
                        .ConfigureAwait(false);
                }
                catch (JsonException ex)
                {
                    throw new HubException(
                        $"Failed to parse tree listing for '{repo}' at '{revision}': {ex.Message}", ex);
                }

                if (entries is not null)
                {
                    foreach (var entry in entries)
                    {
                        if (!string.Equals(entry.Type, "file", StringComparison.OrdinalIgnoreCase))
                            continue;
                        if (string.IsNullOrEmpty(entry.Path))
                            continue;

                        long? size = entry.Lfs?.Size ?? entry.Size;
                        string? sha256 = entry.Lfs?.Oid;
                        files.Add(new RepoFile(entry.Path, size, sha256));
                    }
                }

                nextUrl = GetNextLink(response);
            }
        }

        return files;
    }

    /// <summary>Escapes each path segment while preserving '/' separators (no double-encoding).</summary>
    private static string EscapePath(string path)
    {
        var normalized = path.Replace('\\', '/').TrimStart('/');
        var segments = normalized.Split('/');
        for (int i = 0; i < segments.Length; i++)
            segments[i] = Uri.EscapeDataString(segments[i]);
        return string.Join('/', segments);
    }

    /// <summary>
    /// Parses the RFC 5988 <c>Link</c> header for a <c>rel="next"</c> URL, if present.
    /// </summary>
    private static string? GetNextLink(HttpResponseMessage response)
    {
        if (!response.Headers.TryGetValues("Link", out var values))
            return null;

        foreach (var headerValue in values)
        {
            // A single Link header may contain multiple comma-separated links.
            foreach (var part in SplitLinks(headerValue))
            {
                int lt = part.IndexOf('<');
                int gt = part.IndexOf('>');
                if (lt < 0 || gt <= lt)
                    continue;

                var url = part.Substring(lt + 1, gt - lt - 1).Trim();
                var paramsPart = part.Substring(gt + 1);
                if (paramsPart.Contains("rel=\"next\"", StringComparison.OrdinalIgnoreCase)
                    || paramsPart.Contains("rel=next", StringComparison.OrdinalIgnoreCase))
                {
                    return url.Length == 0 ? null : url;
                }
            }
        }

        return null;
    }

    /// <summary>Splits a Link header value on commas that separate distinct links (not inside &lt;…&gt;).</summary>
    private static IEnumerable<string> SplitLinks(string header)
    {
        int depth = 0, start = 0;
        for (int i = 0; i < header.Length; i++)
        {
            char c = header[i];
            if (c == '<') depth++;
            else if (c == '>') { if (depth > 0) depth--; }
            else if (c == ',' && depth == 0)
            {
                yield return header.Substring(start, i - start);
                start = i + 1;
            }
        }
        if (start < header.Length)
            yield return header.Substring(start);
    }

    private static async Task<HubException> BuildErrorAsync(
        HttpResponseMessage response, string repo, string revision, CancellationToken ct)
    {
        string body;
        try
        {
            body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
        }
        catch
        {
            body = string.Empty;
        }

        var detail = string.IsNullOrWhiteSpace(body) ? string.Empty : $" Server said: {Truncate(body, 400)}";

        return response.StatusCode switch
        {
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden => new HubException(
                $"Authentication/authorization failed listing files for '{repo}' at '{revision}' " +
                $"(HTTP {(int)response.StatusCode}). The repo may be gated or private; supply a valid token.{detail}"),
            HttpStatusCode.NotFound => new HubException(
                $"Repo or revision not found: '{repo}' at '{revision}' (HTTP 404).{detail}"),
            _ => new HubException(
                $"Failed to list files for '{repo}' at '{revision}' (HTTP {(int)response.StatusCode}).{detail}"),
        };
    }

    private static string Truncate(string value, int max)
        => value.Length <= max ? value : value.Substring(0, max) + "…";

    private sealed class TreeEntry
    {
        [JsonPropertyName("type")]
        public string? Type { get; set; }

        [JsonPropertyName("path")]
        public string? Path { get; set; }

        [JsonPropertyName("size")]
        public long? Size { get; set; }

        [JsonPropertyName("lfs")]
        public LfsInfo? Lfs { get; set; }
    }

    private sealed class LfsInfo
    {
        [JsonPropertyName("oid")]
        public string? Oid { get; set; }

        [JsonPropertyName("size")]
        public long? Size { get; set; }
    }

    [JsonSerializable(typeof(List<TreeEntry>))]
    [JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
    private sealed partial class HubJsonContext : JsonSerializerContext
    {
    }
}
