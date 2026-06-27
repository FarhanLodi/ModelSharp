using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;

namespace ModelSharp.Hub;

/// <summary>
/// A pure-managed <see cref="IModelDownloader"/> built on <see cref="HttpClient"/>. Streams the body to a
/// <c>.part</c> file, resumes via HTTP <c>Range</c> requests when the server supports them, retries transient
/// failures with exponential backoff, and atomically promotes the completed file into place.
/// </summary>
public sealed class HttpDownloader : IModelDownloader
{
    /// <summary>Copy buffer / progress-report granularity (1 MiB).</summary>
    private const int BufferSize = 1 << 20;

    /// <summary>Maximum number of attempts (1 initial + up to 4 retries) for transient failures.</summary>
    private const int MaxAttempts = 5;

    /// <summary>
    /// Shared client used when the caller does not supply one. Configured to follow redirects automatically
    /// and with no per-request timeout (large files are governed by <see cref="CancellationToken"/> instead).
    /// </summary>
    private static readonly HttpClient SharedClient = CreateDefaultClient();

    private readonly HttpClient _client;

    /// <summary>
    /// Creates a downloader. When <paramref name="client"/> is null a process-wide shared client is used;
    /// pass an explicit client to control proxies, handlers, or timeouts.
    /// </summary>
    public HttpDownloader(HttpClient? client = null) => _client = client ?? SharedClient;

    private static HttpClient CreateDefaultClient()
    {
        var handler = new SocketsHttpHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 10,
            AutomaticDecompression = DecompressionMethods.None,
            PooledConnectionLifetime = TimeSpan.FromMinutes(2),
        };
        return new HttpClient(handler)
        {
            // Disable the overall timeout; long streaming downloads rely on the CancellationToken.
            Timeout = Timeout.InfiniteTimeSpan,
        };
    }

    /// <inheritdoc />
    public async Task DownloadAsync(
        string url, string destinationPath, string? token = null,
        IProgress<DownloadProgress>? progress = null, DownloadProgress template = default,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(url))
            throw new ArgumentException("URL must be non-empty.", nameof(url));
        if (string.IsNullOrWhiteSpace(destinationPath))
            throw new ArgumentException("Destination path must be non-empty.", nameof(destinationPath));

        var fullDestination = Path.GetFullPath(destinationPath);
        var directory = Path.GetDirectoryName(fullDestination);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var partPath = fullDestination + ".part";

        Exception? lastError = null;
        for (var attempt = 1; attempt <= MaxAttempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                await DownloadOnceAsync(url, fullDestination, partPath, token, progress, template, cancellationToken)
                    .ConfigureAwait(false);
                return;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                // Caller-requested cancellation is never a transient error: surface it directly.
                throw;
            }
            catch (Exception ex) when (IsTransient(ex) && attempt < MaxAttempts)
            {
                lastError = ex;
            }
            catch (HubException)
            {
                // Non-retryable hub failure (e.g. 401/403/404) or the final attempt: propagate as-is.
                throw;
            }
            catch (Exception ex)
            {
                // Final transient attempt exhausted, or an unexpected error.
                throw new HubException(
                    $"Failed to download '{url}' to '{fullDestination}' after {attempt} attempt(s): {ex.Message}", ex);
            }

            // Exponential backoff with a small jitter: ~0.5s, 1s, 2s, 4s.
            var delayMs = (int)(500 * Math.Pow(2, attempt - 1));
            delayMs += Random.Shared.Next(0, 250);
            await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);
        }

        // Loop exhausted all attempts on transient failures.
        throw new HubException(
            $"Failed to download '{url}' to '{fullDestination}' after {MaxAttempts} attempts.",
            lastError ?? new HttpRequestException("Unknown download failure."));
    }

    /// <summary>Performs a single download attempt, resuming from an existing <c>.part</c> file when possible.</summary>
    private async Task DownloadOnceAsync(
        string url, string fullDestination, string partPath, string? token,
        IProgress<DownloadProgress>? progress, DownloadProgress template, CancellationToken cancellationToken)
    {
        long resumeOffset = 0;
        if (File.Exists(partPath))
        {
            try { resumeOffset = new FileInfo(partPath).Length; }
            catch (IOException) { resumeOffset = 0; }
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (!string.IsNullOrEmpty(token))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        if (resumeOffset > 0)
            request.Headers.Range = new RangeHeaderValue(resumeOffset, null);

        using var response = await _client
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        // Decide how to open the local file based on the server's response.
        FileMode fileMode;
        long startOffset;
        if (resumeOffset > 0 && response.StatusCode == HttpStatusCode.PartialContent)
        {
            // Server honoured the Range request: append to the existing partial file.
            fileMode = FileMode.Append;
            startOffset = resumeOffset;
        }
        else if (resumeOffset > 0 && response.StatusCode == HttpStatusCode.OK)
        {
            // Server ignored the Range request (full 200): discard the partial and restart.
            fileMode = FileMode.Create;
            startOffset = 0;
            resumeOffset = 0;
        }
        else
        {
            fileMode = FileMode.Create;
            startOffset = 0;
        }

        if (!response.IsSuccessStatusCode)
        {
            var status = (int)response.StatusCode;
            var message =
                $"Download of '{url}' failed with HTTP {status} ({response.ReasonPhrase ?? response.StatusCode.ToString()}).";
            // 5xx and 429 are transient and bubble up as HttpRequestException so the retry loop catches them;
            // everything else (401/403/404/...) is a hard failure surfaced as HubException immediately.
            if (status == 429 || status >= 500)
                throw new HttpRequestException(message);
            throw new HubException(message);
        }

        // contentLength is the size of *this* response body (the remaining bytes when resuming).
        var contentLength = response.Content.Headers.ContentLength;
        long? totalBytes = contentLength.HasValue ? contentLength.Value + startOffset : template.TotalBytes;

        await using var responseStream = await response.Content
            .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);

        var totalSoFar = startOffset;
        long sinceLastReport = 0;

        // Emit an initial progress tick so consumers see the resume baseline.
        progress?.Report(template with { BytesDownloaded = totalSoFar, TotalBytes = totalBytes });

        await using (var fileStream = new FileStream(
            partPath, fileMode, FileAccess.Write, FileShare.None,
            BufferSize, FileOptions.Asynchronous))
        {
            var buffer = new byte[BufferSize];
            int read;
            while ((read = await responseStream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                totalSoFar += read;
                sinceLastReport += read;

                if (progress is not null && sinceLastReport >= BufferSize)
                {
                    progress.Report(template with { BytesDownloaded = totalSoFar, TotalBytes = totalBytes });
                    sinceLastReport = 0;
                }
            }

            await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
        }

        // Final progress tick at the true byte count.
        progress?.Report(template with
        {
            BytesDownloaded = totalSoFar,
            TotalBytes = totalBytes ?? totalSoFar,
        });

        // Atomically promote the completed partial into its final location.
        File.Move(partPath, fullDestination, overwrite: true);
    }

    /// <summary>True for low-level exceptions worth retrying (network/IO/timeout).</summary>
    private static bool IsTransient(Exception ex) => ex switch
    {
        HttpRequestException => true,
        IOException => true,
        // HttpClient surfaces request timeouts as a TaskCanceledException whose token is not signalled.
        TaskCanceledException => true,
        OperationCanceledException => true,
        _ => false,
    };
}
