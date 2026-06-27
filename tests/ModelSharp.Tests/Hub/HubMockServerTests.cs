using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using ModelSharp.Hub;
using Xunit;

namespace ModelSharp.Tests.Hub;

/// <summary>
/// Integration tests for the ModelSharp.Hub package against a local in-process fake Hugging Face
/// server (<see cref="HttpListener"/>) — no real network. The server serves:
/// <list type="bullet">
///   <item><c>GET /api/models/{repo}/tree/{rev}?recursive=true</c> → a JSON file listing.</item>
///   <item><c>GET /{repo}/resolve/{rev}/{path}</c> → the bytes of each fake file (Range-aware).</item>
/// </list>
/// </summary>
public sealed class HubMockServerTests
{
    [Fact]
    public async Task HuggingFaceSource_Resolve_DownloadsBundleIntoOneDirectory()
    {
        using var server = new FakeHubServer();
        server.AddFile("onnx/model.onnx", RandomBytes(2048));
        server.AddFile("onnx/model.onnx.data", RandomBytes(8192)); // external-data shard
        server.AddFile("tokenizer.json", Encoding.UTF8.GetBytes("{\"tok\":true}"));
        server.AddFile("config.json", Encoding.UTF8.GetBytes("{\"cfg\":1}"));
        server.Start();

        var cacheDir = HubPureLogicTests.NewTempDir();
        try
        {
            using var http = new HttpClient();
            var client = new HuggingFaceClient(http, endpoint: server.BaseUrl);
            var cache = new ModelCache(cacheDir);
            var downloader = new HttpDownloader();
            var source = new HuggingFaceSource(client, downloader, cache);

            var reference = ModelRefParser.Parse("fake/repo");
            var options = new HubOptions
            {
                CacheDirectory = cacheDir,
                Endpoint = server.BaseUrl,
                // The fake server reports sizes but no sha256; size verification is sufficient & deterministic.
                VerifyIntegrity = true,
            };

            var resolved = await source.ResolveAsync(reference, options);

            // Primary model file is model.onnx.
            Assert.EndsWith("model.onnx", resolved.ModelPath);
            Assert.True(File.Exists(resolved.ModelPath), "model.onnx should exist locally");

            var bundleDir = Path.GetDirectoryName(resolved.ModelPath)!;
            Assert.Equal(Path.GetFullPath(bundleDir), Path.GetFullPath(resolved.Directory));

            // External-data file landed in the SAME local directory as the .onnx.
            var localData = Path.Combine(bundleDir, "model.onnx.data");
            Assert.True(File.Exists(localData), "external-data must sit next to the .onnx");

            // Tokenizer + config came too, into the same directory.
            var localTokenizer = Path.Combine(bundleDir, "tokenizer.json");
            var localConfig = Path.Combine(bundleDir, "config.json");
            Assert.True(File.Exists(localTokenizer), "tokenizer.json must be downloaded");
            Assert.True(File.Exists(localConfig), "config.json must be downloaded");

            // Bytes match exactly what the server served.
            Assert.Equal(server.GetFile("onnx/model.onnx"), await File.ReadAllBytesAsync(resolved.ModelPath));
            Assert.Equal(server.GetFile("onnx/model.onnx.data"), await File.ReadAllBytesAsync(localData));
            Assert.Equal(server.GetFile("tokenizer.json"), await File.ReadAllBytesAsync(localTokenizer));
            Assert.Equal(server.GetFile("config.json"), await File.ReadAllBytesAsync(localConfig));

            // ResolvedModel.Files lists all four local paths.
            Assert.Contains(resolved.Files, f => f.EndsWith("model.onnx"));
            Assert.Contains(resolved.Files, f => f.EndsWith("model.onnx.data"));
            Assert.Contains(resolved.Files, f => f.EndsWith("tokenizer.json"));
            Assert.Contains(resolved.Files, f => f.EndsWith("config.json"));
        }
        finally
        {
            HubPureLogicTests.Cleanup(cacheDir);
        }
    }

    [Fact]
    public async Task HuggingFaceSource_SecondResolve_ReusesCache_NoRedownload()
    {
        using var server = new FakeHubServer();
        server.AddFile("onnx/model.onnx", RandomBytes(1024));
        server.AddFile("tokenizer.json", Encoding.UTF8.GetBytes("{}"));
        server.AddFile("config.json", Encoding.UTF8.GetBytes("{}"));
        server.Start();

        var cacheDir = HubPureLogicTests.NewTempDir();
        try
        {
            using var http = new HttpClient();
            var client = new HuggingFaceClient(http, endpoint: server.BaseUrl);
            var cache = new ModelCache(cacheDir);
            var downloader = new HttpDownloader();
            var source = new HuggingFaceSource(client, downloader, cache);

            var reference = ModelRefParser.Parse("fake/repo");
            var options = new HubOptions { CacheDirectory = cacheDir, Endpoint = server.BaseUrl };

            var first = await source.ResolveAsync(reference, options);
            long resolveRequestsAfterFirst = server.ResolveRequestCount;
            Assert.True(resolveRequestsAfterFirst >= 3, "first resolve should download every file");

            // Second resolve: files are cached and valid, so no resolve-URL downloads should occur.
            var second = await source.ResolveAsync(reference, options);
            long resolveRequestsAfterSecond = server.ResolveRequestCount;

            Assert.Equal(resolveRequestsAfterFirst, resolveRequestsAfterSecond);
            Assert.Equal(first.ModelPath, second.ModelPath);
        }
        finally
        {
            HubPureLogicTests.Cleanup(cacheDir);
        }
    }

    [Fact]
    public async Task HuggingFaceClient_ListFiles_ParsesTreeJson()
    {
        using var server = new FakeHubServer();
        server.AddFile("onnx/model.onnx", RandomBytes(100));
        server.AddFile("onnx/model.onnx.data", RandomBytes(200));
        server.AddFile("config.json", RandomBytes(10));
        server.Start();

        using var http = new HttpClient();
        var client = new HuggingFaceClient(http, endpoint: server.BaseUrl);

        var files = await client.ListFilesAsync("fake/repo", "main");

        Assert.Equal(3, files.Count);
        var onnx = files.Single(f => f.Path == "onnx/model.onnx");
        Assert.Equal(100, onnx.Size);
    }

    [Fact]
    public async Task HttpDownloader_Resume_AppendsViaRangeRequest()
    {
        using var server = new FakeHubServer();
        var content = RandomBytes(64 * 1024);
        server.AddFile("big.bin", content);
        server.Start();

        var dir = HubPureLogicTests.NewTempDir();
        try
        {
            var dest = Path.Combine(dir, "big.bin");

            // Simulate a previous partial download: write the first half to the .part file.
            int half = content.Length / 2;
            await File.WriteAllBytesAsync(dest + ".part", content.Take(half).ToArray());

            var downloader = new HttpDownloader();
            var url = new HuggingFaceClient(endpoint: server.BaseUrl).ResolveUrl("fake/repo", "main", "big.bin");

            await downloader.DownloadAsync(url, dest);

            // The completed file must match the full content, and the server must have seen a Range request.
            Assert.True(File.Exists(dest));
            Assert.Equal(content, await File.ReadAllBytesAsync(dest));
            Assert.True(server.SawRangeRequest, "downloader should resume via an HTTP Range request");
            Assert.False(File.Exists(dest + ".part"), ".part file should be promoted away on completion");
        }
        finally
        {
            HubPureLogicTests.Cleanup(dir);
        }
    }

    [Fact]
    public async Task HttpDownloader_FreshDownload_WritesFullFile()
    {
        using var server = new FakeHubServer();
        var content = RandomBytes(4096);
        server.AddFile("plain.bin", content);
        server.Start();

        var dir = HubPureLogicTests.NewTempDir();
        try
        {
            var dest = Path.Combine(dir, "plain.bin");
            var downloader = new HttpDownloader();
            var url = new HuggingFaceClient(endpoint: server.BaseUrl).ResolveUrl("fake/repo", "main", "plain.bin");

            await downloader.DownloadAsync(url, dest);

            Assert.Equal(content, await File.ReadAllBytesAsync(dest));
        }
        finally
        {
            HubPureLogicTests.Cleanup(dir);
        }
    }

    private static byte[] RandomBytes(int n)
    {
        var b = new byte[n];
        RandomNumberGenerator.Fill(b);
        return b;
    }

    // ==================================================================
    // Fake Hugging Face HTTP server (HttpListener)
    // ==================================================================

    /// <summary>
    /// A minimal in-process stand-in for the Hugging Face Hub. Picks a free loopback port, serves a tree
    /// listing and Range-aware file bytes, and tracks request counts so tests can assert cache behaviour.
    /// </summary>
    private sealed class FakeHubServer : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly Dictionary<string, byte[]> _files = new(StringComparer.Ordinal);
        private readonly CancellationTokenSource _cts = new();
        private Task? _loop;

        private long _resolveRequests;
        private volatile bool _sawRange;

        public string BaseUrl { get; }

        public long ResolveRequestCount => Interlocked.Read(ref _resolveRequests);
        public bool SawRangeRequest => _sawRange;

        public FakeHubServer()
        {
            int port = GetFreePort();
            BaseUrl = $"http://127.0.0.1:{port}";
            _listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        }

        public void AddFile(string repoRelativePath, byte[] bytes) => _files[repoRelativePath] = bytes;

        public byte[] GetFile(string repoRelativePath) => _files[repoRelativePath];

        public void Start()
        {
            _listener.Start();
            _loop = Task.Run(AcceptLoopAsync);
        }

        private static int GetFreePort()
        {
            var l = new TcpListener(IPAddress.Loopback, 0);
            l.Start();
            try { return ((IPEndPoint)l.LocalEndpoint).Port; }
            finally { l.Stop(); }
        }

        private async Task AcceptLoopAsync()
        {
            while (!_cts.IsCancellationRequested)
            {
                HttpListenerContext ctx;
                try
                {
                    ctx = await _listener.GetContextAsync();
                }
                catch (Exception) when (_cts.IsCancellationRequested)
                {
                    return;
                }
                catch (HttpListenerException)
                {
                    return;
                }
                catch (ObjectDisposedException)
                {
                    return;
                }

                // Handle each request on the thread pool so concurrent bundle downloads work.
                _ = Task.Run(() => HandleSafe(ctx));
            }
        }

        private void HandleSafe(HttpListenerContext ctx)
        {
            try { Handle(ctx); }
            catch
            {
                try { ctx.Response.StatusCode = 500; ctx.Response.Close(); }
                catch { /* ignore */ }
            }
        }

        private void Handle(HttpListenerContext ctx)
        {
            string path = Uri.UnescapeDataString(ctx.Request.Url!.AbsolutePath);

            // Tree listing: /api/models/{owner}/{repo}/tree/{rev}
            int treeIdx = path.IndexOf("/tree/", StringComparison.Ordinal);
            if (path.StartsWith("/api/models/", StringComparison.Ordinal) && treeIdx >= 0)
            {
                WriteTreeJson(ctx);
                return;
            }

            // Resolve URL: /{owner}/{repo}/resolve/{rev}/{repoRelativePath}
            int resolveIdx = path.IndexOf("/resolve/", StringComparison.Ordinal);
            if (resolveIdx >= 0)
            {
                string afterResolve = path.Substring(resolveIdx + "/resolve/".Length);
                // Drop the leading "{rev}/" segment.
                int firstSlash = afterResolve.IndexOf('/');
                string repoRelative = firstSlash >= 0 ? afterResolve.Substring(firstSlash + 1) : afterResolve;
                WriteFile(ctx, repoRelative);
                return;
            }

            ctx.Response.StatusCode = 404;
            ctx.Response.Close();
        }

        private void WriteTreeJson(HttpListenerContext ctx)
        {
            var sb = new StringBuilder();
            sb.Append('[');
            bool first = true;
            foreach (var kv in _files)
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append("{\"type\":\"file\",\"path\":\"")
                  .Append(JsonEscape(kv.Key))
                  .Append("\",\"size\":")
                  .Append(kv.Value.Length)
                  .Append('}');
            }
            sb.Append(']');

            byte[] body = Encoding.UTF8.GetBytes(sb.ToString());
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            ctx.Response.ContentLength64 = body.Length;
            ctx.Response.OutputStream.Write(body, 0, body.Length);
            ctx.Response.Close();
        }

        private void WriteFile(HttpListenerContext ctx, string repoRelative)
        {
            if (!_files.TryGetValue(repoRelative, out var bytes))
            {
                ctx.Response.StatusCode = 404;
                ctx.Response.Close();
                return;
            }

            Interlocked.Increment(ref _resolveRequests);

            // Honour a single Range request: "bytes=<start>-".
            string? rangeHeader = ctx.Request.Headers["Range"];
            int start = 0;
            bool partial = false;
            if (!string.IsNullOrEmpty(rangeHeader) &&
                rangeHeader.StartsWith("bytes=", StringComparison.OrdinalIgnoreCase))
            {
                _sawRange = true;
                string spec = rangeHeader.Substring("bytes=".Length);
                int dash = spec.IndexOf('-');
                string startStr = dash >= 0 ? spec.Substring(0, dash) : spec;
                if (int.TryParse(startStr, out int parsed) && parsed >= 0 && parsed < bytes.Length)
                {
                    start = parsed;
                    partial = true;
                }
            }

            int length = bytes.Length - start;
            if (partial)
            {
                ctx.Response.StatusCode = 206; // Partial Content
                ctx.Response.Headers["Content-Range"] = $"bytes {start}-{bytes.Length - 1}/{bytes.Length}";
            }
            else
            {
                ctx.Response.StatusCode = 200;
            }
            ctx.Response.Headers["Accept-Ranges"] = "bytes";
            ctx.Response.ContentType = "application/octet-stream";
            ctx.Response.ContentLength64 = length;
            ctx.Response.OutputStream.Write(bytes, start, length);
            ctx.Response.Close();
        }

        private static string JsonEscape(string s) =>
            s.Replace("\\", "\\\\").Replace("\"", "\\\"");

        public void Dispose()
        {
            _cts.Cancel();
            try { _listener.Stop(); } catch { /* ignore */ }
            try { _listener.Close(); } catch { /* ignore */ }
            try { _loop?.Wait(TimeSpan.FromSeconds(2)); } catch { /* ignore */ }
            _cts.Dispose();
        }
    }
}
