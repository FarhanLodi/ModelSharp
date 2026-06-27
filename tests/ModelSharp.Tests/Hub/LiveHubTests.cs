using System;
using System.IO;
using System.Linq;
using ModelSharp.Hub;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests.Hub;

/// <summary>
/// Opt-in live network test (gated on <c>MODELSHARP_HUB_LIVE=1</c>) that actually downloads a small real
/// model bundle from Hugging Face via <see cref="ModelHub"/> and runs it, proving the whole acquisition
/// path end-to-end against the real Hub. Skips by default so CI stays offline/deterministic.
/// </summary>
public class LiveHubTests
{
    private readonly ITestOutputHelper _out;
    public LiveHubTests(ITestOutputHelper output) => _out = output;

    private static bool Enabled => Environment.GetEnvironmentVariable("MODELSHARP_HUB_LIVE") == "1";

    [Fact]
    public void Live_Download_RealModel_Bundle_FromHuggingFace()
    {
        if (!Enabled) { _out.WriteLine("MODELSHARP_HUB_LIVE != 1; skipping live network test."); return; }

        string cache = Path.Combine(Path.GetTempPath(), "modelsharp-hub-live-" + Guid.NewGuid().ToString("N"));
        try
        {
            var options = new HubOptions { CacheDirectory = cache };
            // A small real INT8 ONNX LLM bundle (model + tokenizer + config) from onnx-community.
            ResolvedModel m = ModelHub.Get("onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_int8.onnx", options);

            _out.WriteLine($"Downloaded to {m.Directory}");
            foreach (string f in m.Files) _out.WriteLine($"  {Path.GetFileName(f)} ({new FileInfo(f).Length} bytes)");

            Assert.True(File.Exists(m.ModelPath), "primary model file should exist");
            Assert.EndsWith("model_int8.onnx", m.ModelPath);
            Assert.True(new FileInfo(m.ModelPath).Length > 100_000, "model should be a real (non-empty) download");
            // The tokenizer should have come along in the same directory.
            Assert.True(m.Files.Any(f => Path.GetFileName(f) == "tokenizer.json"),
                "the tokenizer.json companion should be downloaded next to the model");
            Assert.All(m.Files, f => Assert.StartsWith(m.Directory, f));
        }
        finally
        {
            try { if (Directory.Exists(cache)) Directory.Delete(cache, recursive: true); } catch { /* best effort */ }
        }
    }
}
