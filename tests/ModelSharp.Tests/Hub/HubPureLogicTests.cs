using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using ModelSharp.Hub;
using Xunit;

namespace ModelSharp.Tests.Hub;

/// <summary>
/// Pure-logic tests for the ModelSharp.Hub package: no network, no servers, no temp servers.
/// These exercise <see cref="ModelRefParser"/>, <see cref="OnnxBundleResolver"/>,
/// <see cref="ModelAliases"/>, <see cref="HubCredentials"/>, and <see cref="IntegrityVerifier"/>.
/// They are fully deterministic and always run.
/// </summary>
public class HubPureLogicTests
{
    // ------------------------------------------------------------------
    // ModelRefParser
    // ------------------------------------------------------------------

    [Fact]
    public void Parse_BareRepo_IsHfWithDefaultRevision()
    {
        var r = ModelRefParser.Parse("onnx-community/Qwen2.5-0.5B-Instruct");

        Assert.Equal("hf", r.Kind);
        Assert.Equal("onnx-community/Qwen2.5-0.5B-Instruct", r.Repo);
        Assert.Equal("main", r.Revision);
        Assert.Null(r.FileHint);
        Assert.Null(r.Url);
    }

    [Fact]
    public void Parse_BareRepo_HonoursDefaultRevisionArgument()
    {
        var r = ModelRefParser.Parse("owner/repo", defaultRevision: "v2");
        Assert.Equal("v2", r.Revision);
    }

    [Fact]
    public void Parse_HfScheme_WithRevisionAndSubPathFileHint()
    {
        var r = ModelRefParser.Parse("hf:owner/repo@main/onnx/model_q4.onnx");

        Assert.Equal("hf", r.Kind);
        Assert.Equal("owner/repo", r.Repo);
        Assert.Equal("main", r.Revision);
        Assert.Equal("onnx/model_q4.onnx", r.FileHint);
        Assert.Null(r.Url);
    }

    [Fact]
    public void Parse_RevisionSuffix_WithoutTrailingPath()
    {
        var r = ModelRefParser.Parse("owner/repo@v1.0");

        Assert.Equal("owner/repo", r.Repo);
        Assert.Equal("v1.0", r.Revision);
        Assert.Null(r.FileHint);
    }

    [Fact]
    public void Parse_GgufScheme_InfersKindAndFileHint()
    {
        var r = ModelRefParser.Parse("gguf:TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf");

        Assert.Equal("gguf", r.Kind);
        Assert.Equal("TheBloke/Llama-2-7B-GGUF", r.Repo);
        Assert.Equal("main", r.Revision);
        Assert.Equal("llama-2-7b.Q4_K_M.gguf", r.FileHint);
    }

    [Fact]
    public void Parse_SafetensorsScheme_SetsKind()
    {
        var r = ModelRefParser.Parse("safetensors:owner/repo/model.safetensors");

        Assert.Equal("safetensors", r.Kind);
        Assert.Equal("owner/repo", r.Repo);
        Assert.Equal("model.safetensors", r.FileHint);
    }

    [Fact]
    public void Parse_BareRepo_InfersGgufKindFromExtension()
    {
        // No scheme prefix, but the file hint ends in .gguf → kind inferred as gguf.
        var r = ModelRefParser.Parse("TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf");
        Assert.Equal("gguf", r.Kind);
        Assert.Equal("TheBloke/Llama-2-7B-GGUF", r.Repo);
        Assert.Equal("llama-2-7b.Q4_K_M.gguf", r.FileHint);
    }

    [Fact]
    public void Parse_BareRepo_InfersSafetensorsKindFromExtension()
    {
        var r = ModelRefParser.Parse("owner/repo/model.safetensors");
        Assert.Equal("safetensors", r.Kind);
        Assert.Equal("model.safetensors", r.FileHint);
    }

    [Fact]
    public void Parse_DirectHttpsUrl_IsUrlKind()
    {
        var r = ModelRefParser.Parse("https://example.com/model.onnx");

        Assert.Equal("url", r.Kind);
        Assert.Equal("https://example.com/model.onnx", r.Url);
        Assert.Null(r.Repo);
        Assert.Null(r.FileHint);
    }

    [Fact]
    public void Parse_HttpUrl_IsUrlKind()
    {
        var r = ModelRefParser.Parse("http://127.0.0.1:8080/some/model.onnx");
        Assert.Equal("url", r.Kind);
        Assert.Equal("http://127.0.0.1:8080/some/model.onnx", r.Url);
    }

    [Fact]
    public void Parse_UrlScheme_StripsPrefix()
    {
        var r = ModelRefParser.Parse("url:https://example.com/x.onnx");
        Assert.Equal("url", r.Kind);
        Assert.Equal("https://example.com/x.onnx", r.Url);
    }

    [Fact]
    public void Parse_RevisionWithSubPath_SplitsRevisionFromFileHint()
    {
        // "@<rev>/<sub/path>" — revision is the segment immediately after '@', the rest is the hint.
        var r = ModelRefParser.Parse("owner/repo@abc123/onnx/model.onnx");

        Assert.Equal("owner/repo", r.Repo);
        Assert.Equal("abc123", r.Revision);
        Assert.Equal("onnx/model.onnx", r.FileHint);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Parse_EmptySpec_Throws(string spec)
    {
        Assert.Throws<HubException>(() => ModelRefParser.Parse(spec));
    }

    [Fact]
    public void Parse_SingleSegment_NotARepo_Throws()
    {
        Assert.Throws<HubException>(() => ModelRefParser.Parse("justaname"));
    }

    [Fact]
    public void Parse_TrimsSurroundingWhitespace()
    {
        var r = ModelRefParser.Parse("  owner/repo  ");
        Assert.Equal("owner/repo", r.Repo);
    }

    // ------------------------------------------------------------------
    // OnnxBundleResolver: picker + companions
    // ------------------------------------------------------------------

    private static IReadOnlyList<RepoFile> SyntheticOnnxRepo() => new[]
    {
        new RepoFile("onnx/model.onnx", 1024),
        new RepoFile("onnx/model.onnx.data", 4096),   // external-data shard, same dir
        new RepoFile("tokenizer.json", 512),          // repo root
        new RepoFile("config.json", 64),              // repo root
        new RepoFile("README.md", 10),                // should NOT be a companion
    };

    [Fact]
    public void PickModelFile_PicksTheOnnx_FromSyntheticRepo()
    {
        var files = SyntheticOnnxRepo();
        var picked = OnnxBundleResolver.PickModelFile(files, hint: null);
        Assert.Equal("onnx/model.onnx", picked);
    }

    [Fact]
    public void CompanionFiles_IncludesExternalDataTokenizerAndConfig_ExcludesOnnxItselfAndUnrelated()
    {
        var files = SyntheticOnnxRepo();
        var companions = OnnxBundleResolver.CompanionFiles(files, "onnx/model.onnx");

        Assert.Contains("onnx/model.onnx.data", companions);
        Assert.Contains("tokenizer.json", companions);
        Assert.Contains("config.json", companions);

        // Never the .onnx itself, never unrelated files.
        Assert.DoesNotContain("onnx/model.onnx", companions);
        Assert.DoesNotContain("README.md", companions);
    }

    [Fact]
    public void CompanionFiles_ExternalDataComesBeforeNamedArtifacts()
    {
        var files = SyntheticOnnxRepo();
        var companions = OnnxBundleResolver.CompanionFiles(files, "onnx/model.onnx").ToList();

        int dataIdx = companions.IndexOf("onnx/model.onnx.data");
        int tokIdx = companions.IndexOf("tokenizer.json");
        Assert.True(dataIdx >= 0 && tokIdx >= 0);
        Assert.True(dataIdx < tokIdx, "external-data shard should precede tokenizer/config");
    }

    [Fact]
    public void CompanionFiles_IsDeduplicated()
    {
        var files = SyntheticOnnxRepo();
        var companions = OnnxBundleResolver.CompanionFiles(files, "onnx/model.onnx");
        Assert.Equal(companions.Count, companions.Distinct(StringComparer.OrdinalIgnoreCase).Count());
    }

    [Fact]
    public void PickModelFile_HonoursExplicitSubPathHint()
    {
        var files = new[]
        {
            new RepoFile("onnx/model.onnx", 1),
            new RepoFile("onnx/model_q4.onnx", 1),
            new RepoFile("onnx/model_int8.onnx", 1),
        };
        var picked = OnnxBundleResolver.PickModelFile(files, hint: "model_q4.onnx");
        Assert.Equal("onnx/model_q4.onnx", picked);
    }

    [Fact]
    public void PickModelFile_PrefersModelOnnxInOnnxDir_WhenMultiplePresent()
    {
        var files = new[]
        {
            new RepoFile("decoder_model.onnx", 1),
            new RepoFile("onnx/model.onnx", 1),
            new RepoFile("model_quantized.onnx", 1),
        };
        var picked = OnnxBundleResolver.PickModelFile(files, hint: null);
        Assert.Equal("onnx/model.onnx", picked);
    }

    [Fact]
    public void PickModelFile_ReturnsNull_WhenNoOnnx()
    {
        var files = new[] { new RepoFile("config.json", 1), new RepoFile("tokenizer.json", 1) };
        Assert.Null(OnnxBundleResolver.PickModelFile(files, hint: null));
    }

    [Fact]
    public void CompanionFiles_MatchesShardedAndUnderscoreExternalData()
    {
        var files = new[]
        {
            new RepoFile("model.onnx", 1),
            new RepoFile("model.onnx.data_0", 1),
            new RepoFile("model.onnx.data_1", 1),
            new RepoFile("model.onnx_data", 1),
        };
        var companions = OnnxBundleResolver.CompanionFiles(files, "model.onnx");
        Assert.Contains("model.onnx.data_0", companions);
        Assert.Contains("model.onnx.data_1", companions);
        Assert.Contains("model.onnx_data", companions);
    }

    // ------------------------------------------------------------------
    // ModelAliases
    // ------------------------------------------------------------------

    [Fact]
    public void ModelAliases_KnownAlias_ResolvesToFullSpec()
    {
        var spec = ModelAliases.Resolve("qwen2.5-0.5b-int4");
        Assert.Equal("onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx", spec);
    }

    [Fact]
    public void ModelAliases_IsCaseInsensitive()
    {
        Assert.Equal(
            ModelAliases.Resolve("qwen2.5-0.5b-int4"),
            ModelAliases.Resolve("Qwen2.5-0.5B-INT4"));
    }

    [Fact]
    public void ModelAliases_UnknownName_PassesThroughUnchanged()
    {
        const string spec = "some-unknown/repo";
        Assert.Equal(spec, ModelAliases.Resolve(spec));
    }

    [Fact]
    public void ModelAliases_ResolvedAlias_ParsesCleanly()
    {
        // An alias must round-trip through the parser into a valid ModelRef.
        var spec = ModelAliases.Resolve("qwen2.5-0.5b-int4");
        var r = ModelRefParser.Parse(spec);
        Assert.Equal("hf", r.Kind);
        Assert.Equal("onnx-community/Qwen2.5-0.5B-Instruct", r.Repo);
        Assert.Equal("onnx/model_q4.onnx", r.FileHint);
    }

    // ------------------------------------------------------------------
    // HubCredentials (pure, but reads env — assert precedence of explicit token)
    // ------------------------------------------------------------------

    [Fact]
    public void HubCredentials_ExplicitToken_WinsOverEverything()
    {
        var token = HubCredentials.Resolve(new HubOptions { Token = "explicit-tok" });
        Assert.Equal("explicit-tok", token);
    }

    [Fact]
    public void HubCredentials_ExplicitToken_IsTrimmed()
    {
        var token = HubCredentials.Resolve(new HubOptions { Token = "  tok-with-spaces  " });
        Assert.Equal("tok-with-spaces", token);
    }

    // ------------------------------------------------------------------
    // IntegrityVerifier
    // ------------------------------------------------------------------

    [Fact]
    public async Task IntegrityVerifier_Sha256_MatchesKnownHash()
    {
        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "blob.bin");
            var bytes = Encoding.ASCII.GetBytes("hello modelsharp hub");
            await File.WriteAllBytesAsync(path, bytes);

            var expected = Convert.ToHexStringLower(SHA256.HashData(bytes));
            var actual = await IntegrityVerifier.Sha256Async(path);

            Assert.Equal(expected, actual);
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public async Task IntegrityVerifier_Verify_BySha256_Succeeds_And_Fails()
    {
        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "blob.bin");
            var bytes = Encoding.ASCII.GetBytes("integrity payload");
            await File.WriteAllBytesAsync(path, bytes);
            var sha = Convert.ToHexStringLower(SHA256.HashData(bytes));

            Assert.True(await IntegrityVerifier.VerifyAsync(path, new RepoFile(path, bytes.Length, sha)));

            // Wrong sha → fails.
            var wrong = new string('0', 64);
            Assert.False(await IntegrityVerifier.VerifyAsync(path, new RepoFile(path, bytes.Length, wrong)));
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public async Task IntegrityVerifier_Verify_BySize_Succeeds_And_Fails()
    {
        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "blob.bin");
            var bytes = new byte[123];
            await File.WriteAllBytesAsync(path, bytes);

            Assert.True(await IntegrityVerifier.VerifyAsync(path, new RepoFile(path, 123)));
            Assert.False(await IntegrityVerifier.VerifyAsync(path, new RepoFile(path, 999)));
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public async Task IntegrityVerifier_MissingFile_ReturnsFalse()
    {
        var missing = Path.Combine(Path.GetTempPath(), "modelsharp-missing-" + Guid.NewGuid().ToString("N"));
        Assert.False(await IntegrityVerifier.VerifyAsync(missing, new RepoFile(missing, 10)));
    }

    // ------------------------------------------------------------------
    // ModelCache path layout (pure path computation)
    // ------------------------------------------------------------------

    [Fact]
    public void ModelCache_PathFor_UsesSnapshotLayout()
    {
        var dir = NewTempDir();
        try
        {
            var cache = new ModelCache(dir);
            var p = cache.PathFor("owner/repo", "main", "onnx/model.onnx");

            // models--owner--repo/snapshots/main/onnx/model.onnx
            Assert.Contains("models--owner--repo", p);
            Assert.Contains(Path.Combine("snapshots", "main"), p);
            Assert.EndsWith(Path.Combine("onnx", "model.onnx"), p);
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void ModelCache_IsCached_ReflectsExistenceAndSize()
    {
        var dir = NewTempDir();
        try
        {
            var cache = new ModelCache(dir);
            var p = Path.Combine(dir, "f.bin");
            Assert.False(cache.IsCached(p));

            File.WriteAllBytes(p, new byte[10]);
            Assert.True(cache.IsCached(p));
            Assert.True(cache.IsCached(p, new RepoFile(p, 10)));
            Assert.False(cache.IsCached(p, new RepoFile(p, 11)));   // size mismatch
        }
        finally { Cleanup(dir); }
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    internal static string NewTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "modelsharp-hub-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    internal static void Cleanup(string dir)
    {
        try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
        catch { /* best-effort cleanup */ }
    }
}
