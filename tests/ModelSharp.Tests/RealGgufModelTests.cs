using System;
using System.IO;
using System.Linq;
using ModelSharp.Tensors;
using ModelSharp.Weights;
using Xunit;
using Xunit.Abstractions;

namespace ModelSharp.Tests;

/// <summary>
/// Opt-in, asset-gated end-to-end test against a real <c>.gguf</c> model. Like
/// <see cref="MiniLmTests"/>, it no-ops (logs and returns) unless a model asset is present, so the
/// default suite stays green without large gitignored binaries.
/// <para>
/// A <c>.gguf</c> is discovered via the <c>MODELSHARP_MODELS_DIR</c> environment variable (if set
/// and existing), otherwise a repo-relative <c>models/</c> directory, otherwise the test's
/// <c>assets/models/</c> output directory (mirroring how <see cref="MiniLmTests"/> gates). When a
/// file is found, every quantized tensor whose ggml type ModelSharp can dequantize is materialized
/// through <see cref="GgufFile.GetTensor"/> and checked for the right element count and all-finite
/// values; types that are not yet supported are skipped (not failed), so adding support later only
/// widens coverage.
/// </para>
/// </summary>
public class RealGgufModelTests
{
    private readonly ITestOutputHelper _out;
    public RealGgufModelTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Resolves a candidate <c>.gguf</c> file, or <c>null</c> if no asset is present. Search order:
    /// <c>MODELSHARP_MODELS_DIR</c>, a repo-relative <c>models/</c> dir, then <c>assets/models/</c>.
    /// </summary>
    private static string? FindGgufModel()
    {
        foreach (string dir in CandidateDirs())
        {
            if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) continue;
            string? hit = Directory
                .EnumerateFiles(dir, "*.gguf", SearchOption.AllDirectories)
                .OrderBy(p => p, StringComparer.Ordinal)
                .FirstOrDefault();
            if (hit is not null) return hit;
        }
        return null;
    }

    private static System.Collections.Generic.IEnumerable<string> CandidateDirs()
    {
        string? env = Environment.GetEnvironmentVariable("MODELSHARP_MODELS_DIR");
        if (!string.IsNullOrEmpty(env)) yield return env;

        // Walk up from the test's base directory looking for a repo-relative models/ folder.
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && d is not null; i++, d = d.Parent)
        {
            string candidate = Path.Combine(d.FullName, "models");
            if (Directory.Exists(candidate)) yield return candidate;
        }

        yield return Path.Combine(AppContext.BaseDirectory, "assets", "models");
    }

    [Fact]
    public void RealGguf_DequantizesEveryTensor_FiniteAndCorrectlySized()
    {
        string? path = FindGgufModel();
        if (path is null)
        {
            _out.WriteLine(
                "No .gguf asset present (set MODELSHARP_MODELS_DIR or drop a model in ./models); skipping.");
            return;
        }

        _out.WriteLine($"Opening GGUF: {path}");
        using var gguf = GgufFile.FromFile(path);
        _out.WriteLine($"version={gguf.Version}  tensors={gguf.Count}");

        int materialized = 0;
        int dequantized = 0;
        int skippedUnsupported = 0;
        var skippedTypes = new System.Collections.Generic.SortedSet<string>();

        foreach (string name in gguf.TensorNames)
        {
            GgufTensorInfo info = gguf.GetInfo(name);
            bool quantized = GgmlTypeInfo.IsQuantized(info.Type);

            if (quantized && !GgufDequant.IsSupported(info.Type))
            {
                skippedUnsupported++;
                skippedTypes.Add(info.Type.ToString());
                continue;
            }

            Tensor t = gguf.GetTensor(name); // must not throw
            materialized++;
            if (quantized) dequantized++;

            Assert.Equal(info.ElementCount, t.Length);

            // Validate finiteness for the float-materialized tensors (all quantized types and the
            // float scalar types decode to Tensor<float>); skip integer tensors.
            if (t.Dtype == ElementType.Float32)
            {
                ReadOnlySpan<float> span = t.AsFloat().Span;
                for (int i = 0; i < span.Length; i++)
                {
                    if (!float.IsFinite(span[i]))
                    {
                        Assert.Fail($"Tensor '{name}' ({info.Type}) has non-finite value {span[i]} at index {i}.");
                    }
                }
            }
        }

        _out.WriteLine(
            $"materialized={materialized}  dequantized={dequantized}  " +
            $"skippedUnsupportedQuant={skippedUnsupported}");
        if (skippedTypes.Count > 0)
            _out.WriteLine("skipped (unsupported quant) types: " + string.Join(", ", skippedTypes));

        // At least every tensor we attempted must have a sane element count; the model had > 0 tensors.
        Assert.True(gguf.Count > 0, "GGUF file contained no tensors.");
        Assert.True(materialized > 0, "No tensors could be materialized from the GGUF file.");
    }
}
