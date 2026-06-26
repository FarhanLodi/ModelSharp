using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using ModelSharp.Graph;

namespace ModelSharp.Manifest;

/// <summary>
/// Resolves a <see cref="ModelManifest"/> for a model file. Sources are tried in
/// precedence order — first hit wins:
/// <list type="number">
///   <item>Sidecar JSON next to the model (<c>&lt;model&gt;.onnx.manifest.json</c> or <c>&lt;model&gt;.manifest.json</c>).</item>
///   <item>ONNX embedded metadata (<see cref="ModelGraph.MetadataProps"/>).</item>
///   <item>A tiny built-in registry keyed by a file-name heuristic.</item>
/// </list>
/// Any source falls back to <see cref="ModelTask.Unknown"/> with neutral defaults when
/// nothing matches. Relative <c>vocab</c> paths are resolved against the model directory.
/// </summary>
public static class ManifestResolver
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    /// <summary>Resolves the manifest for <paramref name="modelPath"/> using the loaded graph's metadata.</summary>
    public static ModelManifest Resolve(string modelPath, ModelGraph graph)
    {
        if (modelPath is null) throw new ArgumentNullException(nameof(modelPath));

        string fullPath = Path.GetFullPath(modelPath);
        string modelDir = Path.GetDirectoryName(fullPath) ?? Directory.GetCurrentDirectory();

        // (a) Sidecar JSON.
        foreach (string candidate in SidecarCandidates(fullPath))
        {
            if (File.Exists(candidate))
            {
                ModelManifest? sidecar = FromSidecarJson(candidate, modelDir);
                if (sidecar is not null) return sidecar;
            }
        }

        // (b) Embedded ONNX metadata.
        if (graph is not null)
        {
            ModelManifest? meta = FromMetadata(graph.MetadataProps, modelDir);
            if (meta is not null) return meta;
        }

        // (c) Built-in registry.
        return FromBuiltIn(fullPath, modelDir);
    }

    /// <summary>The sidecar file paths to probe, in order.</summary>
    public static IEnumerable<string> SidecarCandidates(string modelPath)
    {
        string full = Path.GetFullPath(modelPath);
        yield return full + ".manifest.json";                 // model.onnx.manifest.json
        string swapped = Path.ChangeExtension(full, ".manifest.json"); // model.manifest.json
        if (!string.Equals(swapped, full + ".manifest.json", StringComparison.OrdinalIgnoreCase))
            yield return swapped;
    }

    // ---- (a) sidecar JSON ----

    /// <summary>The on-disk JSON shape for a sidecar manifest. All fields optional.</summary>
    private sealed class ManifestDto
    {
        public string? Task { get; set; }
        public string? Layout { get; set; }
        public int? Width { get; set; }
        public int? Height { get; set; }
        public float[]? Mean { get; set; }
        public float[]? Std { get; set; }
        public string? Color { get; set; }
        public string[]? Labels { get; set; }
        public Dictionary<string, string>? Extra { get; set; }
    }

    private static ModelManifest? FromSidecarJson(string path, string modelDir)
    {
        string json;
        try { json = File.ReadAllText(path); }
        catch (IOException) { return null; }

        ManifestDto? dto;
        try { dto = JsonSerializer.Deserialize<ManifestDto>(json, JsonOpts); }
        catch (JsonException ex)
        {
            throw new ModelSharpException($"Failed to parse manifest sidecar '{path}': {ex.Message}", ex);
        }
        if (dto is null) return null;

        return new ModelManifest
        {
            Task = ParseEnum(dto.Task, ModelTask.Unknown),
            Layout = ParseEnum(dto.Layout, TensorLayout.Nchw),
            Width = dto.Width ?? 0,
            Height = dto.Height ?? 0,
            Mean = dto.Mean ?? new[] { 0f, 0f, 0f },
            Std = dto.Std ?? new[] { 1f, 1f, 1f },
            Color = ParseEnum(dto.Color, ColorOrder.Rgb),
            Labels = dto.Labels,
            Extra = NormalizeExtra(dto.Extra, modelDir),
        };
    }

    // ---- (b) embedded metadata ----

    private static ModelManifest? FromMetadata(IReadOnlyDictionary<string, string> md, string modelDir)
    {
        if (md is null || md.Count == 0) return null;

        // Case-insensitive snapshot of the metadata props.
        var ci = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (KeyValuePair<string, string> kv in md) ci[kv.Key] = kv.Value;

        bool any = false;
        ModelTask task = ModelTask.Unknown;
        TensorLayout layout = TensorLayout.Nchw;
        ColorOrder color = ColorOrder.Rgb;
        int width = 0, height = 0;

        // A 'task' key counts as a hit only if it parses to a known task — so foreign ONNX
        // metadata (e.g. task="feature-extraction") falls through to the built-in heuristic.
        if (TryGet(ci, out string? ts, "task", "modelsharp.task"))
        {
            ModelTask parsed = ParseEnum(ts, ModelTask.Unknown);
            if (parsed != ModelTask.Unknown) { task = parsed; any = true; }
        }
        if (TryGet(ci, out string? ls, "layout", "modelsharp.layout")) { layout = ParseEnum(ls, TensorLayout.Nchw); any = true; }
        if (TryGet(ci, out string? cs, "color", "modelsharp.color")) { color = ParseEnum(cs, ColorOrder.Rgb); any = true; }
        if (TryGet(ci, out string? ws, "width", "modelsharp.width") && int.TryParse(ws, out int wv)) { width = wv; any = true; }
        if (TryGet(ci, out string? hs, "height", "modelsharp.height") && int.TryParse(hs, out int hv)) { height = hv; any = true; }

        float[]? mean = ParseFloats(GetOrNull(ci, "mean", "modelsharp.mean"));
        if (mean is not null) any = true;
        float[]? std = ParseFloats(GetOrNull(ci, "std", "modelsharp.std"));
        if (std is not null) any = true;
        string[]? labels = ParseLabels(GetOrNull(ci, "labels", "modelsharp.labels"));
        if (labels is not null) any = true;

        var extra = new Dictionary<string, string>();
        if (TryGet(ci, out string? vocab, "vocab", "modelsharp.vocab") && !string.IsNullOrWhiteSpace(vocab))
        {
            extra["vocab"] = vocab!;
            any = true;
        }
        if (TryGet(ci, out string? lower, "lowercase", "modelsharp.lowercase"))
        {
            extra["lowercase"] = lower!;
            any = true;
        }

        if (!any) return null;

        return new ModelManifest
        {
            Task = task,
            Layout = layout,
            Color = color,
            Width = width,
            Height = height,
            Mean = mean ?? new[] { 0f, 0f, 0f },
            Std = std ?? new[] { 1f, 1f, 1f },
            Labels = labels,
            Extra = NormalizeExtra(extra, modelDir),
        };
    }

    // ---- (c) built-in registry ----

    private static ModelManifest FromBuiltIn(string fullPath, string modelDir)
    {
        string name = Path.GetFileNameWithoutExtension(fullPath).ToLowerInvariant();

        // Sentence-embedding models: tokenizer + mean-pool live in core.
        if (Contains(name, "minilm", "sentence", "sbert", "mpnet", "all-mini", "paraphrase", "msmarco"))
        {
            return new ModelManifest
            {
                Task = ModelTask.Embedding,
                Extra = NormalizeExtra(new Dictionary<string, string> { ["vocab"] = "vocab.txt" }, modelDir),
            };
        }

        // Common ImageNet classifiers (the vision adapter supplies the processors).
        if (Contains(name, "resnet", "mobilenet", "squeezenet", "efficientnet", "densenet",
                "inception", "googlenet", "shufflenet", "vgg", "alexnet"))
        {
            return new ModelManifest
            {
                Task = ModelTask.ImageClassification,
                Layout = TensorLayout.Nchw,
                Color = ColorOrder.Rgb,
                Width = 224,
                Height = 224,
                Mean = new[] { 0.485f, 0.456f, 0.406f },
                Std = new[] { 0.229f, 0.224f, 0.225f },
            };
        }

        return new ModelManifest { Task = ModelTask.Unknown };
    }

    // ---- helpers ----

    private static Dictionary<string, string> NormalizeExtra(IDictionary<string, string>? extra, string modelDir)
    {
        var d = new Dictionary<string, string>();
        if (extra is not null)
            foreach (KeyValuePair<string, string> kv in extra) d[kv.Key] = kv.Value;

        if (d.TryGetValue("vocab", out string? vocab) && !string.IsNullOrWhiteSpace(vocab) && !Path.IsPathRooted(vocab))
            d["vocab"] = Path.GetFullPath(Path.Combine(modelDir, vocab));

        return d;
    }

    private static bool Contains(string haystack, params string[] needles)
    {
        foreach (string n in needles)
            if (haystack.Contains(n, StringComparison.Ordinal)) return true;
        return false;
    }

    private static bool TryGet(IReadOnlyDictionary<string, string> map, out string? value, params string[] keys)
    {
        foreach (string k in keys)
            if (map.TryGetValue(k, out value)) return true;
        value = null;
        return false;
    }

    private static string? GetOrNull(IReadOnlyDictionary<string, string> map, params string[] keys)
        => TryGet(map, out string? v, keys) ? v : null;

    private static T ParseEnum<T>(string? s, T fallback) where T : struct, Enum
        => !string.IsNullOrWhiteSpace(s) && Enum.TryParse(s, ignoreCase: true, out T v) ? v : fallback;

    private static float[]? ParseFloats(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return null;
        string trimmed = s.Trim().Trim('[', ']');
        string[] parts = trimmed.Split(new[] { ',', ' ', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var list = new List<float>(parts.Length);
        foreach (string p in parts)
            if (float.TryParse(p, NumberStyles.Float, CultureInfo.InvariantCulture, out float f)) list.Add(f);
        return list.Count > 0 ? list.ToArray() : null;
    }

    private static string[]? ParseLabels(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return null;
        string trimmed = s.Trim();
        if (trimmed.StartsWith("[", StringComparison.Ordinal))
        {
            try
            {
                string[]? arr = JsonSerializer.Deserialize<string[]>(trimmed, JsonOpts);
                if (arr is { Length: > 0 }) return arr;
            }
            catch (JsonException) { /* fall through to delimiter parsing */ }
        }
        string[] parts = trimmed.Split(new[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length > 0 ? parts : null;
    }
}
