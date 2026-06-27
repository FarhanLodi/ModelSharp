using System;
using System.Collections.Generic;

namespace ModelSharp.Hub;

/// <summary>
/// Pure (IO-free) logic that, given a repo file <em>listing</em>, decides which <c>.onnx</c> file is the
/// primary model and which other files (external-data weight shards, tokenizer, config) must be downloaded
/// alongside it so ModelSharp has a complete, runnable bundle in one directory.
/// </summary>
/// <remarks>
/// Real ONNX LLM repos keep the model's weights out-of-band as "external data" when the graph exceeds the
/// 2 GB protobuf limit. The companion blobs live in the <em>same directory</em> as the <c>.onnx</c> and are
/// named with one of several conventions (no single standard across exporters):
/// <list type="bullet">
///   <item><c>&lt;model&gt;.onnx.data</c> (ORT / Optimum default)</item>
///   <item><c>&lt;model&gt;.onnx_data</c> (older Optimum / some HF exports)</item>
///   <item><c>&lt;model&gt;.onnx.data_0</c>, <c>&lt;model&gt;.onnx.data_1</c>, … (sharded external data)</item>
///   <item>plain <c>model.onnx_data</c> sitting next to <c>model.onnx</c></item>
/// </list>
/// We match all of these and, to be safe against naming drift, include every external-data blob found in the
/// same directory as the chosen model.
/// </remarks>
public static class OnnxBundleResolver
{
    private const string OnnxExtension = ".onnx";

    /// <summary>Tokenizer artefacts ModelSharp may need, in stable priority order.</summary>
    private static readonly string[] TokenizerNames =
    {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "spiece.model",
        "sentencepiece.bpe.model",
    };

    /// <summary>Config artefacts ModelSharp may need, in stable priority order.</summary>
    private static readonly string[] ConfigNames =
    {
        "config.json",
        "genai_config.json",
        "generation_config.json",
        "preprocessor_config.json",
    };

    /// <summary>
    /// Chooses the primary <c>.onnx</c> file from a repo listing.
    /// </summary>
    /// <param name="files">The repo file listing.</param>
    /// <param name="hint">An optional user-supplied path or sub-path naming a specific <c>.onnx</c>.</param>
    /// <returns>
    /// The repo-relative path of the chosen <c>.onnx</c>, or <c>null</c> when the listing contains no
    /// <c>.onnx</c> file.
    /// </returns>
    /// <remarks>
    /// Selection rules, in order:
    /// <list type="number">
    ///   <item>If <paramref name="hint"/> is given and matches a listed path exactly or as a suffix, return it.</item>
    ///   <item>Among files ending in <c>.onnx</c>, prefer one literally named <c>model.onnx</c>, preferring a copy
    ///         inside an <c>onnx/</c> directory.</item>
    ///   <item>If exactly one <c>.onnx</c> exists, return it.</item>
    ///   <item>Otherwise pick the most canonical name (<c>model.onnx</c> &gt; <c>model_*.onnx</c> &gt; shorter name &gt;
    ///         ordinal), deterministically.</item>
    /// </list>
    /// </remarks>
    public static string? PickModelFile(IReadOnlyList<RepoFile> files, string? hint)
    {
        if (files is null) throw new ArgumentNullException(nameof(files));

        // 1) Honour an explicit hint: exact path or suffix match.
        if (!string.IsNullOrWhiteSpace(hint))
        {
            string h = Normalize(hint);

            // Exact path wins.
            for (int i = 0; i < files.Count; i++)
            {
                if (string.Equals(Normalize(files[i].Path), h, StringComparison.OrdinalIgnoreCase))
                    return files[i].Path;
            }

            // Suffix match (e.g. hint "model_q4.onnx" matches "onnx/model_q4.onnx").
            // Prefer the longest path so the most specific candidate wins deterministically.
            string? best = null;
            for (int i = 0; i < files.Count; i++)
            {
                string p = Normalize(files[i].Path);
                if (EndsWithSegment(p, h) &&
                    (best is null || files[i].Path.Length > best.Length ||
                     (files[i].Path.Length == best.Length &&
                      string.CompareOrdinal(files[i].Path, best) < 0)))
                {
                    best = files[i].Path;
                }
            }
            if (best is not null) return best;
            // Hint did not resolve — fall through to heuristic selection.
        }

        // Collect all .onnx candidates.
        var onnx = new List<string>();
        for (int i = 0; i < files.Count; i++)
        {
            if (IsOnnx(files[i].Path)) onnx.Add(files[i].Path);
        }

        if (onnx.Count == 0) return null;
        if (onnx.Count == 1) return onnx[0];

        // 2..4) Pick the most canonical candidate deterministically.
        string? choice = null;
        int choiceRank = int.MaxValue;
        foreach (string path in onnx)
        {
            int rank = CanonicalRank(path);
            if (choice is null ||
                rank < choiceRank ||
                (rank == choiceRank && IsBetterTieBreak(path, choice)))
            {
                choice = path;
                choiceRank = rank;
            }
        }

        return choice;
    }

    /// <summary>
    /// Returns the repo-relative paths (excluding the <c>.onnx</c> itself) that must be downloaded next to
    /// <paramref name="onnxRepoPath"/> to form a complete bundle: external-data weight shards, tokenizer
    /// files, and config files. The result is de-duplicated and deterministically ordered.
    /// </summary>
    /// <param name="files">The repo file listing.</param>
    /// <param name="onnxRepoPath">The repo-relative path of the chosen primary <c>.onnx</c>.</param>
    public static IReadOnlyList<string> CompanionFiles(IReadOnlyList<RepoFile> files, string onnxRepoPath)
    {
        if (files is null) throw new ArgumentNullException(nameof(files));
        if (onnxRepoPath is null) throw new ArgumentNullException(nameof(onnxRepoPath));

        string onnxPath = Normalize(onnxRepoPath);
        string onnxDir = DirectoryOf(onnxPath);
        string onnxName = FileName(onnxPath);

        var result = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void Add(string path)
        {
            if (string.Equals(Normalize(path), onnxPath, StringComparison.OrdinalIgnoreCase))
                return; // never include the .onnx itself
            if (seen.Add(Normalize(path)))
                result.Add(path);
        }

        // 1) External-data blobs in the SAME directory as the .onnx.
        //    Add ones whose basename matches this model first (deterministic, model-associated),
        //    then any remaining same-dir external-data blobs to be safe against naming drift.
        var sameDirExternal = new List<string>();
        for (int i = 0; i < files.Count; i++)
        {
            string p = Normalize(files[i].Path);
            if (!string.Equals(DirectoryOf(p), onnxDir, StringComparison.OrdinalIgnoreCase))
                continue;
            if (IsExternalDataName(FileName(p)))
                sameDirExternal.Add(files[i].Path);
        }

        // Stable, model-associated-first ordering.
        sameDirExternal.Sort((a, b) =>
        {
            bool am = IsExternalDataFor(FileName(Normalize(a)), onnxName);
            bool bm = IsExternalDataFor(FileName(Normalize(b)), onnxName);
            if (am != bm) return am ? -1 : 1;
            return string.CompareOrdinal(Normalize(a), Normalize(b));
        });
        foreach (string p in sameDirExternal) Add(p);

        // 2) Tokenizer files: prefer same dir, then repo root, then anywhere.
        AddNamedArtifacts(files, TokenizerNames, onnxDir, Add);

        // 3) Config files: prefer same dir, then repo root.
        AddNamedArtifacts(files, ConfigNames, onnxDir, Add);

        return result;
    }

    /// <summary>
    /// For each wanted file name, picks the best-located copy in the listing (same dir &gt; repo root &gt;
    /// shallowest path &gt; ordinal) and adds it. Names are processed in their declared priority order.
    /// </summary>
    private static void AddNamedArtifacts(
        IReadOnlyList<RepoFile> files, string[] wantedNames, string onnxDir, Action<string> add)
    {
        foreach (string name in wantedNames)
        {
            string? best = null;
            int bestRank = int.MaxValue;
            for (int i = 0; i < files.Count; i++)
            {
                string p = Normalize(files[i].Path);
                if (!string.Equals(FileName(p), name, StringComparison.OrdinalIgnoreCase))
                    continue;

                string dir = DirectoryOf(p);
                int rank;
                if (string.Equals(dir, onnxDir, StringComparison.OrdinalIgnoreCase))
                    rank = 0;                       // same dir as the model
                else if (dir.Length == 0)
                    rank = 1;                       // repo root
                else
                    rank = 2 + DepthOf(p);          // elsewhere, shallower preferred

                if (best is null ||
                    rank < bestRank ||
                    (rank == bestRank && string.CompareOrdinal(p, Normalize(best)) < 0))
                {
                    best = files[i].Path;
                    bestRank = rank;
                }
            }
            if (best is not null) add(best);
        }
    }

    // --- naming helpers --------------------------------------------------

    private static bool IsOnnx(string path) =>
        path.EndsWith(OnnxExtension, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// True if <paramref name="fileName"/> looks like ONNX external-data for any model. Recognises:
    /// <c>*.onnx.data</c>, <c>*.onnx_data</c>, and sharded <c>*.onnx.data_N</c> / <c>*.onnx_data_N</c>.
    /// </summary>
    private static bool IsExternalDataName(string fileName)
    {
        string n = fileName;

        // <stem>.onnx.data  or  <stem>.onnx.data_<n>
        int idx = IndexOfOrdinalIgnoreCase(n, ".onnx.data");
        if (idx >= 0 && IsDataTail(n, idx + ".onnx.data".Length))
            return true;

        // <stem>.onnx_data  or  <stem>.onnx_data_<n>
        idx = IndexOfOrdinalIgnoreCase(n, ".onnx_data");
        if (idx >= 0 && IsDataTail(n, idx + ".onnx_data".Length))
            return true;

        return false;
    }

    /// <summary>
    /// True when the chars after the matched <c>.onnx.data</c> / <c>.onnx_data</c> token are either empty,
    /// or a shard suffix like <c>_0</c>, <c>_12</c>, <c>.0</c>, <c>-1</c>.
    /// </summary>
    private static bool IsDataTail(string s, int start)
    {
        if (start == s.Length) return true; // exact ".onnx.data" / ".onnx_data"

        char sep = s[start];
        if (sep != '_' && sep != '.' && sep != '-') return false;
        if (start + 1 >= s.Length) return false;
        for (int i = start + 1; i < s.Length; i++)
        {
            if (!char.IsAsciiDigit(s[i])) return false;
        }
        return true;
    }

    /// <summary>
    /// True if <paramref name="fileName"/> is external data associated specifically with
    /// <paramref name="onnxName"/> (e.g. <c>model.onnx.data</c> for <c>model.onnx</c>), tolerating the
    /// <c>.onnx_data</c> variant and shard suffixes.
    /// </summary>
    private static bool IsExternalDataFor(string fileName, string onnxName)
    {
        // onnxName is "<stem>.onnx". External data starts with either "<stem>.onnx.data"
        // or "<stem>.onnx_data" (the literal ".onnx" gets turned into ".onnx." or ".onnx_").
        if (!onnxName.EndsWith(OnnxExtension, StringComparison.OrdinalIgnoreCase))
            return false;

        string withDot = onnxName + ".data";   // <stem>.onnx.data
        string withUnderscore = onnxName + "_data"; // <stem>.onnx_data
        // But ".onnx_data" replaces the trailing ".onnx" — build that form too.
        string stem = onnxName.Substring(0, onnxName.Length - OnnxExtension.Length);
        string underscoreForm = stem + ".onnx_data"; // == withUnderscore actually

        if (StartsWithAndDataTail(fileName, withDot)) return true;
        if (StartsWithAndDataTail(fileName, withUnderscore)) return true;
        if (StartsWithAndDataTail(fileName, underscoreForm)) return true;
        return false;
    }

    private static bool StartsWithAndDataTail(string fileName, string prefix)
    {
        if (!fileName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            return false;
        return IsDataTail(fileName, prefix.Length);
    }

    // --- selection ranking -----------------------------------------------

    /// <summary>
    /// Lower is more canonical. <c>onnx/model.onnx</c> = 0, <c>model.onnx</c> (other dir) = 1,
    /// <c>model_*.onnx</c> = 2, anything else = 3.
    /// </summary>
    private static int CanonicalRank(string path)
    {
        string p = Normalize(path);
        string name = FileName(p);
        string dir = DirectoryOf(p);

        bool inOnnxDir = string.Equals(dir, "onnx", StringComparison.OrdinalIgnoreCase) ||
                         dir.EndsWith("/onnx", StringComparison.OrdinalIgnoreCase);

        if (string.Equals(name, "model.onnx", StringComparison.OrdinalIgnoreCase))
            return inOnnxDir ? 0 : 1;

        if (name.StartsWith("model_", StringComparison.OrdinalIgnoreCase) && IsOnnx(name))
            return 2;

        return 3;
    }

    /// <summary>Deterministic tie-break: shallower path, then shorter, then ordinal.</summary>
    private static bool IsBetterTieBreak(string candidate, string current)
    {
        string c = Normalize(candidate);
        string u = Normalize(current);

        int cd = DepthOf(c), ud = DepthOf(u);
        if (cd != ud) return cd < ud;
        if (c.Length != u.Length) return c.Length < u.Length;
        return string.CompareOrdinal(c, u) < 0;
    }

    // --- path helpers ----------------------------------------------------

    /// <summary>Returns everything before the last <c>/</c> in <paramref name="path"/>, or "" if none.</summary>
    private static string DirectoryOf(string path)
    {
        string p = Normalize(path);
        int slash = p.LastIndexOf('/');
        return slash < 0 ? string.Empty : p.Substring(0, slash);
    }

    /// <summary>Returns the final path segment (the file name) of <paramref name="path"/>.</summary>
    private static string FileName(string path)
    {
        string p = Normalize(path);
        int slash = p.LastIndexOf('/');
        return slash < 0 ? p : p.Substring(slash + 1);
    }

    private static int DepthOf(string path)
    {
        int depth = 0;
        for (int i = 0; i < path.Length; i++)
            if (path[i] == '/') depth++;
        return depth;
    }

    /// <summary>Normalises path separators to <c>/</c> and trims a leading <c>./</c> or <c>/</c>.</summary>
    private static string Normalize(string path)
    {
        if (string.IsNullOrEmpty(path)) return string.Empty;
        string p = path.Replace('\\', '/').Trim();
        while (p.StartsWith("./", StringComparison.Ordinal)) p = p.Substring(2);
        while (p.StartsWith("/", StringComparison.Ordinal)) p = p.Substring(1);
        return p;
    }

    /// <summary>True if <paramref name="path"/> ends with <paramref name="suffix"/> on a segment boundary.</summary>
    private static bool EndsWithSegment(string path, string suffix)
    {
        if (!path.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)) return false;
        int at = path.Length - suffix.Length;
        return at == 0 || path[at - 1] == '/';
    }

    private static int IndexOfOrdinalIgnoreCase(string s, string value) =>
        s.IndexOf(value, StringComparison.OrdinalIgnoreCase);
}
