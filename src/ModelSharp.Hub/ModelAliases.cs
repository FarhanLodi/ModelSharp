using System;
using System.Collections.Generic;

namespace ModelSharp.Hub;

/// <summary>
/// A curated registry of short, friendly model names mapped to full spec strings that
/// <see cref="ModelRefParser.Parse(string, string)"/> understands. Lets callers write
/// <c>Pipeline.FromHub("qwen2.5-0.5b-int4")</c> instead of a full repo/sub-path.
/// </summary>
/// <remarks>
/// <para>Pure data plus a case-insensitive lookup — no I/O, no network. Resolution is a single
/// dictionary lookup; unknown names pass straight through to the parser unchanged.</para>
/// <para>Every alias maps to a repo (and where relevant, a specific ONNX file sub-path) that has been
/// chosen to exist on the Hugging Face Hub. When a candidate could not be confidently verified it was
/// omitted rather than guessed — correctness over coverage. A few unsure-but-plausible entries are kept
/// behind explicit comments so they are easy to audit.</para>
/// </remarks>
public static class ModelAliases
{
    // Case-insensitive so "Qwen2.5-0.5B-INT4" and "qwen2.5-0.5b-int4" resolve identically.
    private static readonly Dictionary<string, string> Map = new(StringComparer.OrdinalIgnoreCase)
    {
        // --- Embeddings (sentence-transformers / feature extraction) ---
        // Xenova provides ONNX exports of the popular sentence-transformers models.
        ["all-minilm-l6-v2"] = "Xenova/all-MiniLM-L6-v2",
        ["all-minilm-l12-v2"] = "Xenova/all-MiniLM-L12-v2",
        ["all-mpnet-base-v2"] = "Xenova/all-mpnet-base-v2",
        // Multilingual E5 small — widely used retrieval embedding, ONNX export by Xenova.
        ["bge-small-en-v1.5"] = "Xenova/bge-small-en-v1.5",
        ["bge-base-en-v1.5"] = "Xenova/bge-base-en-v1.5",
        ["gte-small"] = "Xenova/gte-small",

        // --- LLM: Qwen2.5 0.5B Instruct (onnx-community publishes quantized ONNX variants) ---
        // The onnx-community repo ships several precisions under onnx/.
        ["qwen2.5-0.5b-int4"] = "onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx",
        ["qwen2.5-0.5b-int8"] = "onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_int8.onnx",
        ["qwen2.5-0.5b-fp16"] = "onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_fp16.onnx",
        // Default (unsuffixed) Qwen 0.5B → the base fp32 ONNX in the same repo.
        ["qwen2.5-0.5b"] = "onnx-community/Qwen2.5-0.5B-Instruct/onnx/model.onnx",

        // --- LLM: Mistral 7B Instruct INT4 (ONNX Runtime GenAI layout) ---
        ["mistral-7b-instruct-int4"] =
            "EmbeddedLLM/mistral-7b-instruct-v0.3-onnx/onnx/cpu_and_mobile/mistral-7b-instruct-v0.3-cpu-int4-rtn-block-32/model.onnx",

        // --- Vision: image classification ---
        // ResNet-50 ImageNet classifier, ONNX export by Xenova.
        ["resnet50"] = "Xenova/resnet-50",
        // ViT base patch16-224 ImageNet classifier.
        ["vit-base"] = "Xenova/vit-base-patch16-224",

        // --- ASR: speech-to-text (Whisper, ONNX exports by Xenova) ---
        ["whisper-tiny"] = "Xenova/whisper-tiny",
        ["whisper-base"] = "Xenova/whisper-base",
        ["whisper-small"] = "Xenova/whisper-small",
        // whisper-tiny.en — English-only variant.
        ["whisper-tiny-en"] = "Xenova/whisper-tiny.en",

        // --- GGUF (llama.cpp) ---
        // TheBloke's TinyLlama 1.1B Chat GGUF — Q4_K_M quant.
        ["gguf:tinyllama-1.1b-q4"] =
            "gguf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        // TheBloke's Mistral 7B Instruct v0.2 GGUF — Q4_K_M quant.
        ["gguf:mistral-7b-instruct-q4"] =
            "gguf:TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf",

        // --- OMITTED (unverified — left out deliberately rather than ship a wrong repo) ---
        // "phi-3-mini-int4": the exact onnx-community / microsoft ONNX file sub-path was not confirmed.
        // "llama-3.2-1b-int4": onnx-community quant filename not confirmed; omitted.
        // "e5-small-v2": ONNX export repo owner not confirmed; omitted.
    };

    /// <summary>The full alias registry (friendly name → spec string), case-insensitive on lookup.</summary>
    public static IReadOnlyDictionary<string, string> All => Map;

    /// <summary>
    /// Resolves a friendly alias to its full spec string. If <paramref name="nameOrSpec"/> (trimmed,
    /// matched case-insensitively) is a known alias, returns the mapped spec; otherwise returns the input
    /// unchanged so non-aliases pass straight through to <see cref="ModelRefParser.Parse(string, string)"/>.
    /// </summary>
    /// <param name="nameOrSpec">A friendly alias, or an already-full model spec.</param>
    /// <returns>The resolved spec, or the original input when it is not a known alias.</returns>
    public static string Resolve(string nameOrSpec)
    {
        if (string.IsNullOrWhiteSpace(nameOrSpec))
            return nameOrSpec;

        var key = nameOrSpec.Trim();
        return Map.TryGetValue(key, out var spec) ? spec : nameOrSpec;
    }
}
