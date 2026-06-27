<div align="center">
  <img src="modelsharp_logo.png" alt="ModelSharp Logo" width="120" height="120" />

  <h1>ModelSharp</h1>

  <p><strong>Pure-managed, zero-native-dependency ONNX model inference for .NET.</strong></p>

  <p>
    <a href="https://www.nuget.org/packages/ModelSharp"><img src="https://img.shields.io/nuget/v/ModelSharp.svg?label=nuget&color=blue" alt="NuGet" /></a>
    <a href="https://www.nuget.org/packages/ModelSharp"><img src="https://img.shields.io/nuget/dt/ModelSharp.svg?label=downloads&color=blue" alt="NuGet downloads" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License: Apache-2.0" /></a>
    <img src="https://img.shields.io/badge/.NET-10.0-512BD4.svg" alt=".NET 10" />
    <img src="https://img.shields.io/badge/dependencies-zero-brightgreen.svg" alt="Zero dependencies" />
    <img src="https://img.shields.io/badge/tests-passing-brightgreen.svg" alt="Well tested" />
  </p>
</div>

---

**ModelSharp** runs real machine-learning models — vision, text, and audio — entirely in managed .NET. No Python, no native DLLs, no per-model glue code. Point it at an ONNX, GGUF, or safetensors model and call one method: a small *manifest* describes how to feed and decode the model, so the same `Pipeline` API handles embeddings, image classification, object detection, speech recognition, and quantized LLM generation. The same build runs on Windows, Linux, and macOS, x64 and ARM64, on CPU — and on the GPU through an optional backend.

```csharp
using ModelSharp.Hub;

// Downloads the model + tokenizer, then runs it — one line.
using var pipeline = HubPipeline.Load("qwen2.5-0.5b-int4");
string answer = pipeline.Run<string>("The capital of France is");
// → " Paris. It is the largest city in the world by population…"
```

## Why ModelSharp

ONNX Runtime gives you `tensor in → tensor out`, but every model still needs its own pre/post-processing, and the native runtime can't run everywhere. ModelSharp closes both gaps:

- **Self-describing models.** A small *manifest* — embedded ONNX metadata, a sidecar JSON, or a built-in registry — describes how to feed and decode a model. One `Pipeline` API runs any model: text in or image in, typed result out, no per-model glue code.
- **Pure-managed engine.** Managed kernels (SIMD-friendly on CPU, GPU via ILGPU) mean a single build runs everywhere .NET runs, with **no native binaries to ship**. Even the ONNX parser is a hand-rolled protobuf reader — there isn't a `Google.Protobuf` dependency either.
- **Bit-verified.** Outputs are validated against ONNX Runtime on real models, down to exact next-token logits on quantized LLMs.

## Features

- 🧩 **One-line inference** — `Pipeline.Load("model.onnx").Run<T>(input)` for any supported task.
- 📦 **Zero dependencies in the core package** — no native runtime, no Python, no protobuf library.
- 🌍 **Runs everywhere .NET runs** — single managed build, x64 / ARM64, all OSes.
- 🔢 **Multi-dtype engine** — `float32` / `int64` / `int32` / `bool` flow through as their real types, so token ids, masks, and shape tensors all work natively.
- 🧠 **Broad operator coverage** — CNNs, transformers, RNNs (LSTM/GRU), signal ops (DFT/STFT/MelWeightMatrix), control flow (If/Loop/Scan), sequence/optional ops, and quantized `QLinear*` / `MatMulNBits` ops.
- 🧮 **Real quantized LLMs** — runs INT4 / INT8 / fp16 ONNX LLMs end to end, including a **7B** model (`MatMulNBits` INT4 + genai `GroupQueryAttention`) loaded from multi-gigabyte external-data files.
- 🔁 **Encoder-decoder & decoder-only generation** — T5 / BART / MarianMT-style seq2seq (KV-cached decode) alongside decoder-only LLM generation.
- 🔤 **Built-in tokenizers** — WordPiece (BERT) and byte-level BPE (GPT-2 / RoBERTa), pure managed.
- 🎙️ **Audio front end** — FFT, log-mel spectrograms, and CTC decoding (greedy + prefix-beam) for ASR.
- ♻️ **Runs any model on the GPU** — the optional ILGPU backend executes supported ops on-device and falls back to the CPU kernel for the rest, so anything that runs on CPU also runs through the GPU engine.
- ⬇️ **Optional model hub** — `HubPipeline.Load("qwen2.5-0.5b-int4")` downloads a model (plus its external-data shards and tokenizer) from Hugging Face, GGUF, safetensors, or any URL and runs it, with a local cache. Pure-managed (`HttpClient` only).

## Installation

```bash
dotnet add package ModelSharp              # core: load + run any ONNX model on CPU, plus text & audio front ends
dotnet add package ModelSharp.ImageSharp   # optional: image decoding & classification
dotnet add package ModelSharp.Gpu          # optional: ILGPU GPU backend
dotnet add package ModelSharp.Hub          # optional: download models from Hugging Face / URLs
```

Requires **.NET 10**. The core `ModelSharp` package has **no external dependencies**.

## Quick start

### Download & run from the hub

```csharp
using ModelSharp.Hub;

// Downloads the model + tokenizer/config (and any external-data shards) into a local cache, then runs it.
using var pipeline = HubPipeline.Load("qwen2.5-0.5b-int4");
string answer = pipeline.Run<string>("The capital of France is");

// …or resolve any Hugging Face repo / file, GGUF, safetensors, or direct URL:
ResolvedModel m = ModelHub.Get("onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4.onnx");
// m.ModelPath is the local file; m.Files lists the model + tokenizer + config that came with it.
```

### Text embeddings

```csharp
using ModelSharp.Pipeline;

// The manifest is resolved automatically (sidecar JSON → ONNX metadata → built-in registry).
using var pipeline = Pipeline.Load("all-MiniLM-L6-v2.onnx");

float[] a = pipeline.Run<float[]>("A man is playing a guitar.");
float[] b = pipeline.Run<float[]>("Someone strums an acoustic guitar.");
float[] c = pipeline.Run<float[]>("The stock market fell sharply today.");

// a·b ≈ 0.70 (paraphrase)   a·c ≈ -0.05 (unrelated)
```

### Image classification

```csharp
using ModelSharp.Pipeline;
using ModelSharp.ImageSharp;

ImageSharpRegistration.Ensure();   // wire the image processors (or just reference the assembly)

using var pipeline = Pipeline.Load("resnet50.onnx");
var results = pipeline.Run<List<Classification>>("cat.jpg");   // also accepts byte[], Stream, or Image<Rgb24>

foreach (var r in results.Take(3))
    Console.WriteLine(r);   // e.g. "tiger cat (82%)"
```

### LLM text generation

```csharp
using ModelSharp.Pipeline;

using var pipeline = Pipeline.Load("Qwen2.5-0.5B-Instruct-q4.onnx");
string text = pipeline.Run<string>("The capital of France is");
// → " Paris. It is the largest city in the world by population…"
```

### Speech recognition

```csharp
using ModelSharp.Pipeline;

using var pipeline = Pipeline.Load("whisper-tiny.onnx");
string transcript = pipeline.Run<string>("audio.wav");
// → "Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
```

### Raw graph execution (full control)

When you want tensors in and tensors out — no manifest, no processors:

```csharp
using ModelSharp.Onnx;
using ModelSharp.Cpu;
using ModelSharp.Tensors;

ModelGraph graph = OnnxModelLoader.LoadModel("model.onnx");
using var engine = new ManagedCpuEngine(graph);

var feeds = new Dictionary<string, NamedTensor>
{
    ["input_ids"] = new NamedTensor(
        "input_ids",
        Tensor<long>.FromArray(new TensorShape(1, 5), new long[] { 101, 2054, 2003, 2009, 102 })),
};

IReadOnlyDictionary<string, NamedTensor> outputs = engine.Run(feeds);
Tensor<float> hidden = outputs["last_hidden_state"].Tensor.AsFloat();
```

## Supported tasks & formats

| | |
|---|---|
| **Model formats** | ONNX (incl. external-data shards), GGUF (legacy, k-quant, and IQ quant types), safetensors |
| **Quantization** | INT4 (`MatMulNBits`), INT8 (`QLinear*` / dynamic-quant), fp16 / bf16 initializers |
| **Tasks** | Text embeddings, image classification, object detection, speech recognition (CTC + Whisper seq2seq), decoder-only LLM generation, encoder-decoder (T5 / BART / MarianMT) generation |
| **Backends** | Managed CPU engine (core), optional ILGPU GPU engine (CUDA / OpenCL, CPU fallback) |

## Verified on real models

Every supported task is validated **end to end on real, exported models** — with **no Python at inference time and no native dependencies**:

| Task | Model | Result |
|------|-------|--------|
| Embedding | all-MiniLM-L6-v2 | 384-d semantic embeddings (cosine **0.70** paraphrase vs **−0.05** unrelated) |
| Text generation (LLM) | distilgpt2 | deterministic greedy decode: `"The quick brown fox"` → `"es are a common sight in the wild…"` |
| Image classification | ResNet50 | top-1 **"tiger cat" (82%)** |
| Object detection | YOLOv8 | detects **2 cats** with well-formed boxes (auto layout detection) |
| Speech recognition (CTC) | wav2vec2-base-960h | transcribes a LibriSpeech clip exactly |
| Whisper ASR | whisper-tiny | `"Mr. Quilter is the apostle of the middle classes…"` (log-mel → seq2seq decode) |
| INT4 LLM (text) | Qwen2.5-0.5B-Instruct (INT4 q4) | forward pass in **~2 s**; logits **match ONNX Runtime exactly**; `"The capital of France is"` → `" Paris…"` |
| **INT4 LLM (7B)** | **Mistral-7B-Instruct v0.3** (genai INT4) | a **5 GB external-data** model runs a full forward pass in **~16 s** on an RTX 4090; next-token logits **match ONNX Runtime exactly**, bit-verifying the `MatMulNBits` + `GroupQueryAttention` path |
| Quantized LLM on GPU | INT8 gpt2 (dynamic-quant) | the whole quantized graph runs on the GPU engine and greedy-decodes with the **same argmax as CPU** at every step |
| GPU LLM path | distilgpt2 on CUDA | the **full graph runs end-to-end on the GPU** (no CPU fallback), matching CPU logits (Δ ≤ 1.8e-4) and exact greedy argmax, with an on-device KV-cache |
| GPU acceleration | RTX 4090 (CUDA) | GPU outputs match the CPU engine; large MatMul **~556×** and Conv2D **~109×** faster than the managed CPU engine |

## How it works

ModelSharp is built around a single seam: a manifest-driven `Pipeline` on top of a swappable execution engine.

```
            +-----------------------------------------------+
  input --> |  Pipeline  (manifest-driven, engine-agnostic) | --> typed result
            +------+--------------------------+--------------+
             IPreprocessor             IPostprocessor
                          |
                          v
                 IExecutionEngine               <-- swappable backend
                    +-- ManagedCpuEngine  (ModelSharp core) -- pure-managed kernels
                    +-- IlgpuEngine       (ModelSharp.Gpu)  -- C# kernels -> CUDA / OpenCL / CPU
```

- **Manifest-driven pipeline.** A manifest resolves automatically — sidecar JSON next to the model, then embedded ONNX `metadata_props`, then a built-in registry of filename heuristics — or you pass one explicitly. It selects the right pre/post-processors so one API serves every task.
- **Pure-managed, multi-dtype engine.** Tensors carry their real dtype end to end. Kernels are written in plain managed C# and are SIMD-friendly, with no native code anywhere in the core.
- **Swappable backends.** The public API and all processing are fixed behind `IExecutionEngine`; the CPU and GPU engines plug in underneath without changing caller code.
- **Hand-rolled ONNX reader.** The loader is a custom protobuf parser, which is why the core package pulls in nothing — no `Google.Protobuf`, no native runtime.

Need a task ModelSharp doesn't ship? Register your own pre/post-processors with `ProcessorRegistry.RegisterPreprocessor` / `RegisterPostprocessor` — exactly how the ImageSharp package plugs itself in.

## Packages

Packaging is split by **dependency**, not by feature — so the common case is a single download.

| Package | Contains | External deps |
|---------|----------|---------------|
| **`ModelSharp`** | Everything dependency-free: ONNX loader (hand-rolled protobuf), multi-dtype CPU engine + kernel registry, tensors, manifest resolver + auto-wired pipeline, the FFT / log-mel audio front end + CTC decoder, and the WordPiece + BPE tokenizers. | **none** |
| `ModelSharp.ImageSharp` *(optional)* | Image → tensor preprocessing + top-K classification decoding. | SixLabors.ImageSharp (3.x) |
| `ModelSharp.Gpu` *(optional)* | ILGPU backend (C# kernels → CUDA / OpenCL, CPU fallback). | ILGPU (1.5.x) |
| `ModelSharp.Hub` *(optional)* | Model download + resolution from Hugging Face / GGUF / safetensors / URLs, with a local cache. | **none** (`HttpClient` only) |

## Requirements

- **.NET 10** or later.
- The core `ModelSharp` package has no external or native dependencies. Optional packages add only the managed dependencies listed above.

## License

[Apache License 2.0](LICENSE) — permissive, with an explicit patent grant (the prevailing license across the ML inference ecosystem: ONNX, ONNX Runtime, PyTorch, TensorFlow).

## Contributing

Issues and pull requests are welcome. The project targets `net10.0`; `dotnet build` and `dotnet test` build the solution and run the test suite. New operator kernels and model-task processors are wired in behind the stable `IExecutionEngine` and `ProcessorRegistry` seams, so contributions extend coverage without breaking the public API.
