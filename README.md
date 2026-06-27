<div align="center">
  <img src="modelsharp_logo.png" alt="ModelSharp Logo" width="120" height="120" />

  <h1>ModelSharp</h1>

  <p><strong>Universal, manifest-driven, pure-managed model inference for .NET.</strong></p>

  <p>
    <a href="https://www.nuget.org/packages/ModelSharp"><img src="https://img.shields.io/nuget/v/ModelSharp.svg?label=nuget&color=blue" alt="NuGet" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License: Apache-2.0" /></a>
    <img src="https://img.shields.io/badge/.NET-10.0-512BD4.svg" alt=".NET 10" />
    <img src="https://img.shields.io/badge/dependencies-zero-brightgreen.svg" alt="Zero dependencies" />
  </p>
</div>

---

ModelSharp aims to be to model inference what [ImageSharp](https://github.com/SixLabors/ImageSharp) is to imaging: a single, **zero-native-dependency**, cross-platform library where *any* model — vision, text, audio — "just runs" on CPU today and GPU as it matures, with **no Python, no native DLLs, and no per-model glue code**.

```csharp
using ModelSharp.Pipeline;

using var pipeline = Pipeline.Load("all-MiniLM-L6-v2.onnx");
float[] embedding = pipeline.Run<float[]>("A man is playing a guitar.");
// → a 384-dim, L2-normalized sentence embedding. No tokenizer setup, no native runtime.
```

## Why ModelSharp?

ONNX Runtime gives you `tensor in → tensor out`, but every model still needs its own pre/post-processing, and the native runtime can't run everywhere. ModelSharp closes both gaps:

1. **Self-describing models.** A small *manifest* (embedded ONNX metadata, a sidecar JSON, or a built-in registry) describes how to feed and decode a model — so one `Pipeline` API runs any model, text in or image in, typed result out.
2. **Pure-managed engine.** Managed kernels (SIMD-friendly, GPU via ILGPU) mean one build runs on Windows / Linux / macOS, x64 and ARM64, with **no native binaries to ship**. Even the ONNX parser is a hand-rolled protobuf reader — there isn't a `Google.Protobuf` dependency either.

## Features

- 🧩 **One-line inference** — `Pipeline.Load("model.onnx").Run<T>(input)` for any supported task.
- 📦 **Zero dependencies in the core package** — no native runtime, no Python, no protobuf library.
- 🌍 **Runs everywhere .NET runs** — single managed build, x64 / ARM64, all OSes.
- 🔢 **Multi-dtype engine** — `float32` / `int64` / `int32` / `bool` flow through as their real types (token ids, masks, and shape tensors included).
- 🧠 **191 operators** out of the box — CNNs, transformers, RNNs (LSTM/GRU), signal ops (DFT/STFT/MelWeightMatrix), control-flow (If/Loop/Scan), sequence/optional ops, and quantized `QLinear*` ops included.
- ♻️ **Runs *any* model on the GPU** — the ILGPU backend executes natively-supported ops on-device and transparently falls back to the CPU kernel for the rest, so any CPU-runnable model also runs through the GPU engine.
- 🔁 **Encoder-decoder generation** — T5 / BART / MarianMT-style seq2seq (encoder-once + cross-attention KV-cached decode), alongside decoder-only LLM generation.
- 🔤 **Built-in tokenizers** — WordPiece (BERT) and byte-level BPE (GPT-2 / RoBERTa), pure managed.
- 🎙️ **Audio front end** — FFT, log-mel spectrograms, and CTC decoding (greedy + prefix-beam).
- 🔌 **Swappable backends** — the same API runs on the managed CPU engine or the optional ILGPU GPU engine.
- 🖼️ **Optional image adapter** — image → tensor and top-K classification decoding via ImageSharp.

## Verified on real models

ModelSharp has been validated **end to end on real, exported ONNX models** — with **no Python at inference time and no native dependencies** — across every supported task:

| Task | Model | Result |
|------|-------|--------|
| Embedding | all-MiniLM-L6-v2 | 384-d semantic embeddings (cosine **0.70** paraphrase vs **−0.05** unrelated) |
| Text generation (LLM) | distilgpt2 | greedy, deterministic decode — `"The quick brown fox"` → `"es are a common sight in the wild, and are often found in the wild"` |
| Image classification | ResNet50 | top-1 **"tiger cat" (82%)** on a sample image |
| Object detection | YOLOv8 | detects **2 cats** with well-formed boxes (auto layout detection) |
| Speech recognition (CTC) | wav2vec2-base-960h | transcribes a LibriSpeech clip exactly as `"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"` |
| GPU *(optional ILGPU backend)* | NVIDIA RTX 4090 (CUDA) | GPU outputs match the CPU engine across **40+ ops**; large MatMul **~556×** and Conv2D **~109×** faster than the managed CPU engine |
| GPU LLM path | distilgpt2 on CUDA | the **full 1569-node graph runs end-to-end through the GPU engine** (no CPU fallback), matching the CPU engine's logits (Δ ≤ 1.8e-4) and exact greedy argmax; a full decoder layer + multi-step decode run on an **on-device KV-cache** |
| Quantized LLM on GPU | INT8 gpt2 (ONNX, dynamic-quant) | the whole quantized graph (48× `DynamicQuantizeLinear`→`MatMulInteger`) runs through the GPU engine and **greedy-decodes with the exact same argmax as the CPU engine** at every step |

The full test suite is **715 passing (0 failed)** and op coverage is **180 of ~190** standard ONNX ops (plus `QLinear*` quantized and contrib/fused ops). Quantized ONNX models load and run (`uint8`/`int8` and `fp16`/`bf16` initializers, dtype-generic Gather, a native on-device `MatMulInteger` GEMM), every GGUF quant type (legacy, k-quant, and IQ) dequantizes, and Whisper-style ASR runs through the seq2seq path (exact 400-point DFT mel front-end). The real-model integration tests are **opt-in**: they run when the model files are present — via `MODELSHARP_MODELS_DIR` or a repo-relative `models/` directory — and skip cleanly otherwise.

## Installation

```bash
dotnet add package ModelSharp              # core: load + run any ONNX model on CPU, plus text & audio front ends
dotnet add package ModelSharp.ImageSharp   # optional: image decoding & classification
dotnet add package ModelSharp.Gpu          # optional: ILGPU GPU backend
```

> Requires **.NET 10**. The core `ModelSharp` package has **no external dependencies**.

## Quick start

### Text embeddings (sentence-transformers)

```csharp
using ModelSharp.Pipeline;

// Manifest is resolved automatically (sidecar JSON → ONNX metadata → built-in registry).
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
    Console.WriteLine(r);   // e.g. "tabby cat (78.4%)"
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

## How manifests work

A *manifest* tells the pipeline how to feed and decode a model (task, image size, normalization, labels, vocab path, …). `Pipeline.Load` resolves one automatically, with this precedence:

1. **Sidecar JSON** next to the model — `model.onnx.manifest.json` or `model.manifest.json`.
2. **Embedded ONNX metadata** — keys such as `task`, `vocab`, `mean`, `layout`, `color` in the model's `metadata_props`.
3. **Built-in registry** — filename heuristics for common families (e.g. `*minilm*`, `*sentence*` → embedding; `*resnet*`, `*mobilenet*` → ImageNet classification).

A sidecar manifest is just JSON:

```json
{
  "task": "Embedding",
  "extra": { "vocab": "vocab.txt" }
}
```

Or pass one explicitly to bypass resolution entirely:

```csharp
var manifest = new ModelManifest
{
    Task = ModelTask.Embedding,
    Extra = new Dictionary<string, string> { ["vocab"] = "vocab.txt" },
};
using var pipeline = Pipeline.Load("model.onnx", manifest);
```

Need a task ModelSharp doesn't ship? Register your own pre/post processors with `ProcessorRegistry.RegisterPreprocessor` / `RegisterPostprocessor` — that's exactly how the ImageSharp package plugs itself in.

## Architecture

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

The `IExecutionEngine` seam is the key design decision: the public API and all pre/post-processing are fixed, while engines and kernel coverage grow underneath without breaking callers.

```csharp
public interface IExecutionEngine : IDisposable
{
    IReadOnlyList<TensorInfo> Inputs { get; }
    IReadOnlyList<TensorInfo> Outputs { get; }
    IReadOnlyDictionary<string, NamedTensor> Run(IReadOnlyDictionary<string, NamedTensor> feeds);
}
```

## Packages

Packaging is split by **dependency**, not by feature — so the common case is a single download.

| Package | Contains | External deps |
|---------|----------|---------------|
| **`ModelSharp`** | Everything dependency-free: ONNX loader (hand-rolled protobuf), multi-dtype CPU engine + 46-op kernel registry, tensors, manifest resolver + auto-wired pipeline, the FFT / log-mel audio front end + CTC decoder, and the WordPiece + BPE text tokenizers. | **none** |
| `ModelSharp.ImageSharp` *(optional)* | Image → tensor preprocessing + top-K classification decoding. | SixLabors.ImageSharp (3.x) |
| `ModelSharp.Gpu` *(optional)* | ILGPU backend (C# kernels → CUDA / OpenCL, CPU fallback). | ILGPU (1.5.x) |

Inside the core assembly the areas keep their own namespaces (`ModelSharp.Onnx`, `ModelSharp.Cpu`, `ModelSharp.Text`, `ModelSharp.Audio`, `ModelSharp.Pipeline`) — one DLL, clean separation.

## Capabilities

### Operator coverage (191)

- **Arithmetic** (broadcasting): Add, Sub, Mul, Div, Pow
- **Activations**: Relu, Sigmoid, Tanh, Exp, Log, Sqrt, Abs, Neg, Erf, Gelu, Identity, LeakyRelu, Clip, Softmax
- **NN layers**: Conv (auto_pad / dilations / group / bias), MaxPool, GlobalAveragePool, BatchNormalization, LayerNormalization
- **Recurrent**: LSTM (peepholes, clip, input_forget, sequence_lens), GRU (clip, sequence_lens) — forward / reverse / bidirectional, optional bias + initial state
- **Linear**: MatMul (n-D batched, NumPy semantics), Gemm
- **Reduction**: ReduceMean (axes, keepdims)
- **Logical** (bool out, broadcasting): Where, Equal, Less, Greater
- **Shape / data**: Reshape, Flatten, Concat, Transpose, Gather, Unsqueeze, Squeeze, Cast (typed), Shape, Constant, ConstantOfShape, Slice, Expand, Trilu, ScatterND, Range
- **Signal**: DFT (+ FFT fast path), STFT, MelWeightMatrix (opset-17 audio front-end ops)
- **Control flow**: If, Loop, Scan (full ONNX subgraph parsing + execution)
- **Sequence / Optional**: SequenceEmpty/Construct/Insert/Erase/At/Length, SplitToSequence, ConcatFromSequence, Optional/OptionalGetElement/OptionalHasElement
- **Quantized**: DequantizeLinear, QuantizeLinear, DynamicQuantizeLinear, MatMulInteger, QLinearMatMul, QLinearConv, QLinearAdd, QLinearMul, ConvInteger, QLinearGlobalAveragePool

> The list above is a representative sample; the registry now covers **180 of ~190** standard ops plus `QLinear*` quantized and contrib/fused ops. Any op without a native GPU kernel still runs on the GPU engine via CPU fallback. Additional kernels are wired in as they're verified.

### Text

- **`WordPieceTokenizer`** — BERT-family WordPiece with full basic tokenization: punctuation & CJK splitting, accent stripping, NFC normalization, optional lowercasing, and `[CLS]`/`[SEP]`/`[UNK]` handling.
- **`BpeTokenizer`** — byte-level BPE for GPT-2 / RoBERTa: GPT-2 regex pre-tokenization, reversible UTF-8 ↔ byte-level mapping, rank-based merges, and full round-trip encode/decode (emoji included).
- Both load from standard artifacts (`vocab.txt`, or `vocab.json` + `merges.txt`) and are pure managed code.

### Audio

- **`Fft`** — radix-2 Cooley–Tukey FFT and magnitude spectrum.
- **`MelSpectrogram`** — Slaney-scale triangular mel filterbank and log-mel spectrograms (Whisper-style front end).
- **`CtcDecoder`** — CTC decoding for acoustic models: greedy (best-path) and prefix-beam search, with `CtcVocabulary` for token → text rendering.

### GPU (optional)

The `ModelSharp.Gpu` package provides an `IlgpuEngine` that JIT-compiles C# kernels to CUDA / OpenCL with a CPU fallback — selected automatically:

```csharp
using ModelSharp.Gpu;

using var engine = new IlgpuEngine(graph);          // preferCpu: true to force the CPU accelerator
Console.WriteLine(engine.AcceleratorName);          // e.g. the CUDA device, or "CPUAccelerator"
var outputs = engine.Run(feeds);
```

GPU-accelerated ops today: broadcasting elementwise (Add/Sub/Mul/Div), ReLU, **MatMul**, and **Conv2D** — numerically verified against the CPU engine through the same `IExecutionEngine` seam.

## Status

**Verified end to end:**

- The bundled **MNIST** CNN is loaded by ModelSharp's own ONNX reader and run through the managed kernels, reproducing ONNX Runtime's reference output to within `1e-2`.
- The real pretrained **all-MiniLM-L6-v2** sentence-transformer runs end to end (tokenize → 6 transformer layers → mean-pooled embedding), producing semantically correct 384-d embeddings — cosine **0.70** for paraphrases vs **−0.05** for unrelated text — entirely through `Pipeline.Load(...).Run<float[]>(...)`.
- Text generation (**distilgpt2**), image classification (**ResNet50**), object detection (**YOLOv8**), and speech recognition (**wav2vec2-base-960h** via CTC) all run end to end on real exported models — see [Verified on real models](#verified-on-real-models) for the concrete outputs.

| Phase | State |
|-------|-------|
| 1. CNN core | ✅ verified (MNIST vs ONNX-Runtime reference) |
| 2. Sequence | ✅ LSTM + GRU verified (vs ONNX reference) |
| 3. Transformer | ✅ real pretrained all-MiniLM-L6-v2 (embeddings) + distilgpt2 (greedy generation) run end-to-end |
| 4. Audio | ✅ FFT + log-mel front end and CTC decoder verified end-to-end on real wav2vec2-base-960h (exact LibriSpeech transcription) |
| 5. GPU | ✅ ILGPU engine matches the CPU engine across 29 ops on a real RTX 4090 (CUDA); MatMul ~556× / Conv2D ~109× faster than managed CPU |

> This is an **alpha** (`0.1.0-alpha`). Every task above is verified on a real model, but operator and backend coverage is still growing — full breadth across the ONNX op set and GPU multi-dtype is the multi-quarter roadmap, not a finished product.

## Building from source

```bash
dotnet build
dotnet test
```

Everything targets `net10.0`.

## Notes

- **ImageSharp** is pinned to **v3.x**: v4 added a build-time commercial license-key gate that conflicts with the "works the moment you download it" goal.
- The ONNX loader is a hand-rolled protobuf reader, so the core package pulls in **nothing** — no `Google.Protobuf`, no native runtime.

## License

[Apache License 2.0](LICENSE) — permissive, with an explicit patent grant (the prevailing license across the ML inference ecosystem: ONNX, ONNX Runtime, PyTorch, TensorFlow).
