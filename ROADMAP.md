# ModelSharp — Implementation Handoff / Roadmap

> Handoff for continuing development on a GPU machine. Read this top-to-bottom, then start at
> Phase 0. The library is **pure-managed, CPU-first, manifest-driven ONNX inference for .NET** —
> "ImageSharp for model inference." Goal: any model (embedding / LLM / vision / audio) "just runs."

## 0. Current state (read first)

- **Target: `net10.0` ONLY.** Do **not** add `net8.0`/`net9.0` multi-targeting — this is a hard
  constraint the owner set. Single `<TargetFramework>net10.0</TargetFramework>` in every csproj.
- License: **Apache-2.0** (LICENSE + NOTICE at root).
- Build/test baseline: `dotnet test` must be **GREEN — 260 tests, 0 failures**. Run it before and
  after every change. If it's not 260 green on a fresh clone, stop and fix that first.
- Projects: `src/ModelSharp` (core, zero deps), `src/ModelSharp.ImageSharp` (image adapter,
  SixLabors.ImageSharp v3.x pinned), `src/ModelSharp.Gpu` (ILGPU backend), `tests/ModelSharp.Tests`.

### What is VERIFIED end-to-end (works on real models)
- CNN: bundled MNIST vs ONNX Runtime reference (1e-2).
- Embeddings: real **all-MiniLM-L6-v2** → semantic 384-d embeddings (paraphrase 0.70 vs unrelated −0.05).
- `Pipeline.Load("model.onnx").Run<float[]>(text)` works for text embeddings end-to-end.

### What is BUILT + UNIT-TESTED but NOT yet run on a real model
- **LLM generation stack**: `src/ModelSharp/Generation/` — `TextGenerator` (autoregressive loop,
  KV-cache + no-cache fallback), `LogitsProcessor` (greedy/temperature/top-k/top-p/repetition),
  `GenerationConfig`, `DecoderModelOptions`. Tested with a fake engine only.
- **BPE tokenizer**: `src/ModelSharp/Text/BpeTokenizer.cs` + `ByteLevel.cs` (GPT-2/RoBERTa byte-level).
- **WordPiece tokenizer** (BERT) with full basic-tokenization: `WordPieceTokenizer.cs` + `BasicTokenizer.cs`.
- **safetensors loader**: `src/ModelSharp/Weights/` (F16/BF16 decode, validation). ~2 GB array cap (see C4).
- **Vision object detection**: `src/ModelSharp.ImageSharp/` — `NonMaxSuppression`, `DetectionPostprocessor`
  (YOLOv5 `[1,N,5+C]` + YOLOv8 `[1,4+C,N]`), `Detection`. Registered for `ModelTask.ObjectDetection`.
- **Audio CTC**: `src/ModelSharp/Audio/CtcDecoder.cs` (greedy + prefix-beam) + `CtcVocabulary.cs`,
  on top of `Fft.cs` + `MelSpectrogram.cs`.
- **~88 CPU ops** in `src/ModelSharp/Cpu/Kernels/` (registered in `KernelRegistry.CreateDefault()`).
- **GPU ops** (`src/ModelSharp.Gpu/`): Add/Sub/Mul/Div + broadcasting, Relu/Sigmoid/Tanh/Gelu/LeakyRelu/
  Exp/Sqrt, Transpose, Softmax, ReduceSum/ReduceMean, MatMul (batched), Conv2D — **so far validated
  only on ILGPU's CPU accelerator** (`preferCpu: true`), never on real CUDA.

### Working method (this is how the codebase was built — keep doing it)
- Parallelize with subagents on **disjoint file sets**; do ONE central integration `dotnet build`/`test`
  afterward. Never let multiple agents run `dotnet build`/`test` concurrently (corrupts shared bin/obj).
- Watch for **namespace shadowing** (a `…Kernels.Math` namespace once shadowed `System.Math` — now
  `…Kernels.MathOps`). Don't reintroduce a namespace that collides with a BCL type.
- Commit green checkpoints often; branch off `main`.
- Store model files on a large drive, NOT the system drive.

---

## Phase 0 — GPU machine bring-up (do this first)
1. Clone repo, `dotnet build`, `dotnet test` → confirm **260 green** on net10.0.
2. Install the latest **NVIDIA driver + CUDA toolkit**. Verify `nvidia-smi` shows the GPU.
3. Confirm ILGPU sees the CUDA device: construct `new IlgpuEngine(graph, preferCpu: false)` and assert
   `engine.IsHardwareGpu == true` and `AcceleratorName` is the CUDA device. Add a test that **skips**
   when no CUDA device is present (so CI on CPU-only machines still passes), but runs the existing GPU
   op graphs on the real GPU and compares to the CPU engine.
4. Decide a models dir on a big drive (e.g. `D:\models`); keep it out of git (`.gitignore`).

**Done when:** 260 green + at least one GPU op test runs on the real CUDA accelerator and matches CPU.

---

## Phase A — Prove real models end-to-end (HIGHEST VALUE)
The biggest gap is that only embeddings have run on a real model. Each item below: get the model,
run an op-coverage probe (like `MiniLmTests.MiniLm_Op_Coverage_Probe`), wire/verify, add an opt-in
test that **skips if the model asset is absent** (mirror `MiniLmTests`).

- **A1 — Small LLM (the headline).** Export `distilgpt2` (or TinyLlama-1.1B) to ONNX with HF Optimum
  **as a non-merged decoder-with-past** (`--task text-generation-with-past`, NOT `*_merged`). Build:
  BPE tokenizer (`BpeTokenizer.FromFiles(vocab.json, merges.txt)`) → `TextGenerator` (set
  `DecoderModelOptions.KvCacheNumHeads`/`HeadDim` from config.json) → greedy decode → `BpeTokenizer.Decode`.
  Run the op probe first; implement any missing ops it reports (RoPE/RMSNorm usually appear as
  primitives in HF exports — Add/Mul/Sqrt/ReduceMean/Sin/Cos/Concat etc., mostly present).
  **Done when:** distilgpt2 produces coherent greedy continuation of a prompt, and sampled output is
  reproducible with a fixed seed.
- **A2 — Image classification.** A real ResNet50/MobileNet ONNX → `Pipeline.Load` (manifest sets
  ImageNet mean/std, 224×224, labels) → `Run<List<Classification>>("cat.jpg")`. **Done when:** correct
  top-1 on a few sample images.
- **A3 — Object detection.** A YOLOv5 or YOLOv8 ONNX → manifest `task=ObjectDetection`,
  `Extra["det_layout"]` = `yolov5`/`yolov8` → boxes. **Done when:** plausible boxes on a sample image
  (and op probe clean — YOLO often needs `Resize`, `Concat`, `Sigmoid`, maybe `Slice`; most are present).
- **A4 — ASR (CTC).** A `wav2vec2` CTC ONNX → `MelSpectrogram`/feature path → model → `CtcDecoder` +
  `CtcVocabulary` (blank=0). **Done when:** a short clip transcribes recognizably.

---

## Phase B — GPU acceleration (now that real hardware exists)
- **B1 — Validate the CUDA path** for every existing GPU op against the CPU engine on real hardware
  (Phase 0.3 generalizes here). Fix any kernels that pass on the CPU accelerator but diverge on CUDA.
- **B2 — GPU multi-dtype.** The GPU engine is float32-only; CPU is multi-dtype. Add at least int64/int32
  buffers so token-id/mask/shape tensors can live on-device (needed to keep LLM graphs on GPU).
- **B3 — GPU op parity.** Bring the GPU op set up toward the CPU set for the ops real models hit
  (LayerNorm, Gather, Concat, Slice, element-wise broadcasting everywhere, Cast). Each verified vs CPU.
- **B4 — Performance.** Tiled MatMul, fused attention if feasible; benchmark GPU vs CPU on a real model
  and report speedups. (Correctness first, then speed.)
- **B5 — Whole-graph on GPU + GPU KV-cache.** Run an entire decoder forward pass on GPU with the KV
  cache kept on-device between steps (avoid host round-trips per token). This is the real LLM win.

---

## Phase C — LLM productionization
- **C1 — `use_cache_branch`.** Support Optimum **merged** decoder exports (the most common form): they
  add a boolean `use_cache_branch` input and unify prefill/decode. Extend `TextGenerator`/
  `DecoderModelOptions` to feed it. (Audit flagged this as the #1 real-LLM blocker.)
- **C2 — `Pipeline.Load` for TextGeneration.** Wire BPE tokenizer + `TextGenerator` + manifest behind
  the one-line API (a `TextGenerationPreprocessor`/postprocessor + registry entry for
  `ModelTask.TextGeneration`), so completion is as simple as embeddings are today.
- **C3 — Quantization (required for big models on GPU).** int8 + int4 dequant kernels; load GPTQ/AWQ /
  HF quantized safetensors; a quantized Linear/MatMul path. Without this, 7B+ won't fit.
- **C4 — safetensors > 2 GB.** Replace `File.ReadAllBytes` with memory-mapped / chunked reads so large
  shards load (current ~2 GB `byte[]` cap). Parse the `*.index.json` to load sharded checkpoints.
- **C5 — GGUF loader.** Add a GGUF reader (llama.cpp ecosystem) alongside ONNX + safetensors.
- **C6 — Fused LLM ops** if a target model exports them: `RotaryEmbedding`, `SimplifiedLayerNormalization`
  (RMSNorm), grouped-query attention, `MultiHeadAttention` contrib op. Driven by the per-model op probe.

---

## Phase D — Op coverage & correctness cleanups
- Extend ONNX op coverage as models demand (we're ~88 of ~190 standard ops). Add via new kernel files +
  `KernelRegistry`, each with a `NewOpsTests`-style unit test.
- **Minor fixes from the audit (small, non-blocking):**
  - `MeanPoolEmbeddingPostprocessor` looks for `attention_mask` among model **outputs** (it's normally an
    **input**) → it pools over all tokens; fix so padded/batched inputs mask correctly.
  - Add unit tests for `Tan` and `ReduceSumSquare` (implemented + registered, currently untested).
  - Add a manifest test that pits a sidecar JSON against conflicting embedded metadata (precedence).

---

## Definition of "go live"
A user can `Pipeline.Load(...)` (or a thin generation API) and run, with no Python and no native deps:
an **embedding** model (done), an **image classifier**, an **object detector**, an **ASR** model, and a
**text-completion LLM** — at least the small ones on CPU, and 7B-class on the GPU via quantization.
Each proven by an opt-in test against a real model asset.
