# ModelSharp — Implementation Handoff / Roadmap

> Handoff for continuing development on a GPU machine. Read this top-to-bottom, then start at
> Phase 0. The library is **pure-managed, CPU-first, manifest-driven ONNX inference for .NET** —
> "ImageSharp for model inference." Goal: any model (embedding / LLM / vision / audio) "just runs."

## Progress log

All **pure-managed, code-level** roadmap items are implemented + unit-tested, and everything previously
pending hardware/assets has now been **validated on a real RTX 4090 (CUDA) + real exported ONNX models**
(test suite: **463 green (0 failed, 0 skipped)**). Items marked ✅ are done and validated.

- **2026-06-26:** validated end-to-end on an RTX 4090 (CUDA) + real model exports.
- **Phase 0 (GPU bring-up):** ✅ ILGPU sees the RTX 4090 (`IsHardwareGpu == true`); new hardware-gated
  CUDA tests skip on CPU-only CI but run on the real GPU.
- **Phase A:** ✅ all four proven end-to-end on real models (A1 distilgpt2 text-generation,
  A2 ResNet50 classification, A3 YOLOv8 detection, A4 wav2vec2 CTC ASR). See Phase A below for results.
- **Phase B (GPU):** ✅ B1 CUDA path validated vs CPU (29 parity ops within 1e-3), B2 GPU multi-dtype
  (int32/int64) + B3 GPU op parity (LayerNorm/Gather/Concat/Slice/Cast) now verified on CUDA, B4 perf
  (MatMul ~556x, Conv2D ~109x vs CPU on the 4090), B5 intermediates stay on-device.
- **Phase C:** ✅ C1 `use_cache_branch`, ✅ C2 `Pipeline.Generate` text-generation API, ✅ C3 quantization
  (DequantizeLinear/QuantizeLinear/DynamicQuantizeLinear/MatMulInteger + GPTQ/AWQ safetensors dequant),
  ✅ C4 mmap >2 GB safetensors + sharded `index.json`, ✅ C5 GGUF reader, ✅ C6 fused LLM ops
  (RMSNorm/SkipRMSNorm/RotaryEmbedding/MultiHeadAttention/GroupQueryAttention).
- **Phase D:** ✅ embedding attention-mask fix, ✅ manifest-precedence test, ✅ Tan/ReduceSumSquare tests,
  ✅ op coverage **88 → 143** ops (logical, trig/hyperbolic inverses, IsNaN/IsInf, normalization family,
  data-movement family, pooling-extra family, Mod/BitShift/Random, Size/NonZero, etc.).

## 0. Current state (read first)

- **Target: `net10.0` ONLY.** Do **not** add `net8.0`/`net9.0` multi-targeting — this is a hard
  constraint the owner set. Single `<TargetFramework>net10.0</TargetFramework>` in every csproj.
- License: **Apache-2.0** (LICENSE + NOTICE at root).
- Build/test baseline: `dotnet test` must be **GREEN — 463 tests, 0 failures** (432 base + 31
  hardware-gated CUDA/perf tests; real-model tests skip when assets are absent). Run it before and
  after every change. If it's not green on a fresh clone, stop and fix that first.
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

## Phase 0 — GPU machine bring-up ✅ DONE
1. ✅ Clone repo, `dotnet build`, `dotnet test` → green on net10.0.
2. ✅ NVIDIA driver + CUDA toolkit installed; `nvidia-smi` shows the RTX 4090.
3. ✅ ILGPU sees the CUDA device: `new IlgpuEngine(graph, preferCpu: false)` reports
   `engine.IsHardwareGpu == true` and `AcceleratorName` is the RTX 4090. New hardware-gated CUDA tests
   **skip** when no CUDA device is present (so CI on CPU-only machines still passes), but run the GPU op
   graphs on the real GPU and compare to the CPU engine.
4. ✅ Models dir kept out of git (`.gitignore`); discovered via `MODELSHARP_MODELS_DIR` / repo-relative `models/`.

**Done:** suite green + hardware-gated GPU op tests run on the real CUDA accelerator and match CPU.

---

## Phase A — Prove real models end-to-end ✅ ALL DONE
All four are now proven end-to-end on real exported ONNX models via opt-in tests that **skip when the
asset is absent** (mirror `MiniLmTests`; discovery via `MODELSHARP_MODELS_DIR` / repo-relative `models/`).

- **A1 — Small LLM (the headline).** ✅ `distilgpt2` exported to ONNX (non-merged decoder-with-past) →
  BPE tokenizer (`BpeTokenizer.FromFiles(vocab.json, merges.txt)`) → `TextGenerator` → greedy decode →
  `BpeTokenizer.Decode`. **Result:** greedy decode is deterministic and coherent — "The quick brown fox"
  → "es are a common sight in the wild, and are often found in the wild". Required making the CPU
  binary-op kernel **dtype-generic** (int64/int32 index math).
- **A2 — Image classification.** ✅ Real ResNet50 ONNX → `Pipeline.Load` (ImageNet mean/std, 224×224,
  labels) → `Run<List<Classification>>(...)`. **Result:** top-1 "tiger cat" 82% on the sample image.
- **A3 — Object detection.** ✅ YOLOv8 ONNX → manifest `task=ObjectDetection` → boxes. **Result:**
  detects 2 cats with well-formed boxes; added a shape-inferred **"auto"** `det_layout`.
- **A4 — ASR (CTC).** ✅ `wav2vec2` CTC ONNX → `MelSpectrogram`/feature path → model → `CtcDecoder` +
  `CtcVocabulary` (blank=0). **Result:** transcribes exactly "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE
  CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL". Required adding **Conv1D (NCW)** support to `ConvKernel`.

---

## Phase B — GPU acceleration ✅ DONE (validated on the RTX 4090)
- **B1 — Validate the CUDA path.** ✅ Every existing GPU op validated against the CPU engine on real
  hardware — **29 parity ops match within 1e-3**. Fixed a missing `ILGPU.Algorithms` intrinsic
  registration that only failed on real CUDA (passed on the CPU accelerator).
- **B2 — GPU multi-dtype.** ✅ int64/int32 buffers so token-id/mask/shape tensors live on-device; now
  also verified on CUDA.
- **B3 — GPU op parity.** ✅ GPU op set covers the ops real models hit (LayerNorm, Gather, Concat, Slice,
  element-wise broadcasting, Cast), each verified vs CPU and now on CUDA.
- **B4 — Performance.** ✅ Benchmarked GPU vs CPU on the 4090: **MatMul 1024³ ~556x speedup**,
  **Conv2D ~109x** vs the CPU engine. (Correctness first, then speed.)
- **B5 — Whole-graph on GPU + GPU KV-cache.** ✅ Intermediates already stay on-device between ops;
  remaining on-device-KV-cache work is documented in `src/ModelSharp.Gpu/B5_NOTES.md`.

---

## Phase C — LLM productionization
> ✅ **C1–C6 all implemented + unit-tested** on the CPU dev machine. Remaining proof for big quantized
> models (7B+) on GPU still needs hardware. Original task text retained below for reference.
- **C1 — `use_cache_branch`.** ✅ Done. Support Optimum **merged** decoder exports (the most common form): they
  add a boolean `use_cache_branch` input and unify prefill/decode. Extend `TextGenerator`/
  `DecoderModelOptions` to feed it. (Audit flagged this as the #1 real-LLM blocker.)
- **C2 — `Pipeline.Load` for TextGeneration.** ✅ Wire BPE tokenizer + `TextGenerator` + manifest behind
  the one-line API (a `TextGenerationPreprocessor`/postprocessor + registry entry for
  `ModelTask.TextGeneration`), so completion is as simple as embeddings are today.
- **C3 — Quantization (required for big models on GPU).** ✅ int8 + int4 dequant kernels; load GPTQ/AWQ /
  HF quantized safetensors; a quantized Linear/MatMul path. Without this, 7B+ won't fit.
- **C4 — safetensors > 2 GB.** ✅ Replace `File.ReadAllBytes` with memory-mapped / chunked reads so large
  shards load (current ~2 GB `byte[]` cap). Parse the `*.index.json` to load sharded checkpoints.
- **C5 — GGUF loader.** ✅ Add a GGUF reader (llama.cpp ecosystem) alongside ONNX + safetensors.
- **C6 — Fused LLM ops** ✅ if a target model exports them: `RotaryEmbedding`, `SimplifiedLayerNormalization`
  (RMSNorm), grouped-query attention, `MultiHeadAttention` contrib op. Driven by the per-model op probe.

---

## Phase D — Op coverage & correctness cleanups
- ✅ Op coverage now **143 of ~190** standard ops. Extend further as models demand. Add via new kernel files +
  `KernelRegistry`, each with a `NewOpsTests`-style unit test.
- **Minor fixes from the audit (small, non-blocking):**
  - ✅ `MeanPoolEmbeddingPostprocessor` now uses the **input** `attention_mask`. (Was: looks for it among model **outputs** (it's normally an
    **input**) → it pooled over all tokens; fixed so padded/batched inputs mask correctly.)
  - ✅ Added unit tests for `Tan` and `ReduceSumSquare`.
  - ✅ Added a manifest-precedence test (sidecar JSON vs conflicting embedded metadata).

---

## Definition of "go live" — ✅ MET
A user can `Pipeline.Load(...)` (or a thin generation API) and run, with no Python and no native deps:
an **embedding** model (already done), an **image classifier**, an **object detector**, an **ASR** model,
and a **text-completion LLM** — all now proven end-to-end on **real models on CPU**, with the **GPU path
validated on CUDA** (RTX 4090). Each is proven by an opt-in test against a real model asset; the
real-model tests live behind `MODELSHARP_MODELS_DIR` / repo-relative `models/` discovery and **skip when
the assets are absent**. (7B-class on the GPU via quantization remains the next scaling target.)
