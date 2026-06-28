# Changelog

All notable changes to ModelSharp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-06-28

### Added
- **Optional native acceleration layer** (in `native/`, built separately; **not** shipped in the
  NuGet packages, which stay pure-managed). When the shared library is present the engine routes
  its hot paths through it and **falls back to the managed kernels otherwise**, so the
  zero-native-dependency default and cross-platform portability are unchanged.
  - **CPU** (`libms_kernels.so`): hand-tuned AVX-512 packed/blocked fp32 GEMM (used by `MatMul`
    and `Conv`), AVX512-VNNI W4A8 quantized matmul, fused attention, and conv kernels.
    ~2.5–3× over the managed `BlockedGemm` on GEMM-bound work; SGEMM runs at ~92 % of the host's
    FMA roofline. Toggle with `MODELSHARP_NATIVE`.
  - **Portable + safe build**: the library builds on a portable `-mavx2` baseline and selects
    AVX-512 / AVX512-VNNI / AVX2 / scalar paths at **runtime** (`cpu_features.h`), so it never
    executes an unsupported instruction on older x86. `ms_has_avx512` / `ms_has_vnni` are runtime
    checks.
  - **GPU** (`libms_cuda.so`, optional): cuBLAS fast path wired into the ILGPU engine, running
    inside ILGPU's own CUDA context on the resident device buffers with **no extra copies** —
    single and strided-batched `MatMul` (the latter covers decomposed-attention Q·Kᵀ / scores·V),
    plus optional TF32 Tensor Cores. Enable with `MODELSHARP_CUBLAS` / `MODELSHARP_TF32`.
- **Cross-`Run` weight residency** for the GPU engine: graph initializers (weights) stay resident
  on the device across `Run()` calls instead of re-uploading every call
  (`MODELSHARP_RESIDENT_WEIGHTS`, default on), removing the dominant per-call PCIe upload for
  repeated inference.

### Notes
- The native layer accelerates ModelSharp's own hot paths and is opt-in; the managed engine
  remains the portable, zero-dependency default.

## [1.0.1] - 2026-06-27

### Fixed
- Integer-dtype handling in `Identity`, `TopK`, `Clip` (plus `Ceil`/`Floor`/`Round`), and the
  variadic `Min`/`Max`/`Sum`/`Mean` ops. These forced their inputs to float32 and threw
  *"Tensor dtype is Int64; expected Float32"* on the int64 shape/index tensors that real
  detection, layout, table-structure and formula graphs route through them — crashing the
  models mid-graph. They are now dtype-aware (preserving int64/int32), matching the existing
  broadcast-binary / Gather idiom. This unblocks RT-DETR (PP-DocLayoutV3 + table-cell),
  PicoDet (PP-DocLayout), SLANeXt, SLANet_plus and LaTeX-OCR.

### Performance
- New `BlockedGemm`: a register-tiled, multithreaded, pure-managed
  (`System.Numerics.Vector<float>`) float32 GEMM. `MatMul`, `Gemm` and `Conv` (via im2col,
  with a 1×1 fast path) now route through it, reusing each loaded value across an output tile
  instead of recomputing per element. ~2–5× faster CPU inference (the SVTR recognizer and
  PP-LCNet classifier ~4–5×). Pure-managed, no new dependencies; results stay within float
  tolerance (SIMD accumulation order only).

### Added
- `IntegerDtypeKernelTests` covering the int64/int32 paths of the dtype-fixed ops.

### Notes
- CPU output **parity verified against ONNX Runtime on 19 real OCR models** (detection,
  recognition, classification, layout, table-structure, formula) — all match within tolerance.
  Full test suite: 958 passing.

## [0.2.0-alpha] - 2026-06-27

### Verified & hardened
- Next-token logits **bit-verified against ONNX Runtime** on real models — Qwen-0.5B
  (`MatMulNBits`) and Mistral-7B (genai `GroupQueryAttention`) match ORT exactly.
- Real text generation via a Hugging Face `tokenizer.json` (BPE) loader; Whisper-tiny
  transcribes correctly.
- `local_window_size` (sliding-window attention) in GroupQueryAttention; Sequence/Optional
  values now cross If/Loop/Scan subgraph boundaries; ONNX `ImageDecoder` (via ImageSharp).
- Hardening: concurrency/thread-safety tests for the multithreaded kernels, malformed-model
  fuzzing tests, a loader allocation DoS guard, and clean exceptions for missing context values.

### Performance
- Multithreaded (`Parallel.For`) + SIMD (`System.Numerics.Vector<float>`) the hot CPU
  kernels (MatMul/MatMulNBits/MatMulInteger/QLinear, GroupQueryAttention/MHA, RMSNorm/
  LayerNorm, RoPE, Conv), and added native on-device GPU `MatMulNBits` + `GroupQueryAttention`.
  Bit-identical results, much faster: on an RTX 4090 a Mistral-7B INT4 forward pass went from
  ~4m11s to ~16s (~16×) and Qwen-0.5B from ~9s to ~2s. All pure-managed, no new dependencies.

### Added
- **`ModelSharp.Hub`** optional package: download models (+ external-data shards, tokenizer, config)
  from Hugging Face / GGUF / safetensors / direct URL into a local cache and run them in one line
  (`HubPipeline.Load`, `ModelHub.Get`). Resumable downloads, integrity verification, token auth,
  friendly aliases. Pure-managed (HttpClient only); the core stays dependency-free.
- fp16 weight storage: fp16 ONNX initializers load as compact `System.Half` (half the memory),
  widened to float at the compute boundary (no per-kernel changes).
- Real **INT4 LLMs run end-to-end**, including a **7B** (Mistral-7B-Instruct):
  ONNX **external-data** loading (`data_location=EXTERNAL`, memory-mapped) for
  >2 GB models; the `MatMulNBits` block-quant op; and the onnxruntime-genai
  `GroupQueryAttention` variant (packed-QKV, in-op rotary via cos/sin caches,
  seqlens_k). Qwen2.5-0.5B (INT4, decomposed attention) and Mistral-7B (INT4,
  genai GQA) both validated end-to-end.
- 9 more standard/contrib ops (CastLike, Scatter, RNN, AffineGrid, RoiAlign,
  DeformConv, NLLLoss, SoftmaxCrossEntropyLoss, SequenceMap) → ~200 registered.
- Native GPU kernels for many former CPU-fallback ops (Pad/Tile/Clip/reductions/
  activations) and a shared-memory-tiled int8 GEMM.
- Native on-device `MatMulInteger` GEMM (uint8/int8, bit-exact vs the CPU
  engine), so quantized matmuls run on the GPU instead of the CPU fallback;
  the real INT8 GPT-2 decodes whole-graph on CUDA through it.
- fp16 (`FLOAT16`) and bf16 (`BFLOAT16`) ONNX initializer loading (decoded to
  float32 on load), so half-precision models run.
- Exact 400-point DFT (Bluestein) for the Whisper log-mel front-end, making the
  mel features bit-accurate (was a radix-2 512-point zero-pad approximation).
- Whole-graph GPU dispatch: GPU prologue ops let larger contiguous subgraphs
  execute on the device without round-tripping intermediate tensors back to the CPU.
- GGUF quantized-tensor dequantization, covering legacy quant types, k-quants,
  and the IQ4 family.
- ONNX signal-processing ops: `DFT`, `STFT`, and `MelWeightMatrix`, backed by a
  shared FFT implementation.
- ONNX control-flow ops: `If`, `Loop`, and `Scan`, with subgraph execution.
- Full distilgpt2 graph runs end-to-end on the GPU engine (no CPU fallback),
  matching the CPU engine's logits and exact greedy argmax on CUDA.
- GPU engine **runs any CPU-runnable model**: ops without a native GPU kernel
  transparently fall back to the CPU kernel (download → CPU op → re-upload).
- Sequence ops (`SequenceEmpty/Construct/Insert/Erase/At/Length`,
  `SplitToSequence`, `ConcatFromSequence`) and Optional ops (`Optional`,
  `OptionalGetElement`, `OptionalHasElement`) with sequence/optional value types.
- Quantized `QLinear*` ops: `QLinearMatMul`, `QLinearConv`, `QLinearAdd`,
  `QLinearMul`, `ConvInteger`, `QLinearGlobalAveragePool`; plus `FastGelu`,
  `BiasGelu`, `QuickGelu`, `Affine`, `ImageScaler`.
- Encoder-decoder (seq2seq) generation (`Seq2SeqGenerator` + Pipeline
  `Seq2SeqGeneration` task): encoder-once + cross-attention KV-cached decode for
  T5 / BART / MarianMT-style models.
- GGUF `IQ4_NL` / `IQ4_XS` dequantization; `DFT` FFT fast path (radix-2 +
  Bluestein) and opset-20 axis-as-input.
- GGUF grid-codebook IQ dequantization for `IQ1_S`, `IQ1_M`, `IQ2_XXS`,
  `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S` (ggml lattice tables vendored from a
  pinned llama.cpp commit; MIT attribution added to `NOTICE`). Every GGUF
  quantization type now dequantizes.
- Whisper-style ASR: `WhisperFeatureExtractor` (log-mel front-end) + forced-prompt
  decoding through the seq2seq path (`ModelTask.SpeechToTextSeq2Seq`).
- Quantized ONNX model support: the loader parses `uint8`/`int8` initializers and
  `Gather` is dtype-generic, so dynamic-INT8 ONNX models (e.g. quantized GPT-2)
  load and run — validated whole-graph on CUDA with exact greedy-argmax parity
  vs the CPU engine.
- NuGet packaging metadata for the shippable packages (`ModelSharp`,
  `ModelSharp.ImageSharp`, `ModelSharp.Gpu`): authors, description, license
  expression, project/repository URLs, and tags. The core package now embeds the
  README.

### Changed
- Expanded CPU operator coverage and grew the test suite.
- Improved GPU LLM decode path and on-device KV-cache handling.

### Notes
- GPU validation has been exercised on CUDA hardware; full 7B-class LLM-on-GPU
  proof on dedicated hardware is still pending.
- GGUF grid-codebook IQ quants (`IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S`)
  are not yet dequantized — they require vendored ggml lattice tables and license
  attribution to implement bit-exactly, and intentionally throw rather than emit
  unverified weights.

## [0.1.0-alpha]

### Added
- Initial pure-managed, zero-native-dependency model inference for .NET 10.
- ONNX loader, multi-dtype CPU execution engine, tensors, and the high-level
  `Pipeline` API.
- Audio (FFT / log-mel) and text (WordPiece) front ends.
- `ModelSharp.ImageSharp` adapter for image pre/post-processing.
- `ModelSharp.Gpu` ILGPU backend with CUDA / OpenCL kernels and CPU fallback.
- Real-model end-to-end validation on CPU and CUDA.
</content>
</invoke>
