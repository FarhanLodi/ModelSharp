# Changelog

All notable changes to ModelSharp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-alpha] - 2026-06-27

### Added
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
