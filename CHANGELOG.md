# Changelog

All notable changes to ModelSharp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-alpha] - 2026-06-27

### Added
- Whole-graph GPU dispatch: GPU prologue ops let larger contiguous subgraphs
  execute on the device without round-tripping intermediate tensors back to the CPU.
- GGUF quantized-tensor dequantization, covering legacy quant types, k-quants,
  and the IQ4 family.
- ONNX signal-processing ops: `DFT`, `STFT`, and `MelWeightMatrix`, backed by a
  shared FFT implementation.
- ONNX control-flow ops: `If`, `Loop`, and `Scan`, with subgraph execution.
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
