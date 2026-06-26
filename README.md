<div align="center">
  <img src="modelsharp_logo.png" alt="ModelSharp Logo" width="120" height="120" />
</div>

# ModelSharp

> Universal, manifest-driven, **pure-managed** model inference for .NET.

ModelSharp aims to be to model inference what ImageSharp is to imaging: a single,
**zero-native-dependency**, cross-platform library where *any* model — vision, text,
audio — "just runs" on CPU today and GPU as it matures, with no Python and no per-model glue.

## Why

ONNX Runtime gives you `tensor in → tensor out`, but every model still needs its own
pre/post-processing, and the native runtime can't run everywhere. ModelSharp closes both gaps:

1. **Self-describing models.** A small *manifest* (embedded ONNX metadata, a sidecar JSON, or a
   built-in registry) describes how to feed and decode a model, so one `Pipeline` API runs any model.
2. **Pure-managed engine.** Managed kernels (SIMD-friendly, GPU via ILGPU) mean one build runs on
   Windows/Linux/macOS, x64 and ARM64, with no native DLLs to ship. Even the ONNX parser is a
   hand-rolled protobuf reader — no Google.Protobuf dependency.

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
                    +-- IlgpuEngine       (ModelSharp.Gpu)  -- C# kernels -> CUDA/OpenCL/CPU
```

The `IExecutionEngine` seam is the key decision: the public API and all pre/post-processing are
fixed, while engines and kernel coverage grow underneath without breaking callers.

Packaging is split by **dependency**, not by feature — so the common case is a single download:

| Package | Contains | External deps |
|---------|----------|---------------|
| **`ModelSharp`** | Everything dependency-free: ONNX loader (hand-rolled protobuf), multi-dtype CPU engine + 46-op kernel registry, tensors, manifest resolver + auto-wired pipeline (`Pipeline.Load`), the FFT/log-mel audio front end + CTC decoder, and the WordPiece text tokenizer | **none** |
| `ModelSharp.ImageSharp` *(optional)* | Image → tensor + classification decoding | SixLabors.ImageSharp (v3.x) |
| `ModelSharp.Gpu` *(optional)* | ILGPU backend (C# kernels → CUDA/OpenCL, CPU fallback) | ILGPU |

```bash
dotnet add package ModelSharp              # load + run any ONNX model on CPU, plus audio — nothing else to pull in
dotnet add package ModelSharp.ImageSharp   # only if you need image decoding
dotnet add package ModelSharp.Gpu          # only if you need GPU
```

Inside the core assembly the areas keep their own namespaces (`ModelSharp.Onnx`, `ModelSharp.Cpu`,
`ModelSharp.Audio`) — one DLL, clean separation.

## Status

**Verified end to end:** the bundled MNIST model is loaded by our own ONNX reader and run through
the managed kernels, reproducing ONNX Runtime's reference output to within 1e-2. And the real
pretrained **all-MiniLM-L6-v2** sentence-transformer runs end to end (tokenize → 6 transformer
layers → mean-pooled embedding), producing semantically correct 384-d embeddings — cosine **0.70**
for paraphrases vs **−0.05** for unrelated text.

The **manifest → pipeline path is now wired end to end**: `Pipeline.Load("model.onnx")` resolves a
manifest (sidecar JSON → embedded ONNX metadata → built-in registry), builds the engine, and picks
the pre/post-processors — so `Pipeline.Load("all-MiniLM-L6-v2.onnx").Run<float[]>("some text")`
returns a normalized embedding with no per-model glue. The engine is **multi-dtype** (float32 /
int64 / int32 / bool) — token ids, masks, and shape tensors flow through as their real types.

### Operator coverage (46)

- **Arithmetic** (broadcasting): Add, Sub, Mul, Div, Pow
- **Activations**: Relu, Sigmoid, Tanh, Exp, Log, Sqrt, Abs, Neg, Erf, Gelu, Identity, LeakyRelu, Clip, Softmax
- **NN layers**: Conv (auto_pad/dilations/group/bias), MaxPool, GlobalAveragePool, BatchNormalization, LayerNormalization
- **Recurrent**: LSTM (peepholes, clip, input_forget, sequence_lens), GRU (clip, sequence_lens) — both forward/reverse/bidirectional, optional bias + initial state
- **Linear**: MatMul (n-D batched, NumPy semantics), Gemm
- **Reduction**: ReduceMean (axes, keepdims)
- **Logical** (bool out, broadcasting): Where, Equal, Less, Greater
- **Shape/data**: Reshape, Flatten, Concat, Transpose, Gather, Unsqueeze, Squeeze, Cast (typed), Shape, Constant, ConstantOfShape, Slice, Expand

### Phase progress

| Phase | State |
|-------|-------|
| 1. CNN core | ✅ verified (MNIST vs ONNX-Runtime reference) |
| 2. Sequence | ✅ LSTM + GRU verified (vs ONNX reference) |
| 3. Transformer | ✅ **real pretrained all-MiniLM-L6-v2 runs end-to-end** — semantic embeddings verified (paraphrase 0.70 vs unrelated −0.05) |
| 4. Audio | 🟡 FFT + log-mel front end **and** CTC decoder (greedy + prefix-beam) in & unit-tested; end-to-end with a pretrained acoustic model next |
| 5. GPU | 🟡 ILGPU engine now covers broadcasting elementwise + **MatMul + Conv2D**, numerically verified against the CPU engine through the same seam; real-hardware tuning + multi-dtype next |

A separate strand — the **manifest-driven `Pipeline.Load`** wiring — is now in and tested: manifest
resolution (sidecar JSON / embedded ONNX metadata / built-in registry), a processor registry, and a
built-in text-embedding pre/post path, so text-embedding models run end to end from a single call.
The BERT **WordPiece tokenizer** now does full basic-tokenization (punctuation + CJK splitting,
accent stripping, NFC normalization).

## Build & test

```bash
dotnet build
dotnet test
```

Targets `net10.0`.

## Notes

- **ImageSharp** is pinned to **v3.x**: v4 added a build-time commercial license-key gate that
  conflicts with the "works the moment you download it" goal.
- Phases 2–5 are seeded with **real, tested foundations**, not finished — full sequence/transformer/
  audio/GPU support is the multi-quarter roadmap.

## License

[Apache License 2.0](LICENSE) — permissive, with an explicit patent grant (the prevailing
license across the ML inference ecosystem: ONNX, ONNX Runtime, PyTorch, TensorFlow).
