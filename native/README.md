# ModelSharp native kernels

Hand-tuned native compute kernels (C++/AVX-512, optional CUDA) that ModelSharp can
P/Invoke as a fast path, falling back to the pure-managed kernels when the shared
library is absent. This keeps the "zero-native default" promise while offering
native speed when `libms_kernels.so` is built and present.

## Layout

```
native/
  ms_kernels.h        C ABI — the integration contract (read this first)
  Makefile            builds build/libms_kernels.so and the self-tests
  Makefile.cuda       nvcc build for the optional cuBLAS GPU path
  GPU.md              CUDA build/run notes (UNTESTED — needs CUDA toolkit)
  src/
    info.cpp          capability/info exports
    sgemm.cpp         AVX-512 packed fp32 GEMM  (ms_sgemm_f32)
    qgemm.cpp         quantized matmul + VNNI int8 demo  (ms_qgemm_nbits, ms_vnni_i8_*)
    attn.cpp          FlashAttention-style fused attention  (ms_attention_f32)
    conv.cpp          im2col + AVX-512 GEMM conv2d  (ms_conv2d_f32)
    cublas_gemm.cu    cuBLAS SGEMM wrapper (GPU, untested here)
  test/
    *_test.cpp        standalone parity + GFLOP/s self-tests (own main())
```

## Build & test

```bash
cd native
make            # -> build/libms_kernels.so
make run-tests  # build + run every self-test (parity + GFLOP/s)
```

Build flags (`Makefile`): `-O3 -march=native -mfma -fopenmp -funroll-loops -ffast-math`.
`-march=native` targets this host (znver4 = AVX-512 + VNNI). For a portable build replace
with explicit `-mavx512f -mavx512bw -mavx512vl -mavx512vnni -mfma`.

## Head-to-head benchmark (managed vs native)

```bash
cd bench/ModelSharp.Bench
dotnet run -c Release        # OMP_NUM_THREADS=N to vary native threads
```

### Results — Ryzen 7 7800X3D (8 cores / 16 threads, AVX-512 + VNNI), 16 threads

SGEMM, native `ms_sgemm_f32` vs managed `BlockedGemm` (parity max-rel-err ≤ 1e-5):

| shape (M×N×K)   | managed GF/s | native GF/s | speedup |
|-----------------|-------------:|------------:|--------:|
| 256³            |          147 |         813 |  5.52×  |
| 512³            |          307 |         984 |  3.20×  |
| 1024³           |          381 |         983 |  2.58×  |
| 2048³           |          399 |        1102 |  2.76×  |
| 8×4096×4096 (LLM MLP) | 114 |         352 |  3.08×  |
| 1×4096×4096 (GEMV/decode) | 31 |        43 |  1.38×  |

Native fp32 GEMM reaches **~1.1 TFLOP/s (~46 % of the ~2.5 TFLOP/s machine peak,
~89 % of the 8-physical-core ceiling)**. Managed `Vector<T>` peaks ~400 GF/s.

Standalone kernel measurements (see each `test/*_test.cpp`):
- **Attention** (BH=32, S=512, D=64, causal): ~29 GF/s 1T → ~225–248 GF/s 16T.
- **Conv2d** (ResNet 3×3, 256ch, 56²): ~82 GF/s 1T → ~210–371 GF/s 16T.
- **Quantized matmul** (W4A32, M=1, decode): memory-bound, ~3 GF/s 1T → ~13–18 GF/s 16T.
- **VNNI int8 ceiling** (`vpdpbusd`): ~220 GINT8OP/s/core — ~70× the fp-dequant path.
  Realizing this for LLMs needs a W4A8 path (int8 activations consumed directly in
  the VNNI loop) rather than dequantizing weights to fp32 first.

## Using it from the engine (wired in)

The core engine routes its hot fp32 GEMM through the native kernel automatically when
`libms_kernels.so` is present and the CPU has AVX-512, falling back to the managed
`BlockedGemm` otherwise. Seams: `MatMulKernel` (single-matrix MatMul — LLM projections/MLP)
and `BlockedGemm.Multiply` (conv 1×1 / im2col GEMM). See `src/ModelSharp/Native/NativeGemm.cs`.

Controls:
- `MODELSHARP_NATIVE=0` (or `off`/`false`) — disable the native path (force pure managed).
- `MODELSHARP_NATIVE_LIB=/abs/path/libms_kernels.so` — explicit library location.
- The library is also auto-probed next to the assembly and up the tree at `native/build/`.

Verified: all 958 engine tests pass with the native path enabled (results match the managed
engine within ≤1e-5, well inside the ORT tolerance). In-engine `BlockedGemm.Multiply` measured
412 → 1134 GF/s (2.75×) on the 1024³ shape.

## Notes
- All kernels are parity-verified against naive references (and the managed engine
  for SGEMM) within the engine's float tolerance.
- The GPU path (`cublas_gemm.cu`) is written but **UNTESTED**: this host has the
  NVIDIA driver only, no CUDA toolkit. See `GPU.md`.
