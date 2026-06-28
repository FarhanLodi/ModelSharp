# GPU GEMM path (cuBLAS) — TESTED

A cuBLAS-backed SGEMM for ModelSharp's GEMM hot path, targeting the
**NVIDIA RTX 4090** (Ada, compute capability `sm_89`).

> **✅ TESTED — BUILT, VERIFIED, BENCHMARKED.**
> Built and run on **NVIDIA GeForce RTX 4090** (24 GB, driver 580.159.03) with
> the **CUDA 12.0 toolkit** (`nvcc` V12.0.140, cuBLAS 12). Results:
> - `build/libms_cuda.so` builds cleanly with `make -f Makefile.cuda`.
> - Correctness verified vs a CPU triple-loop reference for 64³, 256×384×500,
>   and 1024³ (relative error ≤ 4.1e-7, threshold 1e-3) — **all PASS**.
> - Benchmarked for 1024³ / 2048³ / 4096³, copy-inclusive and compute-only —
>   see the table in §7. Peak compute-only ≈ **60 TFLOP/s** (fp32).
>
> Standalone harness: `test/cublas_test.cu`.

Files:
- `src/cublas_gemm.cu` — the cuBLAS SGEMM wrapper (`extern "C"` entry points).
- `Makefile.cuda` — `nvcc` build into `build/libms_cuda.so`.
- `test/cublas_test.cu` — standalone correctness + benchmark harness.
- this file.

---

## 1. Install the CUDA toolkit (to enable building)

The driver is already present (`nvidia-smi` reports `NVIDIA GeForce RTX 4090`,
driver `580.x`). You only need the **toolkit** (`nvcc` + cuBLAS dev libs).

### Option A — Ubuntu/Debian apt (simplest)

```bash
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit
```

This pulls a distro-packaged `nvcc` and cuBLAS. It may lag the latest CUDA, but
is fine for `sm_89`. Verify: `nvcc --version`.

### Option B — NVIDIA CUDA repo (newer toolkit, recommended for Ada)

```bash
# Ubuntu 22.04 example — adjust distro string for your release.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# Toolkit only (driver already installed). cuBLAS is part of this metapackage.
sudo apt-get install -y cuda-toolkit-12-6
# Make nvcc visible:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Confirm: `nvcc --version` and `ls /usr/local/cuda/include/cublas_v2.h`.

---

## 2. Verify the GPU is visible (driver-only check)

This works today, even without the toolkit:

```bash
nvidia-smi
# Expect: NVIDIA GeForce RTX 4090, a driver version, CUDA "runtime" capability.
```

The `CUDA Version` shown by `nvidia-smi` is the **max** the driver supports, not
an installed toolkit — you still need step 1 to compile.

---

## 3. Build

```bash
cd native
make -f Makefile.cuda
# -> build/libms_cuda.so
```

Exact compile command the Makefile ran (verified working with CUDA 12.0 on the
RTX 4090; `nvcc 12.0` accepts `-arch=sm_89`):

```bash
nvcc -O3 -arch=sm_89 -Xcompiler -fPIC --shared \
     -o build/libms_cuda.so src/cublas_gemm.cu -lcublas -lcudart
```

Adjust `-arch` for other GPUs: `sm_80` (A100), `sm_86` (RTX 30xx),
`sm_90` (H100).

### Build & run the test/benchmark

The standalone harness (`test/cublas_test.cu`) has its own `main()` and links
the wrapper directly. Verified working command:

```bash
nvcc -O3 -arch=sm_89 native/test/cublas_test.cu native/src/cublas_gemm.cu \
     -lcublas -lcudart -o /tmp/cublas_test
/tmp/cublas_test
```

It prints `ms_cuda_available()` / `ms_cuda_device_name()`, runs correctness
checks (rel err vs CPU reference, threshold 1e-3), then benchmarks
copy-inclusive and compute-only GEMM (median of several runs after warmup).

---

## 4. The column-major transpose trick (why no explicit transpose)

ModelSharp is **row-major**; cuBLAS is **column-major** (Fortran legacy). We
want, all row-major:

```
C[M,N] = A[M,K] * B[K,N]
```

A row-major buffer `X[r,c]` is bit-identical to the column-major matrix `Xᵀ`.
So if we pass our row-major buffers to cuBLAS unchanged, it sees `Aᵀ` and `Bᵀ`.
Using `Cᵀ = (A·B)ᵀ = Bᵀ·Aᵀ`, and noting that "`Cᵀ` column-major" == "`C`
row-major", we ask cuBLAS to compute (column-major) `C_cm[N,M] = B·A` with both
operands `OP_N`, swapping operand order and the M/N extents:

```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,        // m,n,k in column-major terms
            &alpha,
            dB, N,          // first operand = our B,  lda = N
            dA, K,          // second operand = our A, ldb = K
            &beta,
            dC, N);         // result = our C,         ldc = N
```

Leading dims: a row-major `X[rows,cols]` read column-major has `ld = cols`,
hence `lda(B)=N`, `ldb(A)=K`, `ldc(C)=N`. The result bytes in `dC` are exactly
the desired row-major `C[M,N]` — **no transpose kernels, no extra copies**.

---

## 5. Entry points (`extern "C"`, for C# P/Invoke)

```c
int         ms_cuda_available(void);              // 1 if cudaGetDeviceCount>0
const char* ms_cuda_device_name(void);            // e.g. "NVIDIA GeForce RTX 4090"
void        ms_cuda_sgemm(const float* A, const float* B, float* C,
                          int M, int N, int K);    // row-major C=A*B (copy in/out)

// Optional resident-B helpers (avoid re-uploading a reused weight matrix):
void*       ms_cuda_upload(const float* host, int rows, int cols);
void        ms_cuda_free(void* dev);
void        ms_cuda_sgemm_with_resident_b(const float* A, void* dB, float* C,
                                          int M, int N, int K);
void        ms_cuda_shutdown(void);               // release cached handle
```

The cuBLAS handle is created once and cached. The simple `ms_cuda_sgemm`
allocates device buffers, copies A/B host→device, runs the GEMM, copies C
device→host, and frees — the must-have path. CUDA and cuBLAS status codes are
checked; failures print to `stderr` and clean up.

---

## 6. C# P/Invoke (mirror `bench/ModelSharp.Bench/NativeKernels.cs`)

Use the same `NativeLibrary.SetDllImportResolver` probing pattern as the
existing `NativeKernels` class, just pointing at `libms_cuda.so`:

```csharp
using System;
using System.Runtime.InteropServices;

namespace ModelSharp.Bench;

internal static class CudaKernels
{
    private const string Lib = "ms_cuda";

    static CudaKernels()
    {
        NativeLibrary.SetDllImportResolver(typeof(CudaKernels).Assembly,
            (name, asm, path) =>
        {
            if (name != Lib) return IntPtr.Zero;
            foreach (var cand in new[]
            {
                "native/build/libms_cuda.so",
                "../native/build/libms_cuda.so",
                "../../native/build/libms_cuda.so",
                "../../../native/build/libms_cuda.so",
                "../../../../native/build/libms_cuda.so",
            })
            {
                if (NativeLibrary.TryLoad(System.IO.Path.GetFullPath(cand), out var h))
                    return h;
            }
            return NativeLibrary.TryLoad("libms_cuda.so", out var g) ? g : IntPtr.Zero;
        });
    }

    // false (no throw) when the .so is absent or no GPU is present.
    public static bool Available
    {
        get { try { return ms_cuda_available() != 0; } catch { return false; } }
    }

    public static string DeviceName() =>
        Marshal.PtrToStringAnsi(ms_cuda_device_name()) ?? "?";

    [DllImport(Lib, EntryPoint = "ms_cuda_available")]
    private static extern int ms_cuda_available();

    [DllImport(Lib, EntryPoint = "ms_cuda_device_name")]
    private static extern IntPtr ms_cuda_device_name();

    [DllImport(Lib, EntryPoint = "ms_cuda_sgemm")]
    public static extern unsafe void Sgemm(float* a, float* b, float* c,
                                           int m, int n, int k);
}
```

Note the layout/order matches `ms_cuda_sgemm` exactly: row-major `C[M,N] =
A[M,K] * B[K,N]`, the same `(a, b, c, m, n, k)` argument order as the existing
CPU `ms_sgemm_f32` binding — so it can drop into the same call sites.

---

## 7. Measured performance (RTX 4090, CUDA 12.0, fp32)

Measured on **NVIDIA GeForce RTX 4090** (driver 580.159.03) with the CUDA 12.0
toolkit, via `test/cublas_test.cu` (median of 7–9 runs after warmup).

Square `N×N×N` SGEMM, FLOPs = `2·N³`:

| Shape   | Copy-inclusive (H2D+GEMM+D2H) | Compute-only (A/B/C resident, kernel only) |
|---------|-------------------------------|--------------------------------------------|
| 1024³   | ~1.8–1.9 TFLOP/s (~1.1 ms)    | ~38 TFLOP/s (~0.06 ms)                      |
| 2048³   | ~2.5–4.6 TFLOP/s (~3.8–6.9 ms)| ~60 TFLOP/s (~0.29 ms)  ← peak             |
| 4096³   | ~7.6–8.1 TFLOP/s (~17 ms)     | ~50–56 TFLOP/s (~2.5 ms)                    |

Correctness: relative error vs CPU reference ≤ **4.1e-7** for all tested shapes
(threshold 1e-3) — all PASS.

Notes:

- The **compute-only** numbers are the real cuBLAS kernel ceiling (~60 TFLOP/s
  fp32 peak at 2048³, settling to ~50–56 TFLOP/s at 4096³ — typical cuBLAS fp32
  on Ada; TF32 mode would be far higher again).
- The **copy-inclusive** `ms_cuda_sgemm` pays PCIe transfer **plus a per-call
  `cudaMalloc`/`cudaFree`** of all three buffers every invocation, which
  dominates and makes these numbers noisy run-to-run (e.g. 2048³ varied
  3.8–6.9 ms). For reused weights use `ms_cuda_upload` +
  `ms_cuda_sgemm_with_resident_b`, or add a persistent buffer pool / streams to
  approach the compute-only ceiling.
- **vs the CPU native kernel** (`libms_kernels.so` `ms_sgemm_f32`, ~1.1
  TFLOP/s): the compute-only cuBLAS path is **~35–55× faster** (≈54× at the
  2048³ peak); even the naive copy-inclusive path is **~1.6–7×** faster, with
  the gap widening as the matrix grows and PCIe/alloc overhead amortizes.
- **vs the ILGPU path** (`src/ModelSharp.Gpu`, which JIT-compiles generic C#
  kernels at runtime): cuBLAS is vendor-tuned hand-optimized SGEMM and is
  expected to be far faster than an ILGPU-generated GEMM kernel for raw
  throughput on NVIDIA hardware. (ILGPU was not re-benchmarked here; keep it as
  the cross-vendor fallback per §8.)

---

## 8. Relationship to the ILGPU path

This **competes with / replaces the ILGPU GEMM** in `src/ModelSharp.Gpu`
(`IlgpuEngine.cs` / `GpuKernels.cs`) for the GEMM hot path. ILGPU JIT-compiles
generic kernels at runtime and is portable, but for raw SGEMM throughput on
NVIDIA hardware cuBLAS is vendor-tuned and substantially faster. Suggested
integration: gate on `ms_cuda_available()`; when true and the matrix is large
enough to amortize transfer, route GEMM through `ms_cuda_sgemm` (or the
resident-B variant), otherwise fall back to ILGPU / the CPU kernels. Keep ILGPU
as the cross-vendor fallback.
