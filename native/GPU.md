# GPU GEMM path (cuBLAS) — UNTESTED

A cuBLAS-backed SGEMM for ModelSharp's GEMM hot path, targeting the
**NVIDIA RTX 4090** (Ada, compute capability `sm_89`).

> **⚠️ UNTESTED — NOT COMPILED, NOT RUN.**
> This code was written on a machine that has the **NVIDIA driver only**
> (`nvidia-smi` works) but **no CUDA toolkit** — `nvcc`, the cuBLAS headers
> (`cublas_v2.h`) and libraries are **not installed**. Therefore:
> - `nvcc` was never invoked (it is not present and would fail).
> - `build/libms_cuda.so` was never produced.
> - Nothing here was verified by execution.
>
> Everything below is written to be correct and buildable, but you **must**
> build and validate it on a CUDA-enabled host before relying on the numbers.

Files:
- `src/cublas_gemm.cu` — the cuBLAS SGEMM wrapper (`extern "C"` entry points).
- `Makefile.cuda` — `nvcc` build into `build/libms_cuda.so`.
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

Exact compile command the Makefile runs (for reference / manual build):

```bash
nvcc -O3 -arch=sm_89 -Xcompiler -fPIC --shared \
     -o build/libms_cuda.so src/cublas_gemm.cu -lcublas -lcudart
```

Adjust `-arch` for other GPUs: `sm_80` (A100), `sm_86` (RTX 30xx),
`sm_90` (H100).

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

## 7. Expected performance (ballpark, UNTESTED)

On an RTX 4090, cuBLAS SGEMM (fp32, large square matrices) typically reaches
**~tens of TFLOP/s** (order ~40–80 TFLOP/s fp32 depending on shape; far more in
TF32 mode). That is far above the current **ILGPU JIT** GEMM in
`src/ModelSharp.Gpu` and the CPU AVX-512 path. Caveats:

- The copy-in/out `ms_cuda_sgemm` pays PCIe transfer + per-call `cudaMalloc`
  each invocation; for small matrices or tight loops that overhead can dominate.
  Use `ms_cuda_upload` + `ms_cuda_sgemm_with_resident_b` for reused weights, or
  extend with a persistent buffer pool / streams for production.
- Real measured numbers require building on a CUDA host — none of the above was
  measured here.

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
