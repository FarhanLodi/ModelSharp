# GPU GEMM path (cuBLAS) — TESTED

A cuBLAS-backed SGEMM for ModelSharp's GEMM hot path, targeting the
**NVIDIA RTX 4090** (Ada, compute capability `sm_89`).

> **✅ TESTED — BUILT, VERIFIED, BENCHMARKED.**
> Built and run on **NVIDIA GeForce RTX 4090** (24 GB, driver 580.159.03) with
> the **CUDA 12.0 toolkit** (`nvcc` V12.0.140, cuBLAS 12). Results:
> - `build/libms_cuda.so` builds cleanly with `make -f Makefile.cuda`.
> - Correctness verified vs a CPU triple-loop reference for 64³, 256×384×500,
>   and 1024³ (relative error ≤ 4.1e-7, threshold 1e-3) — **all PASS**.
> - Benchmarked for 1024³ / 2048³ / 4096³ (copy-inclusive + compute-only) and
>   for the device-resident DECODE / PREFILL / chained paths — see §7. Peak
>   ≈ **61 TFLOP/s** (fp32); resident prefill **55.7 TFLOP/s**; per-token decode
>   **0.070 ms/GEMM** (≈56× faster than copy-in/out).
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
     -o build/libms_cuda.so src/cublas_gemm.cu -lcublas -lcudart -lcuda
```

`-lcuda` (the CUDA **driver** library) is required for the externally-provided
context API (`cuCtxPushCurrent` / `cuCtxPopCurrent`); see §6b. Confirm it is
present with `ldconfig -p | grep libcuda` (it ships with the NVIDIA driver).

Adjust `-arch` for other GPUs: `sm_80` (A100), `sm_86` (RTX 30xx),
`sm_90` (H100).

### Build & run the test/benchmark

The standalone harness (`test/cublas_test.cu`) has its own `main()` and links
the wrapper directly. Verified working command:

```bash
nvcc -O3 -arch=sm_89 native/test/cublas_test.cu native/src/cublas_gemm.cu \
     -lcublas -lcudart -lcuda -o /tmp/cublas_ctx_test
/tmp/cublas_ctx_test
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

// Resident-B helper (avoid re-uploading a reused weight matrix; A in / C out):
void*       ms_cuda_upload(const float* host, int rows, int cols); // -> dev buf
void        ms_cuda_free(void* dev);
void        ms_cuda_sgemm_with_resident_b(const float* A, void* dB, float* C,
                                          int M, int N, int K);
void        ms_cuda_shutdown(void);               // release handle + stream

// Fully device-resident API (the LLM weight-reuse path — NO implicit copies):
void*       ms_cuda_alloc(size_t n);              // uninit device buffer of n floats
int         ms_cuda_upload_into(void* dev, const float* host, size_t n); // refresh in place
int         ms_cuda_download(void* dev, float* host, size_t n);          // dev -> host
int         ms_cuda_sgemm_dev(void* dA, void* dB, void* dC,
                              int M, int N, int K); // C=A*B, all device ptrs, enqueue only
int         ms_cuda_sync(void);                   // wait for the stream

// Externally-provided CUDA DRIVER context API (the ILGPU-interop path — runs on
// the caller's own driver context + stream + device pointers, zero copies):
int         ms_cuda_sgemm_ctx(void* cuContext, void* cuStream,
                              void* dA, void* dB, void* dC,
                              int M, int N, int K); // C=A*B in cuContext on cuStream
int         ms_cuda_bind_context(void* cuContext);   // get-or-create cached handle for ctx
void        ms_cuda_release_context(void* cuContext); // destroy cached handle for ctx
int         ms_cuda_sync_stream(void* cuStream);     // sync stream (NULL = device sync)

// Strided-batched context-aware GEMM (cublasSgemmStridedBatched). For each
// b in [0,batch): C[b]=A[b]*B[b], where the b-th matrices start at
// dA+b*strideA, dB+b*strideB, dC+b*strideC (strides in ELEMENTS). strideA
// and/or strideB may be 0 to broadcast a shared matrix across all batches.
int         ms_cuda_sgemm_strided_batched_ctx(
                void* cuContext, void* cuStream,
                void* dA, void* dB, void* dC,
                int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                int batch);
```

The cuBLAS handle is created once and cached, and a dedicated non-blocking CUDA
stream is bound to it; all copies and GEMMs run on that stream and a single
`ms_cuda_sync()` waits for a whole batch. The simple `ms_cuda_sgemm` allocates
device buffers, copies A/B host→device, runs the GEMM, copies C device→host,
and frees — the must-have path.

**Device-resident pattern (what LLM inference should use):** upload each weight
**once** with `ms_cuda_upload`, allocate activation/output buffers once with
`ms_cuda_alloc`, then call `ms_cuda_sgemm_dev` (raw device pointers, no
`cudaMalloc`/`cudaFree`, no PCIe copies — it just enqueues the kernel on the
stream). Chain many GEMMs/layers this way and `ms_cuda_sync()` / `ms_cuda_download`
only at the end. The same column-major `Cᵀ=Bᵀ·Aᵀ` trick is used, so `dA` is the
row-major activation `[M,K]`, `dB` the weight `[K,N]`, `dC` the output `[M,N]`.

CUDA and cuBLAS status codes are checked everywhere; failures print to `stderr`,
clean up, and return non-zero (device-resident calls) or leave `C` untouched.
`ms_cuda_shutdown` destroys both the handle and the stream (no leaks).

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

## 6b. Externally-provided CUDA driver context (ILGPU interop)

**Why this exists.** The default entry points (`ms_cuda_sgemm`,
`ms_cuda_sgemm_dev`, …) implicitly use the CUDA **runtime primary context** and
cache a single `cublasHandle_t` created in it. ILGPU, however, owns its **own
CUDA driver-API context** and hands out device pointers that are valid only in
THAT context. A `cublasHandle_t` and a device pointer are bound to whichever CUDA
context is current when they are created/used. Calling the runtime-primary-context
handle on ILGPU's pointers dereferenced them in the wrong address space and
produced a **deterministic wrong region** (the bug that motivated this API;
1024³ failed reproducibly). Note it is not a hard fault — the numbers are simply
wrong — so you must route through the context-aware path, not rely on a crash.

**The fix — context-aware entry points.** These run cuBLAS *inside the caller's
driver context*, on the caller's stream, against the caller's device pointers,
with **zero cross-context copies**:

```c
int  ms_cuda_sgemm_ctx(void* cuContext, void* cuStream,
                       void* dA, void* dB, void* dC, int M, int N, int K);
int  ms_cuda_bind_context(void* cuContext);     // optional: pre-create handle
void ms_cuda_release_context(void* cuContext);  // destroy this ctx's handle
int  ms_cuda_sync_stream(void* cuStream);       // NULL => device sync
```

`ms_cuda_sgemm_ctx` computes row-major `C[M,N]=A[M,K]*B[K,N]` (same
`Cᵀ=Bᵀ·Aᵀ` column-major trick) and:

1. `cuCtxPushCurrent(cuContext)` to make the caller's context current.
2. Looks up (or lazily `cublasCreate`s, **while that context is current**) the
   handle from a per-context cache: `std::map<CUcontext, cublasHandle_t>`,
   mutex-guarded so it is thread-safe.
3. `cublasSetStream(handle, cuStream)` — a driver `CUstream` is interchangeable
   with a runtime `cudaStream_t`; `NULL` = that context's default stream.
4. Runs `cublasSgemm`, then **always** `cuCtxPopCurrent` — including on every
   error path (an RAII guard guarantees the push/pop are balanced).

The GEMM is **enqueued only**; call `ms_cuda_sync_stream(cuStream)` when you need
the result. `ms_cuda_bind_context` lets you pay the `cublasCreate` cost once up
front; `ms_cuda_release_context` destroys that context's handle (call it before
you tear down the ILGPU context). All existing runtime-API entry points are
unchanged and keep working independently of this cache.

**Verified:** correctness vs CPU reference (rel err ≤ 8.1e-7, threshold 1e-3)
for 64³, 256×384×500, **1024³** (the previously-failing shape) and 2048³, on a
driver context that the wrapper did **not** create and that was **not current on
entry** — on both a user `cuStreamCreate`'d stream and the NULL default stream.
External-context compute-only throughput matches the native path: ~38 TFLOP/s at
1024³, ~60 TFLOP/s at 2048³, ~56 TFLOP/s at 4096³.

### How the C# side must call it

Pass ILGPU's own context, stream, and buffer device pointers straight through —
no marshalling of data, only handles:

```csharp
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

// accelerator is an ILGPU CudaAccelerator; a, b, c are MemoryBuffer1D<float,...>
// already allocated on it (so their pointers live in ILGPU's driver context).

var cuAccel = (CudaAccelerator)accelerator;

// ILGPU's underlying CUDA driver context (CUcontext) and a stream (CUstream).
IntPtr cuContext = cuAccel.NativePtr;                       // the CUcontext
var    cuStream  = (CudaStream)accelerator.DefaultStream;   // or a CudaStream you made
IntPtr cuStreamPtr = cuStream.StreamPtr;                    // the CUstream (IntPtr.Zero => default)

// Raw device pointers of the ILGPU buffers (valid in cuContext).
IntPtr dA = a.View.LoadEffectiveAddressAsPtr();  // [M,K] activations, row-major
IntPtr dB = b.View.LoadEffectiveAddressAsPtr();  // [K,N] weights,     row-major
IntPtr dC = c.View.LoadEffectiveAddressAsPtr();  // [M,N] output,      row-major

// Optional: create the per-context cuBLAS handle once up front.
CudaKernels.BindContext(cuContext);

int rc = CudaKernels.SgemmCtx(cuContext, cuStreamPtr, dA, dB, dC, M, N, K);
if (rc != 0) throw new Exception($"ms_cuda_sgemm_ctx failed: {rc}");

// Wait for the result on that stream before reading c back through ILGPU.
CudaKernels.SyncStream(cuStreamPtr);

// On shutdown, before disposing the ILGPU accelerator/context:
CudaKernels.ReleaseContext(cuContext);
```

Matching P/Invoke declarations (add to the `CudaKernels` class from §6):

```csharp
[DllImport(Lib, EntryPoint = "ms_cuda_sgemm_ctx")]
public static extern int SgemmCtx(IntPtr cuContext, IntPtr cuStream,
                                  IntPtr dA, IntPtr dB, IntPtr dC,
                                  int m, int n, int k);

[DllImport(Lib, EntryPoint = "ms_cuda_bind_context")]
public static extern int BindContext(IntPtr cuContext);

[DllImport(Lib, EntryPoint = "ms_cuda_release_context")]
public static extern void ReleaseContext(IntPtr cuContext);

[DllImport(Lib, EntryPoint = "ms_cuda_sync_stream")]
public static extern int SyncStream(IntPtr cuStream);
```

Notes:
- Exact accessor names vary slightly by ILGPU version; the load-bearing point is
  that `cuContext` is ILGPU's `CUcontext`, `cuStream` is ILGPU's `CUstream`, and
  `dA/dB/dC` are device pointers allocated *by ILGPU* (so they live in that
  context). Do **not** mix these with the runtime-context `ms_cuda_*` pointers.
- The library links `-lcuda` (the CUDA driver) in addition to `-lcublas
  -lcudart` for the `cuCtx*` calls — see the Makefile / §3.

---

## 6c. Strided-batched context-aware GEMM (attention / batched matmul)

`ms_cuda_sgemm_strided_batched_ctx` is the batched analogue of
`ms_cuda_sgemm_ctx`: a single `cublasSgemmStridedBatched` launch computes, for
every `b` in `[0, batch)`, the row-major

```
C[b][M,N] = A[b][M,K] * B[b][K,N]   (alpha = 1, beta = 0)
```

where the `b`-th matrices begin at `dA + b*strideA`, `dB + b*strideB`,
`dC + b*strideC`. **Strides are in ELEMENTS (floats), not bytes.** It reuses the
exact same infrastructure as `ms_cuda_sgemm_ctx` — the `CtxGuard` push/pop and
the per-context cached `cublasHandle_t` — and the same column-major
`Cᵀ = Bᵀ·Aᵀ` transpose trick, lifted to the batched call:

```c
cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K, &alpha,
    dB, N, strideB,     // first operand  = our B, lda = N, stride pairs with B
    dA, K, strideA,     // second operand = our A, ldb = K, stride pairs with A
    &beta,
    dC, N, strideC,     // output         = our C, ldc = N, stride pairs with C
    batch);
```

Each stride pairs with the operand it belongs to; cuBLAS adds `stride * b` to the
base pointer independently per operand. Leading dims are the row-major
`cols`: `lda(B)=N`, `ldb(A)=K`, `ldc(C)=N`. The GEMM is **enqueued only** on
`cuStream` (no sync); call `ms_cuda_sync_stream(cuStream)` — or, for a
non-blocking user stream feeding a large batched launch, a context/device sync —
before reading results.

**Broadcast / shared weight.** Passing `strideB = 0` reuses one `[K,N]` `B`
matrix for every batch (e.g. a shared projection weight applied to a batch of
activations); `strideA = 0` similarly broadcasts a single `A`. `strideC` must be
non-zero so each batch writes a distinct output.

**Use cases.** Multi-head attention: the per-head `scores = Q·Kᵀ`-shaped batch
(`batch = B*H`, `[M,K]·[K,N]`) and the `context = scores·V` second matmul are
each a single strided-batched call; a per-token decode with a shared weight is
the `strideB = 0` broadcast case.

### How the C# side must call it

Identical to the `ms_cuda_sgemm_ctx` pattern (§6b) — pass ILGPU's `CUcontext`,
`CUstream`, and the batched buffer device pointers straight through, plus the
three element strides as `long`:

```csharp
// dA/dB/dC are device pointers to the full batched buffers (valid in cuContext).
// strides are in ELEMENTS (floats). strideB = 0 broadcasts a shared B weight.
long strideA = (long)M * K;
long strideB = (long)K * N;   // or 0 to share one B across all batches
long strideC = (long)M * N;

int rc = CudaKernels.SgemmStridedBatchedCtx(
    cuContext, cuStreamPtr, dA, dB, dC,
    M, N, K, strideA, strideB, strideC, batch);
if (rc != 0) throw new Exception($"ms_cuda_sgemm_strided_batched_ctx failed: {rc}");

CudaKernels.SyncStream(cuStreamPtr);   // wait before reading C back through ILGPU
```

Matching P/Invoke declaration (add to the `CudaKernels` class from §6):

```csharp
[DllImport(Lib, EntryPoint = "ms_cuda_sgemm_strided_batched_ctx")]
public static extern int SgemmStridedBatchedCtx(
    IntPtr cuContext, IntPtr cuStream,
    IntPtr dA, IntPtr dB, IntPtr dC,
    int m, int n, int k,
    long strideA, long strideB, long strideC,
    int batch);
```

`long` on the C# side marshals to the `long long` strides cuBLAS expects. All
existing entry points are unchanged.

---

## 7. Measured performance (RTX 4090, CUDA 12.0, fp32)

Measured on **NVIDIA GeForce RTX 4090** (driver 580.159.03) with the CUDA 12.0
toolkit, via `test/cublas_test.cu` (median of 7–9 runs after warmup).

Square `N×N×N` SGEMM, FLOPs = `2·N³`:

| Shape   | Copy-inclusive (H2D+GEMM+D2H) | Compute-only (A/B/C resident, kernel only) |
|---------|-------------------------------|--------------------------------------------|
| 1024³   | 2.0 TFLOP/s (1.07 ms)         | 38 TFLOP/s (0.06 ms)                        |
| 2048³   | 4.6 TFLOP/s (3.71 ms)         | 60 TFLOP/s (0.29 ms)  ← peak               |
| 4096³   | 7.9 TFLOP/s (17.4 ms)         | 56 TFLOP/s (2.46 ms)                        |

### Device-resident path (the numbers that matter for LLM inference)

Weight `[K,N]=[4096,4096]` uploaded **once**, activations + output kept on
device, only the final result downloaded — via the new `ms_cuda_alloc` /
`ms_cuda_upload` / `ms_cuda_sgemm_dev` / `ms_cuda_sync` / `ms_cuda_download` API.

| Scenario                         | Shape (M×K×N)   | Latency / GEMM | Throughput   |
|----------------------------------|-----------------|----------------|--------------|
| **DECODE** (greedy, M=1)         | 1×4096×4096     | **0.070 ms**   | 0.48 TFLOP/s |
| **DECODE** (small batch, M=8)    | 8×4096×4096     | **0.045 ms**   | 5.95 TFLOP/s |
| **PREFILL** (M=512)              | 512×4096×4096   | **0.309 ms**   | 55.7 TFLOP/s |

Fully-device chained GEMM (no H2D/D2H at all — the copy-free ceiling):

| Shape   | chain | Latency / GEMM | Throughput      |
|---------|-------|----------------|-----------------|
| 1024³   | 16    | 0.066 ms       | 32.3 TFLOP/s    |
| 2048³   | 8     | 0.282 ms       | 61.0 TFLOP/s    |
| 4096³   | 8     | 2.831 ms       | 48.6 TFLOP/s    |

Correctness: relative error vs CPU reference ≤ **4.1e-7** for all tested shapes,
including the new device-pointer path (`ms_cuda_sgemm_dev`) at M=1/8/512
(threshold 1e-3) — **all PASS**.

### Realistic LLM weight-reuse speedup

For per-token decode the copy-in/out path re-uploads the 64 MB weight on every
call (~3.96 ms/GEMM, PCIe-bound); the resident path runs at **0.070 ms/GEMM**:

> **≈ 56× faster per decode step** with resident weights vs. copy-in/out, and
> **≈ 64× faster than the ~1.1 TFLOP/s CPU kernel** at the M=512 prefill GEMM
> (55.7 vs ~1.1 TFLOP/s). Decode is latency- (kernel-launch-) bound by the tiny
> M=1 GEMM, not bandwidth, so the win there is the eliminated PCIe transfer.

Notes:

- The **compute-only** / **chain-dev** numbers are the real cuBLAS kernel
  ceiling (~60 TFLOP/s fp32 peak at 2048³, ~48–56 TFLOP/s at 4096³ — typical
  cuBLAS fp32 on Ada; TF32 mode would be far higher again). The chained
  device-pointer path (`ms_cuda_sgemm_dev`, single sync, zero copies) realises
  this ceiling, confirming the resident API has no per-call overhead beyond the
  kernel itself.
- The **copy-inclusive** `ms_cuda_sgemm` pays PCIe transfer **plus a per-call
  `cudaMalloc`/`cudaFree`** of all three buffers every invocation, which
  dominates. For reused weights use the device-resident API (upload once,
  `ms_cuda_sgemm_dev`, download once); it reaches the prefill throughput above
  and the decode latency above.
- **vs the CPU native kernel** (`libms_kernels.so` `ms_sgemm_f32`, ~1.1
  TFLOP/s): the resident prefill path is **~50× faster**, the compute-only/
  chained ceiling **~35–55× faster**; even the naive copy-inclusive path is
  **~1.6–7×** faster, with the gap widening as the matrix grows.
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
