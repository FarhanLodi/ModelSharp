// cublas_gemm.cu — cuBLAS SGEMM wrapper for ModelSharp's GPU GEMM hot path.
//
// Target: NVIDIA RTX 4090 (Ada, sm_89). Built into build/libms_cuda.so via
// Makefile.cuda and loaded from C# through P/Invoke (see native/GPU.md).
//
// ============================== UNTESTED ====================================
// This file has NOT been compiled or run. The build machine has the NVIDIA
// driver only (nvidia-smi works); the CUDA toolkit (nvcc, cuBLAS headers/libs)
// is NOT installed, so nvcc cannot run here. See GPU.md for how to install the
// toolkit and build. The code is written to be correct and buildable, but
// verify on a CUDA-enabled machine before trusting results.
// ============================================================================
//
// ---------------------------------------------------------------------------
// Row-major vs column-major: the transpose trick
// ---------------------------------------------------------------------------
// ModelSharp stores matrices row-major (C/C# convention). cuBLAS, inherited
// from Fortran BLAS, is column-major. We want:
//
//     C[M,N] = A[M,K] * B[K,N]            (all row-major)
//
// A row-major matrix M[r,c] occupies the exact same bytes as the column-major
// matrix Mᵀ (its transpose). So if we hand cuBLAS our row-major buffers as-is,
// it "sees" them transposed: it sees Aᵀ (K×M col-major) and Bᵀ (N×K col-major).
//
// The identity we exploit:
//
//     Cᵀ = (A*B)ᵀ = Bᵀ * Aᵀ
//
// In column-major land, Cᵀ is exactly what a row-major reader of C would call
// "C". So we ask cuBLAS to compute, in column-major terms:
//
//     C_cm[N,M] = B_cm[N,K] * A_cm[K,M]       with both ops = OP_N
//
// where B_cm is our row-major B reinterpreted (== Bᵀ), and A_cm is our
// row-major A reinterpreted (== Aᵀ). The result C_cm written column-major is,
// byte-for-byte, our desired row-major C[M,N]. No explicit transposes, no
// extra copies — we just swap the operand order and the M/N dimensions in the
// cublasSgemm call. Concretely:
//
//     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                 N, M, K,            // m, n, k for the column-major call
//                 &alpha,
//                 dB, N,              // "A" arg = our B, lda = N
//                 dA, K,              // "B" arg = our A, ldb = K
//                 &beta,
//                 dC, N);            // C, ldc = N
//
// Leading dimensions: a row-major X[rows,cols] reinterpreted as column-major
// has leading dimension = cols. So lda(B)=N, ldb(A)=K, ldc(C)=N.
// ---------------------------------------------------------------------------

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>          // CUDA driver API (CUcontext, CUstream, cuCtxPush/Pop)

#include <cstdio>
#include <cstring>

#include <map>
#include <mutex>

extern "C" {

// ---------------------------------------------------------------------------
// Error handling helpers. On failure we print to stderr and (for the compute
// path) bail out leaving C untouched; callers can treat a zeroed/garbage C as
// a failure signal, but the stderr message is the authoritative diagnostic.
// ---------------------------------------------------------------------------
#define MS_CUDA_CHECK(call, label)                                            \
    do {                                                                      \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[ms_cuda] %s failed: %s\n", (label),            \
                    cudaGetErrorString(_e));                                  \
            goto cleanup;                                                     \
        }                                                                     \
    } while (0)

#define MS_CUBLAS_CHECK(call, label)                                          \
    do {                                                                      \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "[ms_cuda] %s failed: cublas status %d\n",        \
                    (label), (int)_s);                                        \
            goto cleanup;                                                     \
        }                                                                     \
    } while (0)

// Cached cuBLAS handle. cublasCreate is relatively expensive (allocates a
// workspace / picks kernels), so we create it once on first use and reuse it.
// Not thread-safe for concurrent first-time init; ModelSharp drives GEMM from
// a single worker for the hot path. A real multithreaded host should guard
// this or create one handle per thread.
//
// We also create and bind a dedicated non-blocking CUDA stream to the handle.
// Binding the same stream means all cuBLAS calls and any explicit copies issued
// on g_stream are ordered with respect to each other on the device, while a
// single cudaStreamSynchronize(g_stream) (instead of cudaDeviceSynchronize())
// is enough to wait for results — this matters for the chained device-resident
// path where we want to issue many GEMMs back-to-back and sync only once.
static cublasHandle_t g_handle = nullptr;
static cudaStream_t   g_stream = nullptr;

// ---------------------------------------------------------------------------
// TF32 Tensor Core toggle (RTX 4090 / Ada, sm_89).
// ---------------------------------------------------------------------------
// When enabled, cuBLAS may run fp32 SGEMM on the Tensor Cores in TF32 mode:
// inputs/outputs stay fp32, but the multiply uses a 19-bit TF32 operand with a
// 10-bit mantissa (8-bit exponent). This is ~5-8x faster than the classic fp32
// FMA path on Ada, at ~1e-3 relative accuracy — a deliberate, opt-in tradeoff.
//
// Default is OFF (CUBLAS_DEFAULT_MATH = strict IEEE fp32). The flag is global
// and applied per-GEMM-call via cublasSetMathMode right before each cublasSgemm
// / cublasSgemmStridedBatched, so it takes effect uniformly across the single,
// resident, device, and strided-batched paths and regardless of which cached
// handle (runtime-primary or per-driver-context) is used. We guard the flag
// with a mutex so ms_cuda_set_tf32 / ms_cuda_get_tf32 are safe from any thread.
static int        g_tf32 = 0;          // 0 = fp32 (default), 1 = TF32 tensor op
static std::mutex g_tf32_mutex;

// Read the current TF32 flag under the lock.
static int ms_tf32_flag(void) {
    std::lock_guard<std::mutex> lock(g_tf32_mutex);
    return g_tf32;
}

// Apply the current TF32 flag to `handle` immediately before a GEMM. Cheap
// (a single host-side cuBLAS state set), and it guarantees the math mode always
// reflects ms_cuda_set_tf32 no matter which handle the call path uses. Failures
// are non-fatal: we warn and let the GEMM proceed in whatever the handle's
// previous mode was.
static void ms_apply_math_mode(cublasHandle_t handle) {
    cublasMath_t mode = ms_tf32_flag() ? CUBLAS_TF32_TENSOR_OP_MATH
                                       : CUBLAS_DEFAULT_MATH;
    cublasStatus_t s = cublasSetMathMode(handle, mode);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ms_cuda] cublasSetMathMode failed: status %d\n",
                (int)s);
    }
}

// Enable (enable != 0) or disable TF32 Tensor Core math for all subsequent
// GEMM calls. 0 => CUBLAS_DEFAULT_MATH (strict fp32); non-zero =>
// CUBLAS_TF32_TENSOR_OP_MATH. Thread-safe. Default is disabled.
void ms_cuda_set_tf32(int enable) {
    std::lock_guard<std::mutex> lock(g_tf32_mutex);
    g_tf32 = enable ? 1 : 0;
}

// Return 1 if TF32 Tensor Core math is currently enabled, else 0.
int ms_cuda_get_tf32(void) {
    std::lock_guard<std::mutex> lock(g_tf32_mutex);
    return g_tf32;
}

static cublasHandle_t ms_get_handle(void) {
    if (g_handle == nullptr) {
        cublasStatus_t s = cublasCreate(&g_handle);
        if (s != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[ms_cuda] cublasCreate failed: status %d\n",
                    (int)s);
            g_handle = nullptr;
            return nullptr;
        }
        cudaError_t e = cudaStreamCreateWithFlags(&g_stream,
                                                  cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            fprintf(stderr, "[ms_cuda] cudaStreamCreate failed: %s\n",
                    cudaGetErrorString(e));
            g_stream = nullptr;  // fall back to the default stream
        } else {
            cublasStatus_t ss = cublasSetStream(g_handle, g_stream);
            if (ss != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "[ms_cuda] cublasSetStream failed: status %d\n",
                        (int)ss);
            }
        }
    }
    return g_handle;
}

// Wait for all work on our stream (or the whole device if no stream).
static cudaError_t ms_sync(void) {
    return g_stream ? cudaStreamSynchronize(g_stream)
                    : cudaDeviceSynchronize();
}

// Returns 1 if at least one CUDA device is visible, else 0. Safe to call even
// when no driver/device is present (cudaGetDeviceCount returns an error, which
// we map to "not available" rather than crashing).
int ms_cuda_available(void) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        return 0;
    }
    return count > 0 ? 1 : 0;
}

// Returns the name of device 0 (e.g. "NVIDIA GeForce RTX 4090"), or a short
// diagnostic string on failure. The returned pointer is owned by this library
// (points at a static buffer); do not free it on the C# side. Not reentrant.
const char* ms_cuda_device_name(void) {
    static char name[256];
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) {
        snprintf(name, sizeof(name), "no-cuda-device (%s)",
                 cudaGetErrorString(e));
        return name;
    }
    // prop.name is NUL-terminated and <= 256 chars; copy defensively.
    snprintf(name, sizeof(name), "%s", prop.name);
    return name;
}

// Row-major SGEMM: C[M,N] = A[M,K] * B[K,N], alpha=1, beta=0.
//
// Copy-in / copy-out version: allocates device buffers, H2D copies A and B,
// runs cublasSgemm, D2H copies C, frees. This is the MUST-HAVE simple path.
// For a resident-weights variant that avoids re-uploading B each call, see
// ms_cuda_sgemm_resident below.
void ms_cuda_sgemm(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sgemm: bad args "
                        "(M=%d N=%d K=%d)\n", M, N, K);
        return;
    }

    cublasHandle_t handle = ms_get_handle();
    if (!handle) return;

    const size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
    const size_t bytesB = (size_t)K * (size_t)N * sizeof(float);
    const size_t bytesC = (size_t)M * (size_t)N * sizeof(float);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    MS_CUDA_CHECK(cudaMalloc((void**)&dA, bytesA), "cudaMalloc dA");
    MS_CUDA_CHECK(cudaMalloc((void**)&dB, bytesB), "cudaMalloc dB");
    MS_CUDA_CHECK(cudaMalloc((void**)&dC, bytesC), "cudaMalloc dC");

    MS_CUDA_CHECK(cudaMemcpyAsync(dA, A, bytesA, cudaMemcpyHostToDevice,
                                  g_stream), "H2D A");
    MS_CUDA_CHECK(cudaMemcpyAsync(dB, B, bytesB, cudaMemcpyHostToDevice,
                                  g_stream), "H2D B");

    // Apply the global TF32 toggle to this handle right before the GEMM.
    ms_apply_math_mode(handle);

    // Column-major transpose trick (see file header). We compute, in
    // column-major terms, C_cm[N,M] = B_cm[N,K] * A_cm[K,M], which lands the
    // bytes exactly as our desired row-major C[M,N]. Both ops are OP_N.
    MS_CUBLAS_CHECK(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,        // m, n, k (column-major)
                    &alpha,
                    dB, N,          // first operand: our B, lda = N
                    dA, K,          // second operand: our A, ldb = K
                    &beta,
                    dC, N),         // output: C, ldc = N
        "cublasSgemm");

    MS_CUDA_CHECK(cudaMemcpyAsync(C, dC, bytesC, cudaMemcpyDeviceToHost,
                                  g_stream), "D2H C");
    MS_CUDA_CHECK(ms_sync(), "stream sync");

cleanup:
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
}

// Optional resident-input variant: B is uploaded once into a device buffer the
// caller holds across many GEMMs (e.g. a weight matrix reused for every token).
// Only A is copied in and C copied out per call. Returns a device pointer
// (as an opaque void*) usable as dB in ms_cuda_sgemm_with_resident_b.
//
// Upload B[K,N] (row-major) to the device; returns the device buffer or NULL.
void* ms_cuda_upload(const float* host, int rows, int cols) {
    if (!host || rows <= 0 || cols <= 0) return nullptr;
    const size_t bytes = (size_t)rows * (size_t)cols * sizeof(float);
    ms_get_handle();  // ensure g_stream exists
    void* dev = nullptr;
    if (cudaMalloc(&dev, bytes) != cudaSuccess) return nullptr;
    if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice,
                        g_stream) != cudaSuccess ||
        ms_sync() != cudaSuccess) {
        cudaFree(dev);
        return nullptr;
    }
    return dev;
}

// Free a buffer returned by ms_cuda_upload.
void ms_cuda_free(void* dev) {
    if (dev) cudaFree(dev);
}

// C[M,N] = A[M,K] * B[K,N] where dB is a device pointer previously returned by
// ms_cuda_upload(B, K, N). A is copied H2D, C copied D2H. Same transpose trick.
void ms_cuda_sgemm_with_resident_b(const float* A, void* dB_void, float* C,
                                   int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0 || !A || !dB_void || !C) {
        fprintf(stderr, "[ms_cuda] sgemm_resident: bad args\n");
        return;
    }
    cublasHandle_t handle = ms_get_handle();
    if (!handle) return;

    const size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
    const size_t bytesC = (size_t)M * (size_t)N * sizeof(float);

    float* dA = nullptr;
    float* dC = nullptr;
    float* dB = (float*)dB_void;
    const float alpha = 1.0f, beta = 0.0f;

    MS_CUDA_CHECK(cudaMalloc((void**)&dA, bytesA), "cudaMalloc dA");
    MS_CUDA_CHECK(cudaMalloc((void**)&dC, bytesC), "cudaMalloc dC");
    MS_CUDA_CHECK(cudaMemcpyAsync(dA, A, bytesA, cudaMemcpyHostToDevice,
                                  g_stream), "H2D A");

    ms_apply_math_mode(handle);

    MS_CUBLAS_CHECK(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    dB, N,
                    dA, K,
                    &beta,
                    dC, N),
        "cublasSgemm");

    MS_CUDA_CHECK(cudaMemcpyAsync(C, dC, bytesC, cudaMemcpyDeviceToHost,
                                  g_stream), "D2H C");
    MS_CUDA_CHECK(ms_sync(), "stream sync");

cleanup:
    if (dA) cudaFree(dA);
    if (dC) cudaFree(dC);
}

// ===========================================================================
// Fully device-resident API
// ===========================================================================
// The functions below operate on raw device pointers and never implicitly
// transfer to/from the host. This is the path that matters for LLM inference:
// upload a weight matrix ONCE (ms_cuda_upload), keep activations and outputs on
// the device (ms_cuda_alloc / ms_cuda_upload_into), and chain many GEMMs
// (ms_cuda_sgemm_dev) without any per-call cudaMalloc/cudaFree or PCIe copies.
// Only the final result is brought back with ms_cuda_download. All device
// pointers must be freed with ms_cuda_free.

// Allocate an UNINITIALISED device buffer of `n` floats. Returns NULL on error.
// Use for activation / output buffers that will be written by a GEMM.
void* ms_cuda_alloc(size_t n) {
    if (n == 0) return nullptr;
    void* dev = nullptr;
    cudaError_t e = cudaMalloc(&dev, n * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "[ms_cuda] ms_cuda_alloc(%zu) failed: %s\n", n,
                cudaGetErrorString(e));
        return nullptr;
    }
    return dev;
}

// Copy `n` host floats into an existing device buffer `dev`. Returns 0 on
// success, non-zero on failure. Use to refresh a resident activation buffer
// in place without reallocating.
int ms_cuda_upload_into(void* dev, const float* host, size_t n) {
    if (!dev || !host || n == 0) return -1;
    ms_get_handle();  // ensure g_stream exists
    cudaError_t e = cudaMemcpyAsync(dev, host, n * sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[ms_cuda] ms_cuda_upload_into failed: %s\n",
                cudaGetErrorString(e));
        return (int)e;
    }
    e = ms_sync();
    return (e == cudaSuccess) ? 0 : (int)e;
}

// Copy `n` floats from device buffer `dev` back to the host. Returns 0 on
// success. This is the ONLY host transfer in a chained sequence — call it once
// on the final output.
int ms_cuda_download(void* dev, float* host, size_t n) {
    if (!dev || !host || n == 0) return -1;
    ms_get_handle();
    cudaError_t e = cudaMemcpyAsync(host, dev, n * sizeof(float),
                                    cudaMemcpyDeviceToHost, g_stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[ms_cuda] ms_cuda_download failed: %s\n",
                cudaGetErrorString(e));
        return (int)e;
    }
    e = ms_sync();
    return (e == cudaSuccess) ? 0 : (int)e;
}

// Fully device-resident SGEMM: C[M,N] = A[M,K] * B[K,N], all row-major, where
// dA, dB, dC are device pointers (dA = activations [M,K], dB = weights [K,N],
// dC = output [M,N]). NO host transfers and NO cudaMalloc/free occur here — the
// GEMM is simply enqueued on g_stream. The caller decides when to synchronise
// (ms_cuda_sync) or download. This is what lets a chain of layers run with the
// copy-free on-device throughput.
//
// Uses the identical column-major transpose trick as ms_cuda_sgemm: we compute
// (column-major) C_cm[N,M] = B_cm[N,K] * A_cm[K,M] with both ops OP_N, which
// lands the bytes as row-major C[M,N]. Returns 0 on success.
int ms_cuda_sgemm_dev(void* dA_void, void* dB_void, void* dC_void,
                      int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0 || !dA_void || !dB_void || !dC_void) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sgemm_dev: bad args "
                        "(M=%d N=%d K=%d)\n", M, N, K);
        return -1;
    }
    cublasHandle_t handle = ms_get_handle();
    if (!handle) return -2;

    const float* dA = (const float*)dA_void;
    const float* dB = (const float*)dB_void;
    float*       dC = (float*)dC_void;
    const float alpha = 1.0f, beta = 0.0f;

    ms_apply_math_mode(handle);

    cublasStatus_t s =
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    dB, N,      // first operand = our B (weights), lda = N
                    dA, K,      // second operand = our A (acts),    ldb = K
                    &beta,
                    dC, N);     // output = our C,                   ldc = N
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sgemm_dev: cublas status %d\n",
                (int)s);
        return (int)s;
    }
    return 0;
}

// Block until all enqueued device work on our stream has completed. Returns 0
// on success. Use after a batch of ms_cuda_sgemm_dev calls before timing or
// before reading results another way.
int ms_cuda_sync(void) {
    cudaError_t e = ms_sync();
    return (e == cudaSuccess) ? 0 : (int)e;
}

// Optional: release the cached handle (e.g. on shutdown). Idempotent.
void ms_cuda_shutdown(void) {
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
}

// ===========================================================================
// Externally-provided CUDA driver context API
// ===========================================================================
// The functions above all live in (and implicitly create) the CUDA RUNTIME
// primary context. That is fine when ModelSharp owns the GPU, but it is WRONG
// when an interop layer such as ILGPU owns its own DRIVER-API context and hands
// us device pointers that are only valid in THAT context.
//
// A cublasHandle_t and a device pointer are bound to whatever CUDA context is
// CURRENT at the time they are created / used. If we create a handle in the
// runtime primary context and then run a GEMM on ILGPU's device pointers (which
// belong to ILGPU's driver context), cuBLAS dereferences those pointers in the
// wrong address space — producing a deterministic garbage region rather than a
// hard fault. (This is the bug that motivated this API.)
//
// The fix below: before every cuBLAS call we cuCtxPushCurrent(caller_ctx) to
// make the caller's context current, run the GEMM through a cublasHandle_t that
// was created WHILE that context was current (cached per-context), then always
// cuCtxPopCurrent — even on error paths — so we never corrupt the caller's
// context stack. Zero cross-context copies: we use the caller's device pointers
// and the caller's stream directly.

// Per-context cublas handle cache, keyed by the driver-API CUcontext. Guarded
// by a mutex so it is safe to call from multiple host threads. cublasCreate is
// expensive, so we create one handle per context lazily and reuse it.
static std::map<CUcontext, cublasHandle_t> g_ctx_handles;
static std::mutex                          g_ctx_mutex;

// RAII guard: push a driver context current on construction, pop on
// destruction. Guarantees the push/pop are balanced on every code path,
// including early returns and error paths.
namespace {
struct CtxGuard {
    bool       pushed = false;
    CUresult   err    = CUDA_SUCCESS;
    explicit CtxGuard(CUcontext ctx) {
        err = cuCtxPushCurrent(ctx);
        pushed = (err == CUDA_SUCCESS);
        if (!pushed) {
            fprintf(stderr, "[ms_cuda] cuCtxPushCurrent failed: CUresult %d\n",
                    (int)err);
        }
    }
    ~CtxGuard() {
        if (pushed) {
            CUcontext popped = nullptr;
            CUresult e = cuCtxPopCurrent(&popped);
            if (e != CUDA_SUCCESS) {
                fprintf(stderr, "[ms_cuda] cuCtxPopCurrent failed: "
                                "CUresult %d\n", (int)e);
            }
        }
    }
    CtxGuard(const CtxGuard&)            = delete;
    CtxGuard& operator=(const CtxGuard&) = delete;
};
} // anonymous namespace

// Get-or-create the cached cublasHandle_t for the currently-current context
// `ctx`. MUST be called with `ctx` already pushed current (the handle is bound
// to whatever context is current at cublasCreate time). Returns nullptr on
// failure. Thread-safe.
static cublasHandle_t ms_get_ctx_handle(CUcontext ctx) {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    auto it = g_ctx_handles.find(ctx);
    if (it != g_ctx_handles.end()) {
        return it->second;
    }
    cublasHandle_t h = nullptr;
    cublasStatus_t s = cublasCreate(&h);   // bound to the current (pushed) ctx
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ms_cuda] cublasCreate (ctx) failed: status %d\n",
                (int)s);
        return nullptr;
    }
    g_ctx_handles[ctx] = h;
    return h;
}

// Get-or-create the cached cublasHandle_t for an EXTERNALLY-provided driver
// context. Pushes the context current, creates/looks up the handle, pops.
// Returns 0 on success, non-zero (CUresult or cublas status) on failure.
int ms_cuda_bind_context(void* cuContext) {
    CUcontext ctx = (CUcontext)cuContext;
    if (ctx == nullptr) {
        fprintf(stderr, "[ms_cuda] ms_cuda_bind_context: null context\n");
        return -1;
    }
    CtxGuard guard(ctx);
    if (!guard.pushed) return (int)guard.err;
    cublasHandle_t h = ms_get_ctx_handle(ctx);
    return h ? 0 : -2;
}

// Destroy and remove the cached handle for `cuContext`. Pushes the context
// current for cublasDestroy (the handle belongs to that context), then pops.
// Idempotent: a no-op if no handle is cached for the context.
void ms_cuda_release_context(void* cuContext) {
    CUcontext ctx = (CUcontext)cuContext;
    if (ctx == nullptr) return;

    cublasHandle_t h = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_ctx_mutex);
        auto it = g_ctx_handles.find(ctx);
        if (it == g_ctx_handles.end()) return;
        h = it->second;
        g_ctx_handles.erase(it);
    }
    // Destroy with the owning context current so cublas frees its workspace in
    // the right address space.
    CtxGuard guard(ctx);
    if (h) cublasDestroy(h);
}

// Synchronise the given driver stream. If cuStream is NULL, synchronise the
// device (whichever context is current). Uses the runtime API, which is
// interoperable with driver-created streams (a CUstream IS a cudaStream_t).
// Returns 0 on success, non-zero cudaError_t otherwise.
//
// Note: cudaStreamSynchronize operates on the calling context. The caller
// should ensure the relevant context is current, OR rely on the fact that a
// stream sync issued right after ms_cuda_sgemm_ctx (which already pushed/popped)
// is most robustly done by passing the same CUstream here; if you need it bound
// to a specific context, push it first on the C# side. For the common single-
// context ILGPU case this is sufficient.
int ms_cuda_sync_stream(void* cuStream) {
    cudaError_t e;
    if (cuStream) {
        e = cudaStreamSynchronize((cudaStream_t)cuStream);
    } else {
        e = cudaDeviceSynchronize();
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sync_stream failed: %s\n",
                cudaGetErrorString(e));
        return (int)e;
    }
    return 0;
}

// Row-major SGEMM C[M,N] = A[M,K] * B[K,N] (alpha=1, beta=0) on device pointers
// dA, dB, dC that live in the EXTERNALLY-PROVIDED driver context `cuContext`,
// enqueued on `cuStream` (NULL = that context's default stream). Zero copies:
// we operate directly on the caller's device pointers and stream.
//
// Steps: push cuContext current (balanced pop guaranteed via CtxGuard) -> get
// or lazily create the per-context cublas handle -> bind cuStream to it -> run
// the column-major Cᵀ = Bᵀ·Aᵀ transpose trick (identical to ms_cuda_sgemm_dev)
// -> pop. The GEMM is enqueued only; the caller syncs (ms_cuda_sync_stream)
// when it needs the result. Returns 0 on success, non-zero cublas/cuda status.
int ms_cuda_sgemm_ctx(void* cuContext, void* cuStream,
                      void* dA, void* dB, void* dC,
                      int M, int N, int K) {
    if (cuContext == nullptr) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sgemm_ctx: null context\n");
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0 || !dA || !dB || !dC) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sgemm_ctx: bad args "
                        "(M=%d N=%d K=%d)\n", M, N, K);
        return -1;
    }

    CUcontext ctx = (CUcontext)cuContext;
    CtxGuard guard(ctx);                 // push now, pop on every exit path
    if (!guard.pushed) return (int)guard.err;

    cublasHandle_t handle = ms_get_ctx_handle(ctx);
    if (!handle) return -2;

    // Bind the caller's stream (or the context default stream if NULL). A
    // driver CUstream is interchangeable with a runtime cudaStream_t here.
    cublasStatus_t ss = cublasSetStream(handle, (cudaStream_t)cuStream);
    if (ss != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sgemm_ctx: cublasSetStream "
                        "status %d\n", (int)ss);
        return (int)ss;
    }

    const float* a = (const float*)dA;
    const float* b = (const float*)dB;
    float*       c = (float*)dC;
    const float alpha = 1.0f, beta = 0.0f;

    ms_apply_math_mode(handle);

    // Column-major transpose trick (see file header): compute, in column-major
    // terms, C_cm[N,M] = B_cm[N,K] * A_cm[K,M], which lands the bytes as our
    // desired row-major C[M,N]. Both ops OP_N.
    cublasStatus_t s =
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    b, N,       // first operand = our B (weights), lda = N
                    a, K,       // second operand = our A (acts),    ldb = K
                    &beta,
                    c, N);      // output = our C,                   ldc = N
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ms_cuda] ms_cuda_sgemm_ctx: cublas status %d\n",
                (int)s);
        return (int)s;
    }
    return 0;   // guard pops the context here
}

// ---------------------------------------------------------------------------
// Strided-batched row-major SGEMM, context-aware.
// ---------------------------------------------------------------------------
// For each b in [0, batch):
//     C[b][M,N] = A[b][M,K] * B[b][K,N]     (all row-major, alpha=1, beta=0)
// where the b-th matrices begin at dA + b*strideA, dB + b*strideB,
// dC + b*strideC. Strides are in ELEMENTS (floats), not bytes. strideA and/or
// strideB may be 0 to broadcast a single matrix across every batch (e.g. a
// shared weight reused for all batches); strideC must point at distinct outputs.
//
// Same column-major Cᵀ = Bᵀ·Aᵀ transpose trick as ms_cuda_sgemm_ctx, applied to
// cublasSgemmStridedBatched. In the swapped column-major form our row-major B
// becomes the "A" operand and our row-major A becomes the "B" operand; the
// dimensions become (m,n,k) = (N,M,K) and the leading dims/strides pair with the
// operand they belong to:
//
//   cublasSgemmStridedBatched(handle, OP_N, OP_N,
//       N, M, K, &alpha,
//       dB, N, strideB,     // first operand = our B, lda = N, stride = strideB
//       dA, K, strideA,     // second operand = our A, ldb = K, stride = strideA
//       &beta,
//       dC, N, strideC,     // output = our C, ldc = N, stride = strideC
//       batch);
//
// Enqueued only on cuStream (no sync). Pointers live in driver context
// cuContext. Returns 0 on success, else a CUresult / cublasStatus_t code.
int ms_cuda_sgemm_strided_batched_ctx(void* cuContext, void* cuStream,
                                      void* dA, void* dB, void* dC,
                                      int M, int N, int K,
                                      long long strideA, long long strideB,
                                      long long strideC,
                                      int batch) {
    if (cuContext == nullptr) {
        fprintf(stderr, "[ms_cuda] sgemm_strided_batched_ctx: null context\n");
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0 || !dA || !dB || !dC) {
        fprintf(stderr, "[ms_cuda] sgemm_strided_batched_ctx: bad args "
                        "(M=%d N=%d K=%d batch=%d)\n", M, N, K, batch);
        return -1;
    }

    CUcontext ctx = (CUcontext)cuContext;
    CtxGuard guard(ctx);                 // push now, pop on every exit path
    if (!guard.pushed) return (int)guard.err;

    cublasHandle_t handle = ms_get_ctx_handle(ctx);
    if (!handle) return -2;

    cublasStatus_t ss = cublasSetStream(handle, (cudaStream_t)cuStream);
    if (ss != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ms_cuda] sgemm_strided_batched_ctx: cublasSetStream "
                        "status %d\n", (int)ss);
        return (int)ss;
    }

    const float* a = (const float*)dA;
    const float* b = (const float*)dB;
    float*       c = (float*)dC;
    const float alpha = 1.0f, beta = 0.0f;

    ms_apply_math_mode(handle);

    // Column-major transpose trick (see comment above). cuBLAS takes the strides
    // as `long long int`; pass them straight through.
    cublasStatus_t s =
        cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  b, N, (long long)strideB,  // our B, lda=N
                                  a, K, (long long)strideA,  // our A, ldb=K
                                  &beta,
                                  c, N, (long long)strideC,  // our C, ldc=N
                                  batch);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ms_cuda] sgemm_strided_batched_ctx: cublas status "
                        "%d\n", (int)s);
        return (int)s;
    }
    return 0;   // guard pops the context here
}

} // extern "C"
