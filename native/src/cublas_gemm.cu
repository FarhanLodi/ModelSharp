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

#include <cstdio>
#include <cstring>

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

} // extern "C"
