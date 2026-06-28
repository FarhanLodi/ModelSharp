// cublas_test.cu — standalone correctness + benchmark harness for the
// ModelSharp cuBLAS SGEMM path (src/cublas_gemm.cu).
//
// Build (either link against the .so, or compile both .cu directly):
//   nvcc -O3 -arch=sm_89 native/test/cublas_test.cu native/src/cublas_gemm.cu \
//        -lcublas -lcudart -o /tmp/cublas_test
//
// Verifies ms_cuda_sgemm against a CPU triple-loop reference (rel err < 1e-3
// vs output RMS) and benchmarks copy-inclusive GFLOP/s/TFLOP/s for square
// 1024/2048/4096 GEMMs, plus a compute-only (resident, inputs on device)
// ceiling using the ms_cuda_upload / ms_cuda_sgemm_with_resident_b helpers.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>          // driver API: CUcontext / CUstream / cuMemAlloc ...

// Entry points from src/cublas_gemm.cu (extern "C").
extern "C" {
    int         ms_cuda_available(void);
    const char* ms_cuda_device_name(void);
    void        ms_cuda_sgemm(const float* A, const float* B, float* C,
                              int M, int N, int K);
    void*       ms_cuda_upload(const float* host, int rows, int cols);
    void        ms_cuda_free(void* dev);
    void        ms_cuda_sgemm_with_resident_b(const float* A, void* dB, float* C,
                                              int M, int N, int K);
    void        ms_cuda_shutdown(void);
    // Fully device-resident API (added for the LLM weight-reuse path).
    void*       ms_cuda_alloc(size_t n);
    int         ms_cuda_upload_into(void* dev, const float* host, size_t n);
    int         ms_cuda_download(void* dev, float* host, size_t n);
    int         ms_cuda_sgemm_dev(void* dA, void* dB, void* dC,
                                  int M, int N, int K);
    int         ms_cuda_sync(void);
    // Externally-provided driver-context API (the ILGPU-interop path).
    int         ms_cuda_sgemm_ctx(void* cuContext, void* cuStream,
                                  void* dA, void* dB, void* dC,
                                  int M, int N, int K);
    int         ms_cuda_bind_context(void* cuContext);
    void        ms_cuda_release_context(void* cuContext);
    int         ms_cuda_sync_stream(void* cuStream);
}

using clk = std::chrono::high_resolution_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch()).count();
}

static void fill_rand(std::vector<float>& v, unsigned seed) {
    srand(seed);
    for (auto& x : v) x = (float)rand() / (float)RAND_MAX - 0.5f;
}

// Naive row-major reference: C[M,N] = A[M,K] * B[K,N].
static void cpu_ref(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[(size_t)i * K + k] * (double)B[(size_t)k * N + j];
            C[(size_t)i * N + j] = (float)acc;
        }
    }
}

// Returns ||gpu - ref|| / ||ref|| (both RMS).
static double rel_err(const std::vector<float>& gpu,
                      const std::vector<float>& ref) {
    double sd = 0.0, sr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = (double)gpu[i] - (double)ref[i];
        sd += d * d;
        sr += (double)ref[i] * (double)ref[i];
    }
    double rms_ref = std::sqrt(sr / ref.size());
    double rms_d   = std::sqrt(sd / ref.size());
    return rms_ref > 0 ? rms_d / rms_ref : rms_d;
}

static bool correctness(int M, int N, int K) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    std::vector<float> C((size_t)M * N), Cref((size_t)M * N);
    fill_rand(A, 1234 + M);
    fill_rand(B, 5678 + N);

    ms_cuda_sgemm(A.data(), B.data(), C.data(), M, N, K);
    cpu_ref(A.data(), B.data(), Cref.data(), M, N, K);

    double e = rel_err(C, Cref);
    bool ok = e < 1e-3;
    printf("  [%s] %4dx%4dx%4d  rel_err=%.3e\n",
           ok ? "PASS" : "FAIL", M, N, K, e);
    return ok;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n & 1 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// Copy-inclusive benchmark of ms_cuda_sgemm (square N).
static void bench_copy(int N, int warmup, int iters) {
    std::vector<float> A((size_t)N * N), B((size_t)N * N), C((size_t)N * N);
    fill_rand(A, 11); fill_rand(B, 22);
    const double flops = 2.0 * (double)N * N * N;

    for (int i = 0; i < warmup; ++i)
        ms_cuda_sgemm(A.data(), B.data(), C.data(), N, N, N);

    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        ms_cuda_sgemm(A.data(), B.data(), C.data(), N, N, N);
        cudaDeviceSynchronize();
        t.push_back(now_ms() - t0);
    }
    double ms = median(t);
    double gflops = flops / (ms * 1e6);
    printf("  copy-incl  %4d^3  %8.2f ms   %8.1f GFLOP/s   %6.2f TFLOP/s\n",
           N, ms, gflops, gflops / 1000.0);
}

// Pure compute-only ceiling: A, B, C all resident on device; time only the
// cublasSgemm kernel (no PCIe copies, no cudaMalloc in the loop). Uses the same
// column-major transpose trick as ms_cuda_sgemm so the measured kernel matches.
static void bench_compute_only(int N, int warmup, int iters) {
    std::vector<float> A((size_t)N * N), B((size_t)N * N);
    fill_rand(A, 33); fill_rand(B, 44);
    const double flops = 2.0 * (double)N * N * N;
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc((void**)&dA, bytes);
    cudaMalloc((void**)&dB, bytes);
    cudaMalloc((void**)&dC, bytes);
    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t h; cublasCreate(&h);
    const float alpha = 1.0f, beta = 0.0f;
    auto gemm = [&]() {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
    };

    for (int i = 0; i < warmup; ++i) gemm();
    cudaDeviceSynchronize();

    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        gemm();
        cudaDeviceSynchronize();
        t.push_back(now_ms() - t0);
    }
    double ms = median(t);
    double gflops = flops / (ms * 1e6);
    printf("  compute    %4d^3  %8.2f ms   %8.1f GFLOP/s   %6.2f TFLOP/s "
           "(A/B/C resident, kernel only)\n",
           N, ms, gflops, gflops / 1000.0);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// ---------------------------------------------------------------------------
// Device-resident API correctness: C[M,N]=A[M,K]*B[K,N] via ms_cuda_sgemm_dev
// on pure device pointers, compared to the CPU reference.
// ---------------------------------------------------------------------------
static bool correctness_dev(int M, int N, int K) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    std::vector<float> C((size_t)M * N), Cref((size_t)M * N);
    fill_rand(A, 4321 + M);
    fill_rand(B, 8765 + N);

    void* dA = ms_cuda_upload(A.data(), M, K);
    void* dB = ms_cuda_upload(B.data(), K, N);
    void* dC = ms_cuda_alloc((size_t)M * N);
    ms_cuda_sgemm_dev(dA, dB, dC, M, N, K);
    ms_cuda_sync();
    ms_cuda_download(dC, C.data(), (size_t)M * N);
    ms_cuda_free(dA); ms_cuda_free(dB); ms_cuda_free(dC);

    cpu_ref(A.data(), B.data(), Cref.data(), M, N, K);
    double e = rel_err(C, Cref);
    bool ok = e < 1e-3;
    printf("  [%s] dev %4dx%4dx%4d  rel_err=%.3e\n",
           ok ? "PASS" : "FAIL", M, N, K, e);
    return ok;
}

// ---------------------------------------------------------------------------
// DECODE-reuse simulation. A weight matrix W[K,N] is uploaded ONCE and reused
// for R successive GEMMs with a small activation batch A[M,K] (M=1 for greedy
// decode, M=8 for small-batch). Activations and the output stay on device for
// the whole loop; only the final result is downloaded. This is the LLM
// inference pattern (one persistent weight, many token steps).
// ---------------------------------------------------------------------------
static void bench_decode_reuse(int M, int K, int N, int R) {
    std::vector<float> Wh((size_t)K * N), Ah((size_t)M * K),
                       Ch((size_t)M * N);
    fill_rand(Wh, 77); fill_rand(Ah, 88);

    void* dW = ms_cuda_upload(Wh.data(), K, N);    // upload weight ONCE
    void* dA = ms_cuda_upload(Ah.data(), M, K);    // resident activations
    void* dC = ms_cuda_alloc((size_t)M * N);       // resident output

    const double flops_per = 2.0 * (double)M * N * K;

    // warmup
    for (int i = 0; i < 5; ++i) ms_cuda_sgemm_dev(dA, dW, dC, M, N, K);
    ms_cuda_sync();

    double t0 = now_ms();
    for (int i = 0; i < R; ++i)
        ms_cuda_sgemm_dev(dA, dW, dC, M, N, K);    // all on device, no copies
    ms_cuda_sync();                                // single sync for the batch
    double ms_total = now_ms() - t0;

    ms_cuda_download(dC, Ch.data(), (size_t)M * N);  // download final only

    double ms_per = ms_total / R;
    double gflops = (flops_per * R) / (ms_total * 1e6);
    printf("  decode M=%-3d K=%d N=%d  R=%d  %7.4f ms/gemm  "
           "%8.1f GFLOP/s  %6.3f TFLOP/s\n",
           M, K, N, R, ms_per, gflops, gflops / 1000.0);

    ms_cuda_free(dW); ms_cuda_free(dA); ms_cuda_free(dC);
}

// ---------------------------------------------------------------------------
// PREFILL simulation: same resident weight, a larger activation batch (M=512).
// One GEMM per call, weight stays resident, activations/output on device.
// ---------------------------------------------------------------------------
static void bench_prefill(int M, int K, int N, int R) {
    std::vector<float> Wh((size_t)K * N), Ah((size_t)M * K);
    fill_rand(Wh, 91); fill_rand(Ah, 92);

    void* dW = ms_cuda_upload(Wh.data(), K, N);
    void* dA = ms_cuda_upload(Ah.data(), M, K);
    void* dC = ms_cuda_alloc((size_t)M * N);

    const double flops_per = 2.0 * (double)M * N * K;

    for (int i = 0; i < 3; ++i) ms_cuda_sgemm_dev(dA, dW, dC, M, N, K);
    ms_cuda_sync();

    std::vector<double> t;
    for (int i = 0; i < R; ++i) {
        double t0 = now_ms();
        ms_cuda_sgemm_dev(dA, dW, dC, M, N, K);
        ms_cuda_sync();
        t.push_back(now_ms() - t0);
    }
    double ms = median(t);
    double gflops = flops_per / (ms * 1e6);
    printf("  prefill M=%-4d K=%d N=%d  %7.4f ms/gemm  "
           "%8.1f GFLOP/s  %6.2f TFLOP/s\n",
           M, K, N, ms, gflops, gflops / 1000.0);

    ms_cuda_free(dW); ms_cuda_free(dA); ms_cuda_free(dC);
}

// ---------------------------------------------------------------------------
// Fully-device chained-GEMM ceiling. Square N, all buffers resident; issue
// `chain` GEMMs back-to-back through ms_cuda_sgemm_dev (alternating output
// buffers so each feeds the next, like stacked layers) with a SINGLE sync at
// the end and ZERO host transfers. Shows the copy-free device ceiling.
// ---------------------------------------------------------------------------
static void bench_chain_dev(int N, int chain, int iters) {
    std::vector<float> A((size_t)N * N), B((size_t)N * N);
    fill_rand(A, 55); fill_rand(B, 66);
    const double flops_per = 2.0 * (double)N * N * N;

    void* dA = ms_cuda_upload(A.data(), N, N);
    void* dB = ms_cuda_upload(B.data(), N, N);
    void* dX = ms_cuda_alloc((size_t)N * N);   // ping
    void* dY = ms_cuda_alloc((size_t)N * N);   // pong

    auto run_chain = [&]() {
        // X = A*B, then Y = X*B, X = Y*B, ... (each GEMM reuses resident B)
        void* in = dA; void* out = dX;
        for (int c = 0; c < chain; ++c) {
            ms_cuda_sgemm_dev(in, dB, out, N, N, N);
            void* tmp = (out == dX) ? dY : dX;
            in = out; out = tmp;
        }
    };

    for (int i = 0; i < 2; ++i) run_chain();
    ms_cuda_sync();

    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        run_chain();
        ms_cuda_sync();
        t.push_back((now_ms() - t0) / chain);   // per-GEMM
    }
    double ms = median(t);
    double gflops = flops_per / (ms * 1e6);
    printf("  chain-dev  %4d^3  chain=%d  %8.3f ms/gemm  "
           "%8.1f GFLOP/s  %6.2f TFLOP/s (no H2D/D2H)\n",
           N, chain, ms, gflops, gflops / 1000.0);

    ms_cuda_free(dA); ms_cuda_free(dB); ms_cuda_free(dX); ms_cuda_free(dY);
}

// ---------------------------------------------------------------------------
// EXTERNALLY-PROVIDED DRIVER CONTEXT correctness. We cannot create an ILGPU
// context here, but we can reproduce the exact condition it creates: a CUDA
// DRIVER-API context that ms_cuda_* did NOT make, holding driver-allocated
// device pointers (cuMemAlloc). We:
//   1. cuInit(0); cuDevicePrimaryCtxRetain to get a CUcontext.
//   2. cuMemAlloc dA/dB/dC IN that context; cuMemcpyHtoD the inputs.
//   3. CRITICALLY pop the context so it is NOT current on entry to the wrapper
//      (the whole point: the wrapper must push it itself).
//   4. call ms_cuda_sgemm_ctx(ctx, stream, dA, dB, dC, ...).
//   5. ms_cuda_sync_stream; cuMemcpyDtoH; compare to CPU reference.
// Returns true on PASS (rel err < 1e-3).
// ---------------------------------------------------------------------------
#define DRV_CHECK(call, label)                                                 \
    do {                                                                       \
        CUresult _r = (call);                                                  \
        if (_r != CUDA_SUCCESS) {                                              \
            const char* _m = nullptr; cuGetErrorString(_r, &_m);              \
            printf("  [FAIL] %s: CUresult %d (%s)\n", (label), (int)_r,        \
                   _m ? _m : "?");                                             \
            return false;                                                      \
        }                                                                      \
    } while (0)

static bool correctness_ctx(CUcontext ctx, CUstream stream,
                            int M, int N, int K, const char* tag) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    std::vector<float> C((size_t)M * N), Cref((size_t)M * N);
    fill_rand(A, 2468 + M);
    fill_rand(B, 1357 + N);

    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    // Allocate + upload IN the externally-created context (push it for the
    // driver memory ops, then pop so it is NOT current on the wrapper call).
    DRV_CHECK(cuCtxPushCurrent(ctx), "push (alloc)");
    CUdeviceptr dA = 0, dB = 0, dC = 0;
    DRV_CHECK(cuMemAlloc(&dA, bytesA), "cuMemAlloc dA");
    DRV_CHECK(cuMemAlloc(&dB, bytesB), "cuMemAlloc dB");
    DRV_CHECK(cuMemAlloc(&dC, bytesC), "cuMemAlloc dC");
    DRV_CHECK(cuMemcpyHtoD(dA, A.data(), bytesA), "HtoD A");
    DRV_CHECK(cuMemcpyHtoD(dB, B.data(), bytesB), "HtoD B");
    CUcontext popped = nullptr;
    DRV_CHECK(cuCtxPopCurrent(&popped), "pop (alloc)");

    // Sanity: the context must NOT be current here (proves the wrapper does the
    // push itself, which is the whole point of this API).
    CUcontext cur = (CUcontext)1;
    cuCtxGetCurrent(&cur);
    bool not_current_on_entry = (cur != ctx);

    int rc = ms_cuda_sgemm_ctx((void*)ctx, (void*)stream,
                               (void*)dA, (void*)dB, (void*)dC, M, N, K);
    if (rc != 0) {
        printf("  [FAIL] %s ctx %dx%dx%d: ms_cuda_sgemm_ctx rc=%d\n",
               tag, M, N, K, rc);
        return false;
    }
    int sc = ms_cuda_sync_stream((void*)stream);
    if (sc != 0) {
        printf("  [FAIL] %s ctx %dx%dx%d: ms_cuda_sync_stream rc=%d\n",
               tag, M, N, K, sc);
        return false;
    }

    DRV_CHECK(cuCtxPushCurrent(ctx), "push (D2H)");
    DRV_CHECK(cuMemcpyDtoH(C.data(), dC, bytesC), "DtoH C");
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    DRV_CHECK(cuCtxPopCurrent(&popped), "pop (D2H)");

    cpu_ref(A.data(), B.data(), Cref.data(), M, N, K);
    double e = rel_err(C, Cref);
    bool ok = e < 1e-3;
    printf("  [%s] ctx %s %4dx%4dx%4d  rel_err=%.3e  (not-current-on-entry=%s)\n",
           ok ? "PASS" : "FAIL", tag, M, N, K, e,
           not_current_on_entry ? "yes" : "NO!");
    return ok && not_current_on_entry;
}

// Benchmark the context path on a NULL (default) stream, square N, compute-only
// (buffers resident in the external context, single sync after the loop).
static bool bench_ctx(CUcontext ctx, CUstream stream, int N, int iters) {
    std::vector<float> A((size_t)N * N), B((size_t)N * N);
    fill_rand(A, 13); fill_rand(B, 14);
    const size_t bytes = (size_t)N * N * sizeof(float);
    const double flops = 2.0 * (double)N * N * N;

    DRV_CHECK(cuCtxPushCurrent(ctx), "push bench alloc");
    CUdeviceptr dA = 0, dB = 0, dC = 0;
    DRV_CHECK(cuMemAlloc(&dA, bytes), "alloc dA");
    DRV_CHECK(cuMemAlloc(&dB, bytes), "alloc dB");
    DRV_CHECK(cuMemAlloc(&dC, bytes), "alloc dC");
    DRV_CHECK(cuMemcpyHtoD(dA, A.data(), bytes), "HtoD A");
    DRV_CHECK(cuMemcpyHtoD(dB, B.data(), bytes), "HtoD B");
    CUcontext popped = nullptr;
    DRV_CHECK(cuCtxPopCurrent(&popped), "pop bench alloc");

    for (int i = 0; i < 3; ++i)
        ms_cuda_sgemm_ctx((void*)ctx,(void*)stream,(void*)dA,(void*)dB,(void*)dC,N,N,N);
    ms_cuda_sync_stream((void*)stream);

    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        ms_cuda_sgemm_ctx((void*)ctx,(void*)stream,(void*)dA,(void*)dB,(void*)dC,N,N,N);
        ms_cuda_sync_stream((void*)stream);
        t.push_back(now_ms() - t0);
    }
    double ms = median(t);
    double gflops = flops / (ms * 1e6);
    printf("  ctx-gemm   %4d^3  %8.3f ms/gemm  %8.1f GFLOP/s  %6.2f TFLOP/s\n",
           N, ms, gflops, gflops / 1000.0);

    DRV_CHECK(cuCtxPushCurrent(ctx), "push bench free");
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    DRV_CHECK(cuCtxPopCurrent(&popped), "pop bench free");
    return true;
}

static bool run_ctx_tests() {
    printf("\n--- EXTERNALLY-PROVIDED DRIVER CONTEXT path "
           "(simulates ILGPU interop) ---\n");
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { printf("  cuInit failed (%d)\n", (int)r); return false; }
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { printf("  cuDeviceGet failed\n"); return false; }

    // Externally-created context the wrapper did NOT make (cuCtxCreate gives an
    // independent driver context, the closest analogue to ILGPU's own context).
    CUcontext ctx = nullptr;
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
        printf("  cuCtxCreate failed\n"); return false;
    }
    // Create a stream in that context, then pop it (cuCtxCreate leaves it
    // current; we want the wrapper to push it itself).
    CUstream stream = nullptr;
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);

    // Up-front bind (optional lifecycle): create the cached handle once.
    int br = ms_cuda_bind_context((void*)ctx);
    printf("  ms_cuda_bind_context rc=%d\n", br);

    bool ok = true;
    // On a user stream:
    ok &= correctness_ctx(ctx, stream, 64, 64, 64, "stream");
    ok &= correctness_ctx(ctx, stream, 256, 384, 500, "stream");
    ok &= correctness_ctx(ctx, stream, 1024, 1024, 1024, "stream");  // the shape that failed
    // On the NULL/default stream of that context:
    ok &= correctness_ctx(ctx, nullptr, 1024, 1024, 1024, "null-strm");
    ok &= correctness_ctx(ctx, nullptr, 2048, 2048, 2048, "null-strm");
    printf("CONTEXT CORRECTNESS: %s\n", ok ? "ALL PASS" : "FAILURE");

    printf("\n--- BENCHMARK: external-context GEMM (compute-only) ---\n");
    bench_ctx(ctx, stream, 1024, 9);
    bench_ctx(ctx, stream, 2048, 9);
    bench_ctx(ctx, stream, 4096, 7);

    // Lifecycle: release the per-context handle, then destroy the context.
    ms_cuda_release_context((void*)ctx);
    cuCtxPushCurrent(ctx);
    cuStreamDestroy(stream);
    cuCtxPopCurrent(&popped);
    cuCtxDestroy(ctx);
    return ok;
}

int main() {
    printf("=== ModelSharp cuBLAS SGEMM test/bench ===\n");
    printf("ms_cuda_available() = %d\n", ms_cuda_available());
    printf("ms_cuda_device_name() = %s\n", ms_cuda_device_name());
    if (!ms_cuda_available()) {
        fprintf(stderr, "No CUDA device available; aborting.\n");
        return 1;
    }

    printf("\n--- CORRECTNESS (rel err vs CPU ref, threshold 1e-3) ---\n");
    bool ok = true;
    ok &= correctness(64, 64, 64);
    ok &= correctness(256, 384, 500);
    ok &= correctness(1024, 1024, 1024);
    ok &= correctness_dev(1, 4096, 4096);
    ok &= correctness_dev(8, 4096, 4096);
    ok &= correctness_dev(512, 1024, 1024);
    printf("CORRECTNESS: %s\n", ok ? "ALL PASS" : "FAILURE");

    printf("\n--- BENCHMARK: copy-inclusive (H2D + GEMM + D2H) ---\n");
    bench_copy(1024, 3, 9);
    bench_copy(2048, 3, 9);
    bench_copy(4096, 3, 7);

    printf("\n--- BENCHMARK: compute-only ceiling (A/B/C resident) ---\n");
    bench_compute_only(1024, 3, 9);
    bench_compute_only(2048, 3, 9);
    bench_compute_only(4096, 3, 7);

    printf("\n--- BENCHMARK: DECODE reuse (weight [4096,4096] uploaded once, "
           "R=200 GEMMs resident, download final only) ---\n");
    bench_decode_reuse(1, 4096, 4096, 200);
    bench_decode_reuse(8, 4096, 4096, 200);

    printf("\n--- BENCHMARK: PREFILL (M=512, resident weight [4096,4096]) ---\n");
    bench_prefill(512, 4096, 4096, 30);

    printf("\n--- BENCHMARK: fully-device chained GEMM (no H2D/D2H, "
           "copy-free ceiling) ---\n");
    bench_chain_dev(1024, 16, 9);
    bench_chain_dev(2048, 8, 9);
    bench_chain_dev(4096, 8, 7);

    // The headline new path: externally-provided driver context (ILGPU interop).
    bool ctx_ok = run_ctx_tests();
    ok &= ctx_ok;

    ms_cuda_shutdown();
    printf("\n=== OVERALL: %s ===\n", ok ? "ALL PASS" : "FAILURE");
    return ok ? 0 : 2;
}
