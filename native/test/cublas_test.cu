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
    printf("CORRECTNESS: %s\n", ok ? "ALL PASS" : "FAILURE");

    printf("\n--- BENCHMARK: copy-inclusive (H2D + GEMM + D2H) ---\n");
    bench_copy(1024, 3, 9);
    bench_copy(2048, 3, 9);
    bench_copy(4096, 3, 7);

    printf("\n--- BENCHMARK: compute-only ceiling (A/B/C resident) ---\n");
    bench_compute_only(1024, 3, 9);
    bench_compute_only(2048, 3, 9);
    bench_compute_only(4096, 3, 7);

    ms_cuda_shutdown();
    return ok ? 0 : 2;
}
