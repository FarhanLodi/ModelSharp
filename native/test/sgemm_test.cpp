// Standalone self-test + benchmark for ms_sgemm_f32.
//
//   make build/test_sgemm && ./build/test_sgemm
//
// Verifies parity against a naive triple-loop reference within relative
// tolerance, on several shapes (including non-tile-multiples), then benchmarks
// GFLOP/s for 1024^3 and 2048^3 single- and multi-threaded.

#include "../ms_kernels.h"

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

using clk = std::chrono::high_resolution_clock;

static void ref_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
}

static void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& x : v) x = d(rng);
}

// Returns true on PASS.
static bool check_shape(int M, int N, int K) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    std::vector<float> C((size_t)M * N, -123.0f), R((size_t)M * N);
    fill_random(A, 1234 + M * 7 + K);
    fill_random(B, 9876 + N * 13 + K);

    ms_sgemm_f32(A.data(), B.data(), C.data(), M, N, K);
    ref_gemm(A.data(), B.data(), R.data(), M, N, K);

    // GEMM parity metric: element-wise relative error gated by the matrix
    // magnitude scale. With -ffast-math the kernel reorders/blocks the K-sum
    // vs the naive left-to-right reference, so individual near-zero outputs can
    // suffer catastrophic cancellation (huge per-element rel error on a tiny
    // value). The meaningful test is error relative to the typical output
    // magnitude (rms), which is the standard way to verify a GEMM kernel.
    double sumsq = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i)
        sumsq += (double)R[i] * (double)R[i];
    double rms = std::sqrt(sumsq / std::max<size_t>(1, (size_t)M * N));
    double scale = rms + 1e-4;

    double maxrel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        double rel = std::fabs((double)C[i] - (double)R[i]) / scale;
        if (rel > maxrel) { maxrel = rel; worst = i; }
    }
    bool pass = maxrel < 1e-3;
    std::printf("  %-22s  M=%-5d N=%-5d K=%-5d  maxrel=%.3e  %s\n",
                pass ? "[PASS]" : "[FAIL]", M, N, K, maxrel,
                pass ? "" : "<-- MISMATCH");
    if (!pass)
        std::printf("        worst idx=%zu  got=%.6f  ref=%.6f\n",
                    worst, C[worst], R[worst]);
    return pass;
}

static double bench_gflops(int M, int N, int K, int threads) {
    omp_set_num_threads(threads);
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    fill_random(A, 42);
    fill_random(B, 7);

    // Warmup.
    ms_sgemm_f32(A.data(), B.data(), C.data(), M, N, K);

    const int runs = 5;
    std::vector<double> times;
    times.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = clk::now();
        ms_sgemm_f32(A.data(), B.data(), C.data(), M, N, K);
        auto t1 = clk::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    double median = times[runs / 2];
    double flops = 2.0 * (double)M * N * K;
    return flops / median / 1e9;
}

int main() {
    std::printf("ms_sgemm_f32 self-test  (build: %s)\n", ms_build_info());
    std::printf("AVX512=%d VNNI=%d  max_threads=%d\n\n",
                ms_has_avx512(), ms_has_vnni(), omp_get_max_threads());

    std::printf("Parity checks (rel tol 1e-3):\n");
    struct S { int M, N, K; };
    S shapes[] = {
        {1, 1, 1},
        {7, 5, 3},
        {63, 65, 129},
        {256, 256, 256},
        {512, 384, 500},
        {1024, 1024, 1024},
        {8, 4096, 4096},     // tall-skinny LLM-ish
        {1, 4096, 4096},     // GEMV
    };
    bool all_pass = true;
    for (auto s : shapes)
        all_pass &= check_shape(s.M, s.N, s.K);

    if (!all_pass) {
        std::printf("\nPARITY FAILED.\n");
        return 1;
    }
    std::printf("\nAll parity checks PASSED.\n\n");

    int hw = omp_get_max_threads();
    std::printf("Benchmark GFLOP/s (median of 5 timed runs):\n");
    std::printf("  %-12s %12s %12s\n", "shape", "1 thread", "multi-thread");
    std::printf("  %-12s %12s %12s\n", "-----", "--------", "------------");
    int sizes[] = {1024, 2048};
    for (int n : sizes) {
        double g1 = bench_gflops(n, n, n, 1);
        double gm = bench_gflops(n, n, n, hw);
        char lbl[32];
        std::snprintf(lbl, sizeof(lbl), "%d^3", n);
        std::printf("  %-12s %12.1f %12.1f\n", lbl, g1, gm);
    }
    std::printf("\n");

    // Low-M / GEMV LLM shapes: parallelism must come from the N dimension.
    std::printf("Low-M LLM shapes GFLOP/s (median of 5 timed runs):\n");
    std::printf("  %-18s %12s %12s\n", "shape", "1 thread", "multi-thread");
    std::printf("  %-18s %12s %12s\n", "-----", "--------", "------------");
    struct LM { int M, N, K; } lm[] = {
        {8, 4096, 4096},
        {1, 4096, 4096},   // GEMV
    };
    for (auto s : lm) {
        double g1 = bench_gflops(s.M, s.N, s.K, 1);
        double gm = bench_gflops(s.M, s.N, s.K, hw);
        char lbl[40];
        std::snprintf(lbl, sizeof(lbl), "M=%d N=%d K=%d", s.M, s.N, s.K);
        std::printf("  %-18s %12.1f %12.1f\n", lbl, g1, gm);
    }

    std::printf("\n(multi-thread used %d threads)\n", hw);
    return 0;
}
