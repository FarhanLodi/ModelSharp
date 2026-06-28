/*
 * qgemm_w4a8_test.cpp — standalone self-test + benchmark for ms_qgemm_w4a8.
 *
 * Builds synthetic block-quantized weights (in the exact MatMulNBits layout) and
 * random fp32 activations, computes a high-precision (double) fp-dequant-then-dot
 * reference, and checks the W4A8 VNNI output against it via relative-L2 error and
 * cosine similarity (W4A8 is APPROXIMATE because activations are int8-quantized).
 *
 * Also benchmarks effective GFLOP/s (2*M*N*K/s) for a Mistral decode shape
 * (M=1) and a prefill shape (M=32), single- and multi-thread, vs the fp-dequant
 * path (ms_qgemm_nbits from qgemm.cpp's numbers, quoted in the prompt).
 *
 * Build (do NOT run make):
 *   g++ -O3 -march=native -mfma -fopenmp -funroll-loops -ffast-math -std=c++17 \
 *      native/src/qgemm_w4a8.cpp native/test/qgemm_w4a8_test.cpp -o /tmp/w4a8_test
 */
#include "../ms_kernels.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

using clk = std::chrono::high_resolution_clock;

struct QWeights {
    std::vector<uint8_t> bq;      // packed codes [N * nbpr * blob_size]
    std::vector<float>   scales;  // [N * nbpr]
    std::vector<uint8_t> zp;      // packed zero points [N * zp_row_bytes] (if asym)
    std::vector<int>     code;    // [N * K] raw unsigned codes (for reference)
    std::vector<int>     zpv;     // [N * nbpr] zero point value used per block
    int nbpr, blob_size, zp_row_bytes;
};

// Build synthetic quantized weights: pick a random fp weight, derive per-block
// scale/zp, quantize to codes, pack. Returns codes + packed blobs + the exact
// zp values so the reference dequant matches what the kernel will reconstruct.
static QWeights make_weights(int N, int K, int bits, int block_size, bool asym,
                             std::mt19937& rng) {
    QWeights w;
    int nbpr = (K + block_size - 1) / block_size;
    int blob_size = (block_size * bits + 7) / 8;
    int zp_row_bytes = (nbpr * bits + 7) / 8;
    w.nbpr = nbpr; w.blob_size = blob_size; w.zp_row_bytes = zp_row_bytes;
    w.bq.assign((size_t)N * nbpr * blob_size, 0);
    w.scales.assign((size_t)N * nbpr, 0.0f);
    w.code.assign((size_t)N * K, 0);
    w.zpv.assign((size_t)N * nbpr, 0);
    if (asym) w.zp.assign((size_t)N * zp_row_bytes, 0);

    int maxcode = (1 << bits) - 1;
    int default_zp = 1 << (bits - 1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> zpd(default_zp - 2, default_zp + 2);

    for (int n = 0; n < N; ++n) {
        for (int bk = 0; bk < nbpr; ++bk) {
            int k0 = bk * block_size, k1 = std::min(k0 + block_size, K);
            int zp = asym ? std::max(0, std::min(maxcode, zpd(rng))) : default_zp;
            w.zpv[(size_t)n * nbpr + bk] = zp;
            // random true fp weights for this block, derive a scale that maps to range
            std::vector<float> wf(k1 - k0);
            float amax = 1e-6f;
            for (int k = k0; k < k1; ++k) { wf[k - k0] = nd(rng) * 0.05f; amax = std::max(amax, std::fabs(wf[k - k0])); }
            // scale so that values span roughly the code range around zp
            float scale = amax / (float)(maxcode - default_zp); // half-range
            if (scale <= 0) scale = 1e-6f;
            w.scales[(size_t)n * nbpr + bk] = scale;
            for (int k = k0; k < k1; ++k) {
                int q = (int)lrintf(wf[k - k0] / scale) + zp;
                q = std::max(0, std::min(maxcode, q));
                w.code[(size_t)n * K + k] = q;
                // pack into bq
                int blob_base = (n * nbpr + bk) * blob_size;
                int idx = k - k0;
                if (bits == 8) {
                    w.bq[blob_base + idx] = (uint8_t)q;
                } else {
                    int byte_off = blob_base + (idx >> 1);
                    int shift = (idx & 1) * 4;
                    w.bq[byte_off] |= (uint8_t)((q & 0xF) << shift);
                }
            }
            // pack zero point (if asym)
            if (asym) {
                int zp_base = n * zp_row_bytes;
                if (bits == 8) {
                    w.zp[zp_base + bk] = (uint8_t)zp;
                } else {
                    int byte_off = zp_base + (bk >> 1);
                    int shift = (bk & 1) * 4;
                    w.zp[byte_off] |= (uint8_t)((zp & 0xF) << shift);
                }
            }
        }
    }
    return w;
}

// High-precision reference: y[m,n] = sum_k a[m,k] * (code - zp) * scale  (double).
static std::vector<double> ref_gemm(const std::vector<float>& a, const QWeights& w,
                                    int M, int N, int K, int block_size) {
    std::vector<double> y((size_t)M * N, 0.0);
    int nbpr = w.nbpr;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            double acc = 0.0;
            for (int bk = 0; bk < nbpr; ++bk) {
                int k0 = bk * block_size, k1 = std::min(k0 + block_size, K);
                double scale = w.scales[(size_t)n * nbpr + bk];
                int zp = w.zpv[(size_t)n * nbpr + bk];
                for (int k = k0; k < k1; ++k) {
                    double wv = ((double)w.code[(size_t)n * K + k] - zp) * scale;
                    acc += (double)a[(size_t)m * K + k] * wv;
                }
            }
            y[(size_t)m * N + n] = acc;
        }
    }
    return y;
}

struct Metrics { double rel_l2, cosine; };

static Metrics metrics(const std::vector<float>& y, const std::vector<double>& ref) {
    double num = 0, den = 0, dot = 0, ny = 0, nr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = (double)y[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
        dot += (double)y[i] * ref[i]; ny += (double)y[i] * y[i]; nr += ref[i] * ref[i];
    }
    Metrics m;
    m.rel_l2 = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    m.cosine = dot / (std::sqrt(ny) * std::sqrt(nr) + 1e-30);
    return m;
}

static int g_fail = 0;

static void run_case(int M, int N, int K, int bits, int block_size, bool asym,
                     std::mt19937& rng) {
    QWeights w = make_weights(N, K, bits, block_size, asym, rng);
    std::vector<float> a((size_t)M * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : a) v = nd(rng);

    std::vector<float> y((size_t)M * N, 0.0f);
    ms_qgemm_w4a8(a.data(), w.bq.data(), w.scales.data(),
                  asym ? w.zp.data() : nullptr, y.data(),
                  M, N, K, bits, block_size);

    std::vector<double> ref = ref_gemm(a, w, M, N, K, block_size);
    Metrics mt = metrics(y, ref);

    bool pass = (mt.rel_l2 < 0.03) && (mt.cosine > 0.999);
    if (!pass) g_fail++;
    printf("  M=%-3d N=%-5d K=%-5d bits=%d bs=%-3d %-4s  relL2=%.5f  cos=%.6f  %s\n",
           M, N, K, bits, block_size, asym ? "asym" : "sym",
           mt.rel_l2, mt.cosine, pass ? "PASS" : "FAIL");
}

static double bench(int M, int N, int K, int bits, int block_size, int threads,
                    std::mt19937& rng) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    QWeights w = make_weights(N, K, bits, block_size, false, rng);
    std::vector<float> a((size_t)M * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : a) v = nd(rng);
    std::vector<float> y((size_t)M * N, 0.0f);

    // warmup: settle the OpenMP team + CPU boost/cache state (the team size just
    // changed via omp_set_num_threads, so warm it generously).
    for (int it = 0; it < 30; ++it)
        ms_qgemm_w4a8(a.data(), w.bq.data(), w.scales.data(), nullptr, y.data(), M, N, K, bits, block_size);

    // Take the best of several timed batches (min time = least interference);
    // robust against scheduler/boost noise across the 6 sequential bench configs.
    int iters = (M == 1) ? 400 : 100;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = clk::now();
        for (int it = 0; it < iters; ++it)
            ms_qgemm_w4a8(a.data(), w.bq.data(), w.scales.data(), nullptr, y.data(), M, N, K, bits, block_size);
        auto t1 = clk::now();
        double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
        if (sec < best) best = sec;
    }
    double gflops = 2.0 * M * N * K / best / 1e9;
    return gflops;
}

int main(int argc, char** argv) {
    // Isolated-bench mode: "--bench M threads" runs ONE config in a fresh process
    // and prints just the GF/s. Used by the parent run to avoid cross-config
    // omp_set_num_threads / turbo-state interference (gives stable numbers).
    if (argc == 4 && std::string(argv[1]) == "--bench") {
        std::mt19937 rng(99);
        int M = atoi(argv[2]), threads = atoi(argv[3]);
        printf("%.2f\n", bench(M, 4096, 4096, 4, 32, threads, rng));
        return 0;
    }

    printf("ms_qgemm_w4a8 self-test\n");
#if defined(__AVX512VNNI__)
    printf("build: AVX512-VNNI enabled (vpdpbusd active)\n");
#elif defined(__AVX512F__)
    printf("build: AVX512 (no VNNI) -- scalar dpbusd fallback\n");
#else
    printf("build: no AVX512 -- scalar fallback\n");
#endif
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    printf("omp max threads=%d\n\n", max_threads);

    std::mt19937 rng(1234);

    printf("ACCURACY (rel-L2 < 0.03, cosine > 0.999):\n");
    int Ms[] = {1, 8, 32};
    int Ns[] = {64, 512, 4096};
    int Ks[] = {128, 4096};
    int bss[] = {32, 128};
    struct Cfg { int bits; bool asym; };
    Cfg cfgs[] = {{4, false}, {4, true}, {8, false}};

    for (auto cfg : cfgs)
        for (int M : Ms)
            for (int N : Ns)
                for (int K : Ks)
                    for (int bs : bss)
                        run_case(M, N, K, cfg.bits, bs, cfg.asym, rng);

    // Edge cases: N not a multiple of NR=8 (exercises the tail-tile column path),
    // odd M (exercises the M%4 remainder/decode path on a non-1 M), small ragged
    // shapes, and a K not a multiple of block_size at bs=128 (ragged last block).
    printf("EDGE CASES (tail tiles, odd M, ragged K):\n");
    // (N>=7 so the relative-L2 metric is over a vector, not a single near-zero
    // scalar whose relative error is meaningless; N=7,9 still hit the tail path.)
    int eM[]  = {1, 3, 5, 7, 13};
    int eN[]  = {7, 9, 65, 130, 257};
    for (auto cfg : cfgs)
        for (int M : eM)
            for (int N : eN) {
                run_case(M, N, 128,  cfg.bits, 32,  cfg.asym, rng);
                run_case(M, N, 4096, cfg.bits, 128, cfg.asym, rng);
            }
    // ragged K (last block shorter than block_size)
    run_case(1,  16,  4000, 4, 128, false, rng);
    run_case(8,  9,   320,  4, 128, true,  rng);
    run_case(32, 130, 200,  8, 128, false, rng);

    printf("\nBENCHMARK (effective GFLOP/s = 2*M*N*K/s), bits=4 bs=32, N=K=4096:\n");
    // 7800X3D = 8 physical cores + SMT. Report 1T, the per-physical-core peak
    // (8T), and full SMT (max_threads): SMT often regresses on int-port-bound code.
    // Each config runs in a FRESH child process (self-exec "--bench M T") so that
    // changing the OpenMP team size / CPU boost state across configs cannot skew
    // the numbers (in-process sequential benches were very noisy on this CPU).
    int peakT = (max_threads >= 8) ? 8 : max_threads;
    auto run_iso = [&](int M, int T) -> double {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "%s --bench %d %d", argv[0], M, T);
        FILE* p = popen(cmd, "r");
        if (!p) return -1.0;
        char buf[64] = {0};
        if (!fgets(buf, sizeof(buf), p)) { pclose(p); return -1.0; }
        pclose(p);
        return atof(buf);
    };
    double d1 = run_iso(1, 1),  dP = run_iso(1, peakT),  dM = run_iso(1, max_threads);
    double p1 = run_iso(32, 1), pP = run_iso(32, peakT), pM = run_iso(32, max_threads);

    printf("  shape            1T GF/s   %dT GF/s   %dT GF/s\n", peakT, max_threads);
    printf("  decode  M=1      %7.2f   %7.2f   %7.2f\n", d1, dP, dM);
    printf("  prefill M=32     %7.2f   %7.2f   %7.2f\n", p1, pP, pM);

    // fp-dequant path reference numbers from prompt (qgemm.cpp ms_qgemm_nbits):
    //   M=1: ~3 GF/s 1T, ~13-18 GF/s 16T.
    printf("\n  fp-dequant path (ms_qgemm_nbits, M=1): ~3 GF/s 1T, ~13-18 GF/s multiT\n");
    printf("  W4A8 decode speedup: ~%.1fx 1T, ~%.1fx multiT (vs ~3 / ~15)\n",
           d1 / 3.0, (dP > dM ? dP : dM) / 15.0);

    printf("\n%s\n", g_fail == 0 ? "ALL ACCURACY CHECKS PASSED" : "SOME ACCURACY CHECKS FAILED");
    return g_fail == 0 ? 0 : 1;
}
