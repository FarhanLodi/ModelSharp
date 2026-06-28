/*
 * qgemm_test.cpp — standalone parity + throughput self-test for the native
 * quantized matmul (ms_qgemm_nbits) and the bonus VNNI int8 microkernel.
 *
 * Parity: builds a synthetic packed n-bit weight + scales (+ optional zp),
 * computes a plain-C++ dequant-then-fp-dot reference, and checks the kernel
 * matches within relative tol 2e-2 across a grid of shapes / bits / zp modes.
 *
 * Benchmark: fp dequant path GFLOP/s (1 thread + all threads) for a Mistral-ish
 * shape, and the raw VNNI int8 GOPS ceiling.
 *
 * Build & run:
 *   cd native && make build/test_qgemm && ./build/test_qgemm
 */
#include "../ms_kernels.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Bonus VNNI symbols (not in the public header — declared here). */
extern "C" int32_t ms_vnni_i8_dot(const uint8_t* a_u8, const int8_t* b_s8, int K);
extern "C" void    ms_vnni_i8_gemv(const uint8_t* a_u8, const int8_t* b_s8, int32_t* y, int N, int K);

using clk = std::chrono::high_resolution_clock;
static double now_s() {
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

/* --------------------------------------------------------- packing helpers */

static int n_blocks_per_row(int K, int bs) { return (K + bs - 1) / bs; }
static int blob_size(int bs, int bits)     { return (bs * bits + 7) / 8; }
static int zp_row_bytes(int nbpr, int bits){ return (nbpr * bits + 7) / 8; }

/* Set the index-th bits-wide code in a packed region starting at base_byte. */
static void pack_nbit(std::vector<uint8_t>& data, int base_byte, int index, int bits, int code) {
    if (bits == 8) { data[base_byte + index] = (uint8_t)(code & 0xFF); return; }
    int byte_off = base_byte + (index >> 1);
    int shift = (index & 1) * 4;
    data[byte_off] = (uint8_t)((data[byte_off] & ~(0x0F << shift)) | ((code & 0x0F) << shift));
}
static int get_nbit(const std::vector<uint8_t>& data, int base_byte, int index, int bits) {
    if (bits == 8) return data[base_byte + index];
    int byte_off = base_byte + (index >> 1);
    int shift = (index & 1) * 4;
    return (data[byte_off] >> shift) & 0x0F;
}

/* Reference dequant-then-dot, plain C++, matching MatMulNBitsKernel.cs. */
static void ref_qgemm(const std::vector<float>& a, const std::vector<uint8_t>& bq,
                      const std::vector<float>& scales, const uint8_t* zp,
                      std::vector<float>& y, int M, int N, int K, int bits, int bs) {
    int nbpr = n_blocks_per_row(K, bs);
    int bsz  = blob_size(bs, bits);
    int brb  = nbpr * bsz;
    int zrb  = zp_row_bytes(nbpr, bits);
    float dzp = (float)(1 << (bits - 1));
    std::vector<float> w(K);
    for (int n = 0; n < N; ++n) {
        int brow = n * brb, srow = n * nbpr, zrow = n * zrb;
        for (int bk = 0; bk < nbpr; ++bk) {
            float scale = scales[srow + bk];
            float z = zp ? (float)get_nbit(
                          std::vector<uint8_t>(zp, zp + (size_t)N * zrb), zrow, bk, bits)
                        : dzp;
            int blob_base = brow + bk * bsz;
            int ks = bk * bs, ke = std::min(ks + bs, K);
            for (int k = ks; k < ke; ++k) {
                int q = get_nbit(bq, blob_base, k - ks, bits);
                w[k] = ((float)q - z) * scale;
            }
        }
        for (int m = 0; m < M; ++m) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double)a[(size_t)m * K + k] * (double)w[k];
            y[(size_t)m * N + n] = (float)s;
        }
    }
}

/* ------------------------------------------------------------ parity case */

static bool run_case(int M, int N, int K, int bits, int bs, bool asym, std::mt19937& rng) {
    int nbpr = n_blocks_per_row(K, bs);
    int bsz  = blob_size(bs, bits);
    int zrb  = zp_row_bytes(nbpr, bits);
    int qmax = (1 << bits) - 1;

    std::uniform_int_distribution<int> qd(0, qmax);
    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sd(0.005f, 0.05f);

    std::vector<float> a((size_t)M * K);
    for (auto& v : a) v = ad(rng);

    std::vector<uint8_t> bq((size_t)N * nbpr * bsz, 0);
    for (int n = 0; n < N; ++n) {
        int brow = n * nbpr * bsz;
        for (int bk = 0; bk < nbpr; ++bk) {
            int blob_base = brow + bk * bsz;
            int ks = bk * bs, ke = std::min(ks + bs, K);
            for (int k = ks; k < ke; ++k) pack_nbit(bq, blob_base, k - ks, bits, qd(rng));
        }
    }

    std::vector<float> scales((size_t)N * nbpr);
    for (auto& v : scales) v = sd(rng);

    std::vector<uint8_t> zp;
    const uint8_t* zpp = nullptr;
    if (asym) {
        zp.assign((size_t)N * zrb, 0);
        for (int n = 0; n < N; ++n)
            for (int bk = 0; bk < nbpr; ++bk)
                pack_nbit(zp, n * zrb, bk, bits, qd(rng));
        zpp = zp.data();
    }

    std::vector<float> yref((size_t)M * N), ynat((size_t)M * N);
    ref_qgemm(a, bq, scales, zpp, yref, M, N, K, bits, bs);
    ms_qgemm_nbits(a.data(), bq.data(), scales.data(), zpp, ynat.data(), M, N, K, bits, bs);

    double max_rel = 0;
    for (size_t i = 0; i < yref.size(); ++i) {
        float r = yref[i], g = ynat[i];
        double denom = std::max(1e-4f, std::fabs(r));
        double rel = std::fabs((double)g - (double)r) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    bool pass = max_rel <= 2e-2;
    printf("  [%s] M=%-2d N=%-4d K=%-4d bits=%d bs=%-3d %-5s  maxrel=%.2e  %s\n",
           pass ? "PASS" : "FAIL", M, N, K, bits, bs, asym ? "asym" : "sym",
           max_rel, pass ? "" : "<<<<");
    return pass;
}

/* --------------------------------------------------------------- benchmarks */

static void bench_fp_path(int threads_max) {
    const int M = 1, N = 4096, K = 4096, bits = 4, bs = 32;
    int nbpr = n_blocks_per_row(K, bs), bsz = blob_size(bs, bits);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> qd(0, 15);
    std::uniform_real_distribution<float> ad(-1.f, 1.f), sd(0.005f, 0.05f);

    std::vector<float> a((size_t)M * K);
    for (auto& v : a) v = ad(rng);
    std::vector<uint8_t> bq((size_t)N * nbpr * bsz, 0);
    for (int n = 0; n < N; ++n)
        for (int bk = 0; bk < nbpr; ++bk) {
            int blob_base = n * nbpr * bsz + bk * bsz, ks = bk * bs, ke = std::min(ks + bs, K);
            for (int k = ks; k < ke; ++k) pack_nbit(bq, blob_base, k - ks, bits, qd(rng));
        }
    std::vector<float> scales((size_t)N * nbpr);
    for (auto& v : scales) v = sd(rng);
    std::vector<float> y((size_t)M * N);

    const double flop = 2.0 * M * N * K;
    auto measure = [&](int th) -> double {
#ifdef _OPENMP
        omp_set_num_threads(th);
#endif
        for (int w = 0; w < 3; ++w)
            ms_qgemm_nbits(a.data(), bq.data(), scales.data(), nullptr, y.data(), M, N, K, bits, bs);
        int iters = 200;
        double t0 = now_s();
        for (int it = 0; it < iters; ++it)
            ms_qgemm_nbits(a.data(), bq.data(), scales.data(), nullptr, y.data(), M, N, K, bits, bs);
        double dt = now_s() - t0;
        return flop * iters / dt / 1e9;
    };

    double g1 = measure(1);
    double gN = measure(threads_max);
    printf("\n  fp dequant path (W4A32)  shape M=%d N=%d K=%d bits=4 bs=32\n", M, N, K);
    printf("    1 thread       : %8.2f GFLOP/s\n", g1);
    printf("    %2d threads      : %8.2f GFLOP/s  (%.1fx)\n", threads_max, gN, gN / g1);
}

static void bench_vnni(int threads_max) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> au(0, 255);
    std::uniform_int_distribution<int> bs(-128, 127);

    /* each dpbusd MAC = 2 ops (mul + add). */

    /* (a) Compute ceiling: small cache-resident weight reused -> measures raw
     *     vpdpbusd issue throughput, not memory bandwidth. */
    {
        const int K = 4096;
        std::vector<uint8_t> a((size_t)K);
        for (auto& v : a) v = (uint8_t)au(rng);
        std::vector<int8_t> b((size_t)K);
        for (auto& v : b) v = (int8_t)bs(rng);
        volatile int32_t sink = 0;
        const long reps = 4000000L;
        double t0 = now_s();
        for (long i = 0; i < reps; ++i) sink ^= ms_vnni_i8_dot(a.data(), b.data(), K);
        double dt = now_s() - t0;
        double g = 2.0 * (double)K * reps / dt / 1e9;
        printf("\n  bonus VNNI int8 microkernel (vpdpbusd)  [speed only]\n");
        printf("    compute ceiling (1 thr, K=%d cache-resident): %8.2f GINT8OP/s\n", K, g);
    }

    /* (b) Realistic gemv (weights streamed from memory), 1 vs N threads. */
    {
        const int N = 32768, K = 4096;
        std::vector<uint8_t> a((size_t)K);
        for (auto& v : a) v = (uint8_t)au(rng);
        std::vector<int8_t> b((size_t)N * K);
        for (auto& v : b) v = (int8_t)bs(rng);
        std::vector<int32_t> y((size_t)N);
        const double ops = 2.0 * (double)N * K;
        auto measure = [&](int th) -> double {
#ifdef _OPENMP
            omp_set_num_threads(th);
#endif
            for (int w = 0; w < 5; ++w) ms_vnni_i8_gemv(a.data(), b.data(), y.data(), N, K);
            int iters = 300;
            double t0 = now_s();
            for (int it = 0; it < iters; ++it) ms_vnni_i8_gemv(a.data(), b.data(), y.data(), N, K);
            double dt = now_s() - t0;
            return ops * iters / dt / 1e9;
        };
        double g1 = measure(1);
        double gN = measure(threads_max);
        printf("    gemv N=%d K=%d  1 thr : %8.2f GINT8OP/s\n", N, K, g1);
        printf("    gemv N=%d K=%d %2d thr : %8.2f GINT8OP/s  (%.1fx)\n", N, K, threads_max, gN, gN / g1);
    }
}

/* ------------------------------------------------------------------- main */

int main() {
    printf("== ms_qgemm_nbits self-test ==\n");
    printf("build: %s\n", ms_build_info());
    printf("avx512=%d vnni=%d\n", ms_has_avx512(), ms_has_vnni());

    int threads_max = 1;
#ifdef _OPENMP
    threads_max = omp_get_max_threads();
#endif

    std::mt19937 rng(2024);
    bool all = true;

    printf("\nParity (tol 2e-2):\n");
    int Ms[]   = {1, 8};
    int Ns[]   = {64, 512};
    int Ks[]   = {128, 4096};
    int Bss[]  = {32, 128};
    struct Mode { int bits; bool asym; const char* name; };
    Mode modes[] = {
        {4, false, "b4 sym"}, {4, true, "b4 asym"}, {8, false, "b8 sym"},
    };

    for (auto& md : modes) {
        for (int M : Ms)
            for (int N : Ns)
                for (int K : Ks)
                    for (int bs : Bss)
                        all &= run_case(M, N, K, md.bits, bs, md.asym, rng);
    }
    /* one extra 8-bit asymmetric case for good measure */
    all &= run_case(8, 512, 4096, 8, 128, true, rng);

    printf("\nParity result: %s\n", all ? "ALL PASS" : "FAILURES PRESENT");

    bench_fp_path(threads_max);
    bench_vnni(threads_max);

    printf("\n%s\n", all ? "OVERALL: PASS" : "OVERALL: FAIL");
    return all ? 0 : 1;
}
