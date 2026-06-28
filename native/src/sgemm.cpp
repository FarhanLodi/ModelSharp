/*
 * ModelSharp native fp32 GEMM — GotoBLAS/BLIS-style.
 *
 *   C[M,N] = A[M,K] * B[K,N]   (row-major, contiguous, C fully overwritten)
 *
 * PORTABILITY / DISPATCH:
 *   The library is compiled with a portable -mavx2 baseline. The high-perf
 *   AVX-512 driver (8x32 microkernel, 16 zmm accumulators, tuned for znver4)
 *   lives in an __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
 *   function so it is compiled in full but only RUN when the host supports
 *   AVX-512 (see cpu_features.h). A separate AVX2 driver (8x16 microkernel,
 *   __m256) is the correctness fallback for older CPUs. ms_sgemm_f32 dispatches
 *   ONCE on the cached runtime check. MODELSHARP_FORCE_NOAVX512 forces the AVX2
 *   path for verification.
 *
 * AVX-512 microkernel design (unchanged hot path):
 *   - MR=8 x NR=32 = 16 zmm accumulators (NR = 2 * 16-lane vectors).
 *   - BLIS cache blocking: NC -> KC -> MC; B packed once and reused.
 *   - OpenMP over the 2D tile grid (disjoint MRxNR tiles -> no write races).
 */
#include "../ms_kernels.h"
#include "cpu_features.h"
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace {

// Cache blocking parameters (tuned for znver4: 32KB L1d, 1MB L2/core, big L3).
#ifndef MS_KC
#define MS_KC 256
#endif
#ifndef MS_NC
#define MS_NC 4096
#endif
constexpr int KC = MS_KC;
constexpr int NC = MS_NC;

/* ===========================================================================
 *                      AVX-512 PATH  (MR=8, NR=32)
 * ===========================================================================*/
namespace avx512 {

constexpr int MR = 8;
constexpr int NR = 32;          // 2 zmm lanes of 16 floats

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static void pack_A(const float* A, int lda, int mc, int kc, float* Ap) {
    int mp = 0;
    for (int i = 0; i < mc; i += MR) {
        int mr = std::min(MR, mc - i);
        const float* Asrc = A + (size_t)i * lda;
        float* dst = Ap + (size_t)mp * KC * MR;
        if (mr == MR) {
            for (int p = 0; p < kc; ++p) {
                dst[0] = Asrc[0 * lda + p];
                dst[1] = Asrc[1 * lda + p];
                dst[2] = Asrc[2 * lda + p];
                dst[3] = Asrc[3 * lda + p];
                dst[4] = Asrc[4 * lda + p];
                dst[5] = Asrc[5 * lda + p];
                dst[6] = Asrc[6 * lda + p];
                dst[7] = Asrc[7 * lda + p];
                dst += MR;
            }
        } else {
            for (int p = 0; p < kc; ++p) {
                int r = 0;
                for (; r < mr; ++r) dst[r] = Asrc[r * lda + p];
                for (; r < MR; ++r) dst[r] = 0.0f;
                dst += MR;
            }
        }
        ++mp;
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static void pack_B(const float* B, int ldb, int kc, int nc, float* Bp) {
    int np = 0;
    for (int j = 0; j < nc; j += NR) {
        int nr = std::min(NR, nc - j);
        float* dst = Bp + (size_t)np * kc * NR;
        if (nr == NR) {
            for (int p = 0; p < kc; ++p) {
                const float* src = B + (size_t)p * ldb + j;
                _mm512_storeu_ps(dst,      _mm512_loadu_ps(src));
                _mm512_storeu_ps(dst + 16, _mm512_loadu_ps(src + 16));
                dst += NR;
            }
        } else {
            for (int p = 0; p < kc; ++p) {
                const float* src = B + (size_t)p * ldb + j;
                int c = 0;
                for (; c < nr; ++c) dst[c] = src[c];
                for (; c < NR; ++c) dst[c] = 0.0f;
                dst += NR;
            }
        }
        ++np;
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"), always_inline))
static inline void micro_8x32(int kc, const float* Ap, const float* Bp, float* Cbuf) {
    __m512 c0a = _mm512_setzero_ps(), c0b = _mm512_setzero_ps();
    __m512 c1a = _mm512_setzero_ps(), c1b = _mm512_setzero_ps();
    __m512 c2a = _mm512_setzero_ps(), c2b = _mm512_setzero_ps();
    __m512 c3a = _mm512_setzero_ps(), c3b = _mm512_setzero_ps();
    __m512 c4a = _mm512_setzero_ps(), c4b = _mm512_setzero_ps();
    __m512 c5a = _mm512_setzero_ps(), c5b = _mm512_setzero_ps();
    __m512 c6a = _mm512_setzero_ps(), c6b = _mm512_setzero_ps();
    __m512 c7a = _mm512_setzero_ps(), c7b = _mm512_setzero_ps();
    for (int p = 0; p < kc; ++p) {
        __m512 b0 = _mm512_load_ps(Bp);
        __m512 b1 = _mm512_load_ps(Bp + 16);
        _mm_prefetch((const char*)(Bp + NR * 8), _MM_HINT_T0);
        Bp += NR;
        __m512 a;
        a = _mm512_set1_ps(Ap[0]); c0a = _mm512_fmadd_ps(a, b0, c0a); c0b = _mm512_fmadd_ps(a, b1, c0b);
        a = _mm512_set1_ps(Ap[1]); c1a = _mm512_fmadd_ps(a, b0, c1a); c1b = _mm512_fmadd_ps(a, b1, c1b);
        a = _mm512_set1_ps(Ap[2]); c2a = _mm512_fmadd_ps(a, b0, c2a); c2b = _mm512_fmadd_ps(a, b1, c2b);
        a = _mm512_set1_ps(Ap[3]); c3a = _mm512_fmadd_ps(a, b0, c3a); c3b = _mm512_fmadd_ps(a, b1, c3b);
        a = _mm512_set1_ps(Ap[4]); c4a = _mm512_fmadd_ps(a, b0, c4a); c4b = _mm512_fmadd_ps(a, b1, c4b);
        a = _mm512_set1_ps(Ap[5]); c5a = _mm512_fmadd_ps(a, b0, c5a); c5b = _mm512_fmadd_ps(a, b1, c5b);
        a = _mm512_set1_ps(Ap[6]); c6a = _mm512_fmadd_ps(a, b0, c6a); c6b = _mm512_fmadd_ps(a, b1, c6b);
        a = _mm512_set1_ps(Ap[7]); c7a = _mm512_fmadd_ps(a, b0, c7a); c7b = _mm512_fmadd_ps(a, b1, c7b);
        Ap += MR;
    }
    _mm512_store_ps(Cbuf +  0,      c0a); _mm512_store_ps(Cbuf +  0 + 16, c0b);
    _mm512_store_ps(Cbuf + 32,      c1a); _mm512_store_ps(Cbuf + 32 + 16, c1b);
    _mm512_store_ps(Cbuf + 64,      c2a); _mm512_store_ps(Cbuf + 64 + 16, c2b);
    _mm512_store_ps(Cbuf + 96,      c3a); _mm512_store_ps(Cbuf + 96 + 16, c3b);
    _mm512_store_ps(Cbuf +128,      c4a); _mm512_store_ps(Cbuf +128 + 16, c4b);
    _mm512_store_ps(Cbuf +160,      c5a); _mm512_store_ps(Cbuf +160 + 16, c5b);
    _mm512_store_ps(Cbuf +192,      c6a); _mm512_store_ps(Cbuf +192 + 16, c6b);
    _mm512_store_ps(Cbuf +224,      c7a); _mm512_store_ps(Cbuf +224 + 16, c7b);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"), always_inline))
static inline void micro_8x32_C(int kc, const float* Ap, const float* Bp,
                                float* C, int ldc, bool first) {
    __m512 c0a = _mm512_setzero_ps(), c0b = _mm512_setzero_ps();
    __m512 c1a = _mm512_setzero_ps(), c1b = _mm512_setzero_ps();
    __m512 c2a = _mm512_setzero_ps(), c2b = _mm512_setzero_ps();
    __m512 c3a = _mm512_setzero_ps(), c3b = _mm512_setzero_ps();
    __m512 c4a = _mm512_setzero_ps(), c4b = _mm512_setzero_ps();
    __m512 c5a = _mm512_setzero_ps(), c5b = _mm512_setzero_ps();
    __m512 c6a = _mm512_setzero_ps(), c6b = _mm512_setzero_ps();
    __m512 c7a = _mm512_setzero_ps(), c7b = _mm512_setzero_ps();
    for (int p = 0; p < kc; ++p) {
        __m512 b0 = _mm512_load_ps(Bp);
        __m512 b1 = _mm512_load_ps(Bp + 16);
        _mm_prefetch((const char*)(Bp + NR * 8), _MM_HINT_T0);
        Bp += NR;
        __m512 a;
        a = _mm512_set1_ps(Ap[0]); c0a = _mm512_fmadd_ps(a, b0, c0a); c0b = _mm512_fmadd_ps(a, b1, c0b);
        a = _mm512_set1_ps(Ap[1]); c1a = _mm512_fmadd_ps(a, b0, c1a); c1b = _mm512_fmadd_ps(a, b1, c1b);
        a = _mm512_set1_ps(Ap[2]); c2a = _mm512_fmadd_ps(a, b0, c2a); c2b = _mm512_fmadd_ps(a, b1, c2b);
        a = _mm512_set1_ps(Ap[3]); c3a = _mm512_fmadd_ps(a, b0, c3a); c3b = _mm512_fmadd_ps(a, b1, c3b);
        a = _mm512_set1_ps(Ap[4]); c4a = _mm512_fmadd_ps(a, b0, c4a); c4b = _mm512_fmadd_ps(a, b1, c4b);
        a = _mm512_set1_ps(Ap[5]); c5a = _mm512_fmadd_ps(a, b0, c5a); c5b = _mm512_fmadd_ps(a, b1, c5b);
        a = _mm512_set1_ps(Ap[6]); c6a = _mm512_fmadd_ps(a, b0, c6a); c6b = _mm512_fmadd_ps(a, b1, c6b);
        a = _mm512_set1_ps(Ap[7]); c7a = _mm512_fmadd_ps(a, b0, c7a); c7b = _mm512_fmadd_ps(a, b1, c7b);
        Ap += MR;
    }
    float* r0 = C; float* r1 = C+ldc; float* r2 = C+2*ldc; float* r3 = C+3*ldc;
    float* r4 = C+4*ldc; float* r5 = C+5*ldc; float* r6 = C+6*ldc; float* r7 = C+7*ldc;
    if (first) {
        _mm512_storeu_ps(r0, c0a); _mm512_storeu_ps(r0+16, c0b);
        _mm512_storeu_ps(r1, c1a); _mm512_storeu_ps(r1+16, c1b);
        _mm512_storeu_ps(r2, c2a); _mm512_storeu_ps(r2+16, c2b);
        _mm512_storeu_ps(r3, c3a); _mm512_storeu_ps(r3+16, c3b);
        _mm512_storeu_ps(r4, c4a); _mm512_storeu_ps(r4+16, c4b);
        _mm512_storeu_ps(r5, c5a); _mm512_storeu_ps(r5+16, c5b);
        _mm512_storeu_ps(r6, c6a); _mm512_storeu_ps(r6+16, c6b);
        _mm512_storeu_ps(r7, c7a); _mm512_storeu_ps(r7+16, c7b);
    } else {
        _mm512_storeu_ps(r0, _mm512_add_ps(_mm512_loadu_ps(r0), c0a)); _mm512_storeu_ps(r0+16, _mm512_add_ps(_mm512_loadu_ps(r0+16), c0b));
        _mm512_storeu_ps(r1, _mm512_add_ps(_mm512_loadu_ps(r1), c1a)); _mm512_storeu_ps(r1+16, _mm512_add_ps(_mm512_loadu_ps(r1+16), c1b));
        _mm512_storeu_ps(r2, _mm512_add_ps(_mm512_loadu_ps(r2), c2a)); _mm512_storeu_ps(r2+16, _mm512_add_ps(_mm512_loadu_ps(r2+16), c2b));
        _mm512_storeu_ps(r3, _mm512_add_ps(_mm512_loadu_ps(r3), c3a)); _mm512_storeu_ps(r3+16, _mm512_add_ps(_mm512_loadu_ps(r3+16), c3b));
        _mm512_storeu_ps(r4, _mm512_add_ps(_mm512_loadu_ps(r4), c4a)); _mm512_storeu_ps(r4+16, _mm512_add_ps(_mm512_loadu_ps(r4+16), c4b));
        _mm512_storeu_ps(r5, _mm512_add_ps(_mm512_loadu_ps(r5), c5a)); _mm512_storeu_ps(r5+16, _mm512_add_ps(_mm512_loadu_ps(r5+16), c5b));
        _mm512_storeu_ps(r6, _mm512_add_ps(_mm512_loadu_ps(r6), c6a)); _mm512_storeu_ps(r6+16, _mm512_add_ps(_mm512_loadu_ps(r6+16), c6b));
        _mm512_storeu_ps(r7, _mm512_add_ps(_mm512_loadu_ps(r7), c7a)); _mm512_storeu_ps(r7+16, _mm512_add_ps(_mm512_loadu_ps(r7+16), c7b));
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"), always_inline))
static inline void store_C(float* C, int ldc, int mr, int nr,
                           const float* Cbuf, bool first) {
    if (first) {
        if (mr == MR && nr == NR) {
            for (int i = 0; i < MR; ++i) {
                _mm512_storeu_ps(C + (size_t)i * ldc,      _mm512_load_ps(Cbuf + i * NR));
                _mm512_storeu_ps(C + (size_t)i * ldc + 16, _mm512_load_ps(Cbuf + i * NR + 16));
            }
        } else {
            for (int i = 0; i < mr; ++i)
                for (int j = 0; j < nr; ++j)
                    C[(size_t)i * ldc + j] = Cbuf[i * NR + j];
        }
    } else {
        if (mr == MR && nr == NR) {
            for (int i = 0; i < MR; ++i) {
                float* c = C + (size_t)i * ldc;
                _mm512_storeu_ps(c,      _mm512_add_ps(_mm512_loadu_ps(c),      _mm512_load_ps(Cbuf + i * NR)));
                _mm512_storeu_ps(c + 16, _mm512_add_ps(_mm512_loadu_ps(c + 16), _mm512_load_ps(Cbuf + i * NR + 16)));
            }
        } else {
            for (int i = 0; i < mr; ++i)
                for (int j = 0; j < nr; ++j)
                    C[(size_t)i * ldc + j] += Cbuf[i * NR + j];
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static void run(const float* A, const float* B, float* C, int M, int N, int K) {
    const int lda = K, ldb = N, ldc = N;
    float* Bp = (float*)_mm_malloc(sizeof(float) * (size_t)KC * NC, 64);

    #pragma omp parallel proc_bind(spread)
    {
        float* Ap   = (float*)_mm_malloc(sizeof(float) * (size_t)MR * KC, 64);
        float* Cbuf = (float*)_mm_malloc(sizeof(float) * MR * NR, 64);

        for (int jc = 0; jc < N; jc += NC) {
            int nc = std::min(NC, N - jc);
            int npanels = (nc + NR - 1) / NR;

            for (int pc = 0; pc < K; pc += KC) {
                int kc = std::min(KC, K - pc);
                bool first = (pc == 0);

                #pragma omp for schedule(static)
                for (int jp = 0; jp < npanels; ++jp) {
                    int j = jp * NR;
                    int nr = std::min(NR, nc - j);
                    pack_B(B + (size_t)pc * ldb + jc + j, ldb, kc, nr,
                           Bp + (size_t)jp * KC * NR);
                }

                int mpanels = (M + MR - 1) / MR;
                int nthreads = omp_get_num_threads();

                if (mpanels >= nthreads) {
                    #pragma omp for schedule(dynamic)
                    for (int ip = 0; ip < mpanels; ++ip) {
                        int i = ip * MR;
                        int mr = std::min(MR, M - i);
                        pack_A(A + (size_t)i * lda + pc, lda, mr, kc, Ap);
                        for (int jpn = 0; jpn < npanels; ++jpn) {
                            int j = jpn * NR;
                            int nr = std::min(NR, nc - j);
                            const float* Bpan = Bp + (size_t)jpn * KC * NR;
                            float* Ctile = C + (size_t)i * ldc + jc + j;
                            if (mr == MR && nr == NR) {
                                micro_8x32_C(kc, Ap, Bpan, Ctile, ldc, first);
                            } else {
                                micro_8x32(kc, Ap, Bpan, Cbuf);
                                store_C(Ctile, ldc, mr, nr, Cbuf, first);
                            }
                        }
                    }
                } else {
                    long ntiles = (long)mpanels * npanels;
                    int cached_ip = -1;
                    #pragma omp for schedule(dynamic, 8)
                    for (long t = 0; t < ntiles; ++t) {
                        int ip  = (int)(t / npanels);
                        int jpn = (int)(t % npanels);
                        int i = ip * MR;
                        int mr = std::min(MR, M - i);
                        if (ip != cached_ip) {
                            pack_A(A + (size_t)i * lda + pc, lda, mr, kc, Ap);
                            cached_ip = ip;
                        }
                        int j = jpn * NR;
                        int nr = std::min(NR, nc - j);
                        const float* Bpan = Bp + (size_t)jpn * KC * NR;
                        float* Ctile = C + (size_t)i * ldc + jc + j;
                        if (mr == MR && nr == NR) {
                            micro_8x32_C(kc, Ap, Bpan, Ctile, ldc, first);
                        } else {
                            micro_8x32(kc, Ap, Bpan, Cbuf);
                            store_C(Ctile, ldc, mr, nr, Cbuf, first);
                        }
                    }
                }
            }
        }
        _mm_free(Ap);
        _mm_free(Cbuf);
    }
    _mm_free(Bp);
}

} // namespace avx512

/* ===========================================================================
 *                      AVX2 FALLBACK PATH  (MR=8, NR=16)
 * Correctness fallback for CPUs without AVX-512. Same BLIS blocking, an 8x16
 * __m256 microkernel (8 ymm accumulators). Simpler/slower but correct.
 * ===========================================================================*/
namespace avx2 {

constexpr int MR = 8;
constexpr int NR = 16;          // 2 ymm lanes of 8 floats

static void pack_A(const float* A, int lda, int mc, int kc, float* Ap) {
    int mp = 0;
    for (int i = 0; i < mc; i += MR) {
        int mr = std::min(MR, mc - i);
        const float* Asrc = A + (size_t)i * lda;
        float* dst = Ap + (size_t)mp * KC * MR;
        for (int p = 0; p < kc; ++p) {
            int r = 0;
            for (; r < mr; ++r) dst[r] = Asrc[r * lda + p];
            for (; r < MR; ++r) dst[r] = 0.0f;
            dst += MR;
        }
        ++mp;
    }
}

static void pack_B(const float* B, int ldb, int kc, int nc, float* Bp) {
    int np = 0;
    for (int j = 0; j < nc; j += NR) {
        int nr = std::min(NR, nc - j);
        float* dst = Bp + (size_t)np * kc * NR;
        for (int p = 0; p < kc; ++p) {
            const float* src = B + (size_t)p * ldb + j;
            int c = 0;
            for (; c < nr; ++c) dst[c] = src[c];
            for (; c < NR; ++c) dst[c] = 0.0f;
            dst += NR;
        }
        ++np;
    }
}

static inline void micro_8x16(int kc, const float* Ap, const float* Bp,
                              float* C, int ldc, int mr, int nr, bool first) {
    __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
    __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
    __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
    __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();
    __m256 c4a = _mm256_setzero_ps(), c4b = _mm256_setzero_ps();
    __m256 c5a = _mm256_setzero_ps(), c5b = _mm256_setzero_ps();
    __m256 c6a = _mm256_setzero_ps(), c6b = _mm256_setzero_ps();
    __m256 c7a = _mm256_setzero_ps(), c7b = _mm256_setzero_ps();
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_load_ps(Bp);
        __m256 b1 = _mm256_load_ps(Bp + 8);
        Bp += NR;
        __m256 a;
        a = _mm256_set1_ps(Ap[0]); c0a = _mm256_fmadd_ps(a, b0, c0a); c0b = _mm256_fmadd_ps(a, b1, c0b);
        a = _mm256_set1_ps(Ap[1]); c1a = _mm256_fmadd_ps(a, b0, c1a); c1b = _mm256_fmadd_ps(a, b1, c1b);
        a = _mm256_set1_ps(Ap[2]); c2a = _mm256_fmadd_ps(a, b0, c2a); c2b = _mm256_fmadd_ps(a, b1, c2b);
        a = _mm256_set1_ps(Ap[3]); c3a = _mm256_fmadd_ps(a, b0, c3a); c3b = _mm256_fmadd_ps(a, b1, c3b);
        a = _mm256_set1_ps(Ap[4]); c4a = _mm256_fmadd_ps(a, b0, c4a); c4b = _mm256_fmadd_ps(a, b1, c4b);
        a = _mm256_set1_ps(Ap[5]); c5a = _mm256_fmadd_ps(a, b0, c5a); c5b = _mm256_fmadd_ps(a, b1, c5b);
        a = _mm256_set1_ps(Ap[6]); c6a = _mm256_fmadd_ps(a, b0, c6a); c6b = _mm256_fmadd_ps(a, b1, c6b);
        a = _mm256_set1_ps(Ap[7]); c7a = _mm256_fmadd_ps(a, b0, c7a); c7b = _mm256_fmadd_ps(a, b1, c7b);
        Ap += MR;
    }
    // store via a small scratch then scatter to handle ragged edges / beta
    alignas(64) float buf[MR * NR];
    _mm256_store_ps(buf +  0,     c0a); _mm256_store_ps(buf +  0 + 8, c0b);
    _mm256_store_ps(buf + 16,     c1a); _mm256_store_ps(buf + 16 + 8, c1b);
    _mm256_store_ps(buf + 32,     c2a); _mm256_store_ps(buf + 32 + 8, c2b);
    _mm256_store_ps(buf + 48,     c3a); _mm256_store_ps(buf + 48 + 8, c3b);
    _mm256_store_ps(buf + 64,     c4a); _mm256_store_ps(buf + 64 + 8, c4b);
    _mm256_store_ps(buf + 80,     c5a); _mm256_store_ps(buf + 80 + 8, c5b);
    _mm256_store_ps(buf + 96,     c6a); _mm256_store_ps(buf + 96 + 8, c6b);
    _mm256_store_ps(buf +112,     c7a); _mm256_store_ps(buf +112 + 8, c7b);
    for (int i = 0; i < mr; ++i) {
        float* c = C + (size_t)i * ldc;
        const float* s = buf + i * NR;
        if (first) for (int j = 0; j < nr; ++j) c[j]  = s[j];
        else       for (int j = 0; j < nr; ++j) c[j] += s[j];
    }
}

static void run(const float* A, const float* B, float* C, int M, int N, int K) {
    const int lda = K, ldb = N, ldc = N;
    float* Bp = (float*)_mm_malloc(sizeof(float) * (size_t)KC * NC, 64);

    #pragma omp parallel proc_bind(spread)
    {
        float* Ap = (float*)_mm_malloc(sizeof(float) * (size_t)MR * KC, 64);

        for (int jc = 0; jc < N; jc += NC) {
            int nc = std::min(NC, N - jc);
            int npanels = (nc + NR - 1) / NR;

            for (int pc = 0; pc < K; pc += KC) {
                int kc = std::min(KC, K - pc);
                bool first = (pc == 0);

                #pragma omp for schedule(static)
                for (int jp = 0; jp < npanels; ++jp) {
                    int j = jp * NR;
                    int nr = std::min(NR, nc - j);
                    pack_B(B + (size_t)pc * ldb + jc + j, ldb, kc, nr,
                           Bp + (size_t)jp * KC * NR);
                }

                int mpanels = (M + MR - 1) / MR;
                long ntiles = (long)mpanels * npanels;
                int cached_ip = -1;
                #pragma omp for schedule(dynamic, 4)
                for (long t = 0; t < ntiles; ++t) {
                    int ip  = (int)(t / npanels);
                    int jpn = (int)(t % npanels);
                    int i = ip * MR;
                    int mr = std::min(MR, M - i);
                    if (ip != cached_ip) {
                        pack_A(A + (size_t)i * lda + pc, lda, mr, kc, Ap);
                        cached_ip = ip;
                    }
                    int j = jpn * NR;
                    int nr = std::min(NR, nc - j);
                    const float* Bpan = Bp + (size_t)jpn * KC * NR;
                    float* Ctile = C + (size_t)i * ldc + jc + j;
                    micro_8x16(kc, Ap, Bpan, Ctile, ldc, mr, nr, first);
                }
            }
        }
        _mm_free(Ap);
    }
    _mm_free(Bp);
}

} // namespace avx2

} // namespace

extern "C" void ms_sgemm_f32(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {                       // empty contraction => C = 0
        for (int i = 0; i < M; ++i)
            std::memset(C + (size_t)i * N, 0, sizeof(float) * (size_t)N);
        return;
    }
    if (mscpu::has_avx512()) avx512::run(A, B, C, M, N, K);
    else                     avx2::run(A, B, C, M, N, K);
}
