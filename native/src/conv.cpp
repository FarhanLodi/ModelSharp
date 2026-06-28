/*
 * conv2d (NCHW) via im2col + internal register-tiled GEMM.
 *
 * x:[N,Cin,H,W] w:[Cout,Cin/groups,KH,KW] bias:[Cout] or NULL y:[N,Cout,Ho,Wo]
 * Ho=(H+2*padH-dilH*(KH-1)-1)/strideH+1 ; Wo likewise. Zero padding.
 *
 * SELF-CONTAINED: implements its own static gemm helper (does NOT call
 * ms_sgemm_f32). im2col packs one (n,group) into [K, Ho*Wo] (K=Cin/g*KH*KW);
 * weights for the group are [Cout/g, K]; output group is [Cout/g, Ho*Wo].
 *
 * PORTABILITY: gemm_rows has an AVX-512 (4x32 zmm) implementation in a
 * target-attributed function and a portable AVX2 (4x16 ymm) fallback; the entry
 * dispatches once on the runtime CPU (mscpu::has_avx512()).
 */
#include "../ms_kernels.h"
#include "cpu_features.h"
#include <immintrin.h>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num()  { return 0; }
#endif

namespace {

// ---- AVX-512 microkernel: rows [m0,m1) of C[M,N] = A[M,K]*B[K,N] + bias ----
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static void gemm_rows_avx512(const float* A, const float* B, const float* bias,
                             float* C, int M, int N, int K, int m0, int m1) {
    (void)M;
    constexpr int MR = 4;
    constexpr int NR = 32; // 2 x 16
    int m = m0;
    for (; m + MR <= m1; m += MR) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float b0v = bias ? bias[m + 0] : 0.f;
        float b1v = bias ? bias[m + 1] : 0.f;
        float b2v = bias ? bias[m + 2] : 0.f;
        float b3v = bias ? bias[m + 3] : 0.f;
        int n = 0;
        for (; n + NR <= N; n += NR) {
            __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            const float* bp = B + n;
            for (int kk = 0; kk < K; ++kk) {
                __m512 b0 = _mm512_loadu_ps(bp + (size_t)kk * N);
                __m512 b1 = _mm512_loadu_ps(bp + (size_t)kk * N + 16);
                __m512 va0 = _mm512_set1_ps(a0[kk]);
                __m512 va1 = _mm512_set1_ps(a1[kk]);
                __m512 va2 = _mm512_set1_ps(a2[kk]);
                __m512 va3 = _mm512_set1_ps(a3[kk]);
                c00 = _mm512_fmadd_ps(va0, b0, c00); c01 = _mm512_fmadd_ps(va0, b1, c01);
                c10 = _mm512_fmadd_ps(va1, b0, c10); c11 = _mm512_fmadd_ps(va1, b1, c11);
                c20 = _mm512_fmadd_ps(va2, b0, c20); c21 = _mm512_fmadd_ps(va2, b1, c21);
                c30 = _mm512_fmadd_ps(va3, b0, c30); c31 = _mm512_fmadd_ps(va3, b1, c31);
            }
            __m512 vb0 = _mm512_set1_ps(b0v), vb1 = _mm512_set1_ps(b1v);
            __m512 vb2 = _mm512_set1_ps(b2v), vb3 = _mm512_set1_ps(b3v);
            float* c0 = C + (size_t)(m + 0) * N + n;
            float* c1 = C + (size_t)(m + 1) * N + n;
            float* c2 = C + (size_t)(m + 2) * N + n;
            float* c3 = C + (size_t)(m + 3) * N + n;
            _mm512_storeu_ps(c0,      _mm512_add_ps(c00, vb0));
            _mm512_storeu_ps(c0 + 16, _mm512_add_ps(c01, vb0));
            _mm512_storeu_ps(c1,      _mm512_add_ps(c10, vb1));
            _mm512_storeu_ps(c1 + 16, _mm512_add_ps(c11, vb1));
            _mm512_storeu_ps(c2,      _mm512_add_ps(c20, vb2));
            _mm512_storeu_ps(c2 + 16, _mm512_add_ps(c21, vb2));
            _mm512_storeu_ps(c3,      _mm512_add_ps(c30, vb3));
            _mm512_storeu_ps(c3 + 16, _mm512_add_ps(c31, vb3));
        }
        for (; n < N; n += 16) {
            int rem = std::min(16, N - n);
            __mmask16 msk = (__mmask16)((1u << rem) - 1);
            __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
            const float* bp = B + n;
            for (int kk = 0; kk < K; ++kk) {
                __m512 b = _mm512_maskz_loadu_ps(msk, bp + (size_t)kk * N);
                c0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[kk]), b, c0);
                c1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[kk]), b, c1);
                c2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[kk]), b, c2);
                c3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[kk]), b, c3);
            }
            _mm512_mask_storeu_ps(C + (size_t)(m+0)*N + n, msk, _mm512_add_ps(c0, _mm512_set1_ps(b0v)));
            _mm512_mask_storeu_ps(C + (size_t)(m+1)*N + n, msk, _mm512_add_ps(c1, _mm512_set1_ps(b1v)));
            _mm512_mask_storeu_ps(C + (size_t)(m+2)*N + n, msk, _mm512_add_ps(c2, _mm512_set1_ps(b2v)));
            _mm512_mask_storeu_ps(C + (size_t)(m+3)*N + n, msk, _mm512_add_ps(c3, _mm512_set1_ps(b3v)));
        }
    }
    for (; m < m1; ++m) {
        const float* a = A + (size_t)m * K;
        float bv = bias ? bias[m] : 0.f;
        int n = 0;
        for (; n < N; n += 16) {
            int rem = std::min(16, N - n);
            __mmask16 msk = (__mmask16)((1u << rem) - 1);
            __m512 c = _mm512_setzero_ps();
            const float* bp = B + n;
            for (int kk = 0; kk < K; ++kk)
                c = _mm512_fmadd_ps(_mm512_set1_ps(a[kk]),
                                    _mm512_maskz_loadu_ps(msk, bp + (size_t)kk * N), c);
            _mm512_mask_storeu_ps(C + (size_t)m * N + n, msk, _mm512_add_ps(c, _mm512_set1_ps(bv)));
        }
    }
}

// ---- AVX2 fallback microkernel: 4x16 ymm (correctness path) ----------------
static void gemm_rows_avx2(const float* A, const float* B, const float* bias,
                           float* C, int M, int N, int K, int m0, int m1) {
    (void)M;
    constexpr int MR = 4;
    constexpr int NR = 16; // 2 x 8
    int m = m0;
    for (; m + MR <= m1; m += MR) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float b0v = bias ? bias[m + 0] : 0.f;
        float b1v = bias ? bias[m + 1] : 0.f;
        float b2v = bias ? bias[m + 2] : 0.f;
        float b3v = bias ? bias[m + 3] : 0.f;
        int n = 0;
        for (; n + NR <= N; n += NR) {
            __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
            const float* bp = B + n;
            for (int kk = 0; kk < K; ++kk) {
                __m256 b0 = _mm256_loadu_ps(bp + (size_t)kk * N);
                __m256 b1 = _mm256_loadu_ps(bp + (size_t)kk * N + 8);
                __m256 va0 = _mm256_set1_ps(a0[kk]);
                __m256 va1 = _mm256_set1_ps(a1[kk]);
                __m256 va2 = _mm256_set1_ps(a2[kk]);
                __m256 va3 = _mm256_set1_ps(a3[kk]);
                c00 = _mm256_fmadd_ps(va0, b0, c00); c01 = _mm256_fmadd_ps(va0, b1, c01);
                c10 = _mm256_fmadd_ps(va1, b0, c10); c11 = _mm256_fmadd_ps(va1, b1, c11);
                c20 = _mm256_fmadd_ps(va2, b0, c20); c21 = _mm256_fmadd_ps(va2, b1, c21);
                c30 = _mm256_fmadd_ps(va3, b0, c30); c31 = _mm256_fmadd_ps(va3, b1, c31);
            }
            __m256 vb0 = _mm256_set1_ps(b0v), vb1 = _mm256_set1_ps(b1v);
            __m256 vb2 = _mm256_set1_ps(b2v), vb3 = _mm256_set1_ps(b3v);
            float* c0 = C + (size_t)(m + 0) * N + n;
            float* c1 = C + (size_t)(m + 1) * N + n;
            float* c2 = C + (size_t)(m + 2) * N + n;
            float* c3 = C + (size_t)(m + 3) * N + n;
            _mm256_storeu_ps(c0,     _mm256_add_ps(c00, vb0));
            _mm256_storeu_ps(c0 + 8, _mm256_add_ps(c01, vb0));
            _mm256_storeu_ps(c1,     _mm256_add_ps(c10, vb1));
            _mm256_storeu_ps(c1 + 8, _mm256_add_ps(c11, vb1));
            _mm256_storeu_ps(c2,     _mm256_add_ps(c20, vb2));
            _mm256_storeu_ps(c2 + 8, _mm256_add_ps(c21, vb2));
            _mm256_storeu_ps(c3,     _mm256_add_ps(c30, vb3));
            _mm256_storeu_ps(c3 + 8, _mm256_add_ps(c31, vb3));
        }
        // N remainder (scalar) for the 4-row tile
        for (; n < N; ++n) {
            float s0 = b0v, s1 = b1v, s2 = b2v, s3 = b3v;
            const float* bp = B + n;
            for (int kk = 0; kk < K; ++kk) {
                float bvv = bp[(size_t)kk * N];
                s0 += a0[kk] * bvv; s1 += a1[kk] * bvv;
                s2 += a2[kk] * bvv; s3 += a3[kk] * bvv;
            }
            C[(size_t)(m+0)*N + n] = s0; C[(size_t)(m+1)*N + n] = s1;
            C[(size_t)(m+2)*N + n] = s2; C[(size_t)(m+3)*N + n] = s3;
        }
    }
    for (; m < m1; ++m) {
        const float* a = A + (size_t)m * K;
        float bv = bias ? bias[m] : 0.f;
        for (int n = 0; n < N; ++n) {
            float s = bv;
            const float* bp = B + n;
            for (int kk = 0; kk < K; ++kk) s += a[kk] * bp[(size_t)kk * N];
            C[(size_t)m * N + n] = s;
        }
    }
}

static inline void gemm_rows(const float* A, const float* B, const float* bias,
                             float* C, int M, int N, int K, int m0, int m1) {
    if (mscpu::has_avx512()) gemm_rows_avx512(A, B, bias, C, M, N, K, m0, m1);
    else                     gemm_rows_avx2(A, B, bias, C, M, N, K, m0, m1);
}

} // namespace

extern "C" void ms_conv2d_f32(const float* x, const float* w, const float* bias,
                              float* y, int N, int Cin, int H, int W,
                              int Cout, int KH, int KW,
                              int strideH, int strideW, int padH, int padW,
                              int dilH, int dilW, int groups) {
    const int Ho = (H + 2 * padH - dilH * (KH - 1) - 1) / strideH + 1;
    const int Wo = (W + 2 * padW - dilW * (KW - 1) - 1) / strideW + 1;
    if (Ho <= 0 || Wo <= 0) return;

    const int Cin_g  = Cin / groups;
    const int Cout_g = Cout / groups;
    const int K  = Cin_g * KH * KW;
    const int Np = Ho * Wo;
    const size_t wgroup = (size_t)Cout_g * K;

    int maxthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        maxthreads = omp_get_num_threads();
    }
    const int outer = N * groups;
    const bool outer_parallel = outer >= maxthreads;

    if (outer_parallel) {
        #pragma omp parallel
        {
            std::vector<float> col((size_t)K * Np);
            float* cb = col.data();
            #pragma omp for collapse(2) schedule(static)
            for (int n = 0; n < N; ++n) {
                for (int g = 0; g < groups; ++g) {
                    const float* xn = x + ((size_t)n * Cin + (size_t)g * Cin_g) * H * W;
                    for (int c = 0; c < Cin_g; ++c) {
                        const float* xc = xn + (size_t)c * H * W;
                        for (int kh = 0; kh < KH; ++kh)
                        for (int kw = 0; kw < KW; ++kw) {
                            float* dst = cb + (((size_t)c * KH + kh) * KW + kw) * Np;
                            for (int oh = 0; oh < Ho; ++oh) {
                                int ih = oh * strideH - padH + kh * dilH;
                                float* drow = dst + (size_t)oh * Wo;
                                if (ih < 0 || ih >= H) { std::memset(drow, 0, sizeof(float)*Wo); continue; }
                                const float* xrow = xc + (size_t)ih * W;
                                for (int ow = 0; ow < Wo; ++ow) {
                                    int iw = ow * strideW - padW + kw * dilW;
                                    drow[ow] = (iw < 0 || iw >= W) ? 0.f : xrow[iw];
                                }
                            }
                        }
                    }
                    const float* wg = w + (size_t)g * wgroup;
                    const float* bg = bias ? (bias + (size_t)g * Cout_g) : nullptr;
                    float* yg = y + ((size_t)n * Cout + (size_t)g * Cout_g) * Np;
                    gemm_rows(wg, cb, bg, yg, Cout_g, Np, K, 0, Cout_g);
                }
            }
        }
    } else {
        std::vector<float> col((size_t)K * Np);
        float* cb = col.data();
        for (int n = 0; n < N; ++n)
        for (int g = 0; g < groups; ++g) {
            const float* xn = x + ((size_t)n * Cin + (size_t)g * Cin_g) * H * W;
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < Cin_g; ++c) {
                const float* xc = xn + (size_t)c * H * W;
                for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    float* dst = cb + (((size_t)c * KH + kh) * KW + kw) * Np;
                    for (int oh = 0; oh < Ho; ++oh) {
                        int ih = oh * strideH - padH + kh * dilH;
                        float* drow = dst + (size_t)oh * Wo;
                        if (ih < 0 || ih >= H) { std::memset(drow, 0, sizeof(float)*Wo); continue; }
                        const float* xrow = xc + (size_t)ih * W;
                        for (int ow = 0; ow < Wo; ++ow) {
                            int iw = ow * strideW - padW + kw * dilW;
                            drow[ow] = (iw < 0 || iw >= W) ? 0.f : xrow[iw];
                        }
                    }
                }
            }
            const float* wg = w + (size_t)g * wgroup;
            const float* bg = bias ? (bias + (size_t)g * Cout_g) : nullptr;
            float* yg = y + ((size_t)n * Cout + (size_t)g * Cout_g) * Np;
            #pragma omp parallel
            {
                int nt = omp_get_num_threads();
                int tid = omp_get_thread_num();
                int blocks = (Cout_g + 3) / 4;
                int per = (blocks + nt - 1) / nt;
                int b0 = tid * per, b1 = std::min(blocks, b0 + per);
                int r0 = b0 * 4, r1 = std::min(Cout_g, b1 * 4);
                if (r0 < r1) gemm_rows(wg, cb, bg, yg, Cout_g, Np, K, r0, r1);
            }
        }
    }
}
