/*
 * Fused scaled-dot-product attention (FlashAttention-style online softmax).
 *
 * q:[BH,Sq,D] k:[BH,Sk,D] v:[BH,Sk,D] out:[BH,Sq,D]
 * scores = scale * q.kᵀ ; softmax over Sk (causal masks key j>query i); out = softmax.v
 *
 * We never materialize the full Sq×Sk score matrix. For each query row we stream
 * key/value tiles maintaining a running max (m) and running denominator (l) plus a
 * running, rescaled output accumulator.
 *
 * PORTABILITY: the inner SIMD helpers (dotD/axpyD/scaleD) exist in two flavors —
 * a portable AVX2 version compiled into the baseline, and an AVX-512 version in a
 * __attribute__((target("avx512f"))) function. The whole per-(bh,i) compute body
 * is compiled twice via these and dispatched ONCE at entry based on the runtime
 * CPU (mscpu::has_avx512()), so AVX-512 hosts keep full-width perf and older CPUs
 * run the correct AVX2 path.
 */
#include "../ms_kernels.h"
#include "cpu_features.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace {

constexpr int KTILE = 64;   // key columns processed per streamed tile

// ---- AVX2 (baseline) SIMD helpers -----------------------------------------
static inline float dotD_avx2(const float* a, const float* b, int D) {
    __m256 acc = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= D; d += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float r = _mm_cvtss_f32(s);
    for (; d < D; ++d) r += a[d] * b[d];
    return r;
}
static inline void axpyD_avx2(float* acc, const float* v, float p, int D) {
    __m256 vp = _mm256_set1_ps(p);
    int d = 0;
    for (; d + 8 <= D; d += 8)
        _mm256_storeu_ps(acc + d, _mm256_fmadd_ps(_mm256_loadu_ps(v + d), vp, _mm256_loadu_ps(acc + d)));
    for (; d < D; ++d) acc[d] += p * v[d];
}
static inline void scaleD_avx2(float* acc, float s, int D) {
    __m256 vs = _mm256_set1_ps(s);
    int d = 0;
    for (; d + 8 <= D; d += 8)
        _mm256_storeu_ps(acc + d, _mm256_mul_ps(_mm256_loadu_ps(acc + d), vs));
    for (; d < D; ++d) acc[d] *= s;
}

// ---- AVX-512 SIMD helpers (runtime-gated) ----------------------------------
__attribute__((target("avx512f")))
static inline float dotD_avx512(const float* a, const float* b, int D) {
    __m512 acc = _mm512_setzero_ps();
    int d = 0;
    for (; d + 16 <= D; d += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + d), _mm512_loadu_ps(b + d), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; d < D; ++d) s += a[d] * b[d];
    return s;
}
__attribute__((target("avx512f")))
static inline void axpyD_avx512(float* acc, const float* v, float p, int D) {
    __m512 vp = _mm512_set1_ps(p);
    int d = 0;
    for (; d + 16 <= D; d += 16)
        _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(_mm512_loadu_ps(v + d), vp, _mm512_loadu_ps(acc + d)));
    for (; d < D; ++d) acc[d] += p * v[d];
}
__attribute__((target("avx512f")))
static inline void scaleD_avx512(float* acc, float s, int D) {
    __m512 vs = _mm512_set1_ps(s);
    int d = 0;
    for (; d + 16 <= D; d += 16)
        _mm512_storeu_ps(acc + d, _mm512_mul_ps(_mm512_loadu_ps(acc + d), vs));
    for (; d < D; ++d) acc[d] *= s;
}

// ---- compute body, templated on the SIMD width via function pointers' enum ---
// Implemented once as a macro-free template so the AVX-512 helpers are inlined
// inside an avx512-target wrapper (their target must match the call site to
// inline). We instantiate two specializations via the ISA tag.
enum class Isa { Avx2, Avx512 };

template <Isa I>
static inline float dotD(const float* a, const float* b, int D) {
    if constexpr (I == Isa::Avx512) return dotD_avx512(a, b, D);
    else return dotD_avx2(a, b, D);
}
template <Isa I>
static inline void axpyD(float* acc, const float* v, float p, int D) {
    if constexpr (I == Isa::Avx512) axpyD_avx512(acc, v, p, D);
    else axpyD_avx2(acc, v, p, D);
}
template <Isa I>
static inline void scaleD(float* acc, float s, int D) {
    if constexpr (I == Isa::Avx512) scaleD_avx512(acc, s, D);
    else scaleD_avx2(acc, s, D);
}

template <Isa I>
static void attn_run(const float* q, const float* k, const float* v,
                     float* out, int BH, int Sq, int Sk, int D,
                     float scale, int causal) {
    const int offset = causal ? (Sk - Sq) : 0;

    #pragma omp parallel
    {
        std::vector<float> accbuf(D);
        std::vector<float> score(KTILE);
        float* acc = accbuf.data();

        #pragma omp for collapse(2) schedule(static)
        for (int bh = 0; bh < BH; ++bh) {
            for (int i = 0; i < Sq; ++i) {
                const float* qrow = q + ((size_t)bh * Sq + i) * D;
                const float* kbase = k + (size_t)bh * Sk * D;
                const float* vbase = v + (size_t)bh * Sk * D;
                float* orow = out + ((size_t)bh * Sq + i) * D;

                int kmax = Sk;
                if (causal) {
                    kmax = i + offset + 1;
                    if (kmax > Sk) kmax = Sk;
                }

                if (kmax <= 0) {
                    std::memset(orow, 0, sizeof(float) * D);
                    continue;
                }

                std::memset(acc, 0, sizeof(float) * D);
                float m = -INFINITY;
                float l = 0.f;

                for (int j0 = 0; j0 < kmax; j0 += KTILE) {
                    int jn = std::min(KTILE, kmax - j0);
                    float tilemax = -INFINITY;
                    for (int t = 0; t < jn; ++t) {
                        float s = scale * dotD<I>(qrow, kbase + (size_t)(j0 + t) * D, D);
                        score[t] = s;
                        if (s > tilemax) tilemax = s;
                    }
                    float mnew = m > tilemax ? m : tilemax;
                    float corr = (m == -INFINITY) ? 0.f : std::exp(m - mnew);
                    if (corr != 1.f) { l *= corr; scaleD<I>(acc, corr, D); }
                    for (int t = 0; t < jn; ++t) {
                        float p = std::exp(score[t] - mnew);
                        l += p;
                        axpyD<I>(acc, vbase + (size_t)(j0 + t) * D, p, D);
                    }
                    m = mnew;
                }

                float inv = (l > 0.f) ? (1.f / l) : 0.f;
                for (int d = 0; d < D; ++d) orow[d] = acc[d] * inv;
            }
        }
    }
}

} // namespace

extern "C" void ms_attention_f32(const float* q, const float* k, const float* v,
                                 float* out, int BH, int Sq, int Sk, int D,
                                 float scale, int causal) {
    if (BH <= 0 || Sq <= 0 || D <= 0) return;
    if (mscpu::has_avx512())
        attn_run<Isa::Avx512>(q, k, v, out, BH, Sq, Sk, D, scale, causal);
    else
        attn_run<Isa::Avx2>(q, k, v, out, BH, Sq, Sk, D, scale, causal);
}
