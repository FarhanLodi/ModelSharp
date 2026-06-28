/*
 * Fused scaled-dot-product attention (FlashAttention-style online softmax).
 *
 * q:[BH,Sq,D] k:[BH,Sk,D] v:[BH,Sk,D] out:[BH,Sq,D]
 * scores = scale * q.kᵀ ; softmax over Sk (causal masks key j>query i); out = softmax.v
 *
 * We never materialize the full Sq×Sk score matrix. For each query row we stream
 * key/value tiles maintaining a running max (m) and running denominator (l) plus a
 * running, rescaled output accumulator. All inner loops are AVX-512 vectorized over
 * the head dimension D.
 */
#include "../ms_kernels.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

constexpr int KTILE = 64;   // key columns processed per streamed tile

#if defined(__AVX512F__)
// dot product of two length-D vectors, AVX-512 over D.
static inline float dotD(const float* a, const float* b, int D) {
    __m512 acc = _mm512_setzero_ps();
    int d = 0;
    for (; d + 16 <= D; d += 16) {
        __m512 va = _mm512_loadu_ps(a + d);
        __m512 vb = _mm512_loadu_ps(b + d);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; d < D; ++d) s += a[d] * b[d];
    return s;
}

// acc[0..D) += p * v[0..D)
static inline void axpyD(float* acc, const float* v, float p, int D) {
    __m512 vp = _mm512_set1_ps(p);
    int d = 0;
    for (; d + 16 <= D; d += 16) {
        __m512 va = _mm512_loadu_ps(acc + d);
        __m512 vv = _mm512_loadu_ps(v + d);
        va = _mm512_fmadd_ps(vv, vp, va);
        _mm512_storeu_ps(acc + d, va);
    }
    for (; d < D; ++d) acc[d] += p * v[d];
}

// acc[0..D) *= s
static inline void scaleD(float* acc, float s, int D) {
    __m512 vs = _mm512_set1_ps(s);
    int d = 0;
    for (; d + 16 <= D; d += 16) {
        __m512 va = _mm512_loadu_ps(acc + d);
        va = _mm512_mul_ps(va, vs);
        _mm512_storeu_ps(acc + d, va);
    }
    for (; d < D; ++d) acc[d] *= s;
}
#else
static inline float dotD(const float* a, const float* b, int D) {
    float s = 0.f; for (int d = 0; d < D; ++d) s += a[d] * b[d]; return s;
}
static inline void axpyD(float* acc, const float* v, float p, int D) {
    for (int d = 0; d < D; ++d) acc[d] += p * v[d];
}
static inline void scaleD(float* acc, float s, int D) {
    for (int d = 0; d < D; ++d) acc[d] *= s;
}
#endif

} // namespace

extern "C" void ms_attention_f32(const float* q, const float* k, const float* v,
                                 float* out, int BH, int Sq, int Sk, int D,
                                 float scale, int causal) {
    if (BH <= 0 || Sq <= 0 || D <= 0) return;

    // Causal alignment: for Sq==Sk the diagonal aligns; for Sq!=Sk we right-align
    // queries to keys (common decode case), so query i sees keys [0, i + (Sk-Sq)].
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
                    kmax = i + offset + 1;        // inclusive upper bound count
                    if (kmax > Sk) kmax = Sk;
                }

                if (kmax <= 0) {                  // no visible keys -> zero row
                    std::memset(orow, 0, sizeof(float) * D);
                    continue;
                }

                std::memset(acc, 0, sizeof(float) * D);
                float m = -INFINITY;             // running max
                float l = 0.f;                    // running denom

                for (int j0 = 0; j0 < kmax; j0 += KTILE) {
                    int jn = std::min(KTILE, kmax - j0);
                    // 1) tile scores + tile max
                    float tilemax = -INFINITY;
                    for (int t = 0; t < jn; ++t) {
                        float s = scale * dotD(qrow, kbase + (size_t)(j0 + t) * D, D);
                        score[t] = s;
                        if (s > tilemax) tilemax = s;
                    }
                    // 2) new running max
                    float mnew = m > tilemax ? m : tilemax;
                    // 3) rescale existing accumulator/denom to new max
                    float corr = (m == -INFINITY) ? 0.f : std::exp(m - mnew);
                    if (corr != 1.f) { l *= corr; scaleD(acc, corr, D); }
                    // 4) accumulate this tile
                    for (int t = 0; t < jn; ++t) {
                        float p = std::exp(score[t] - mnew);
                        l += p;
                        axpyD(acc, vbase + (size_t)(j0 + t) * D, p, D);
                    }
                    m = mnew;
                }

                float inv = (l > 0.f) ? (1.f / l) : 0.f;
                for (int d = 0; d < D; ++d) orow[d] = acc[d] * inv;
            }
        }
    }
}
