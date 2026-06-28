/*
 * qgemm_w4a8.cpp — W4A8 (and W8A8) AVX512-VNNI quantized matmul for ModelSharp.
 *
 * Implements ms_qgemm_w4a8 (declared in ms_kernels.h): the llama.cpp-class
 * "keep it integer" fast path for quantized LLM inference. Instead of
 * dequantizing weights to fp32 and doing fp dot products (qgemm.cpp's
 * ms_qgemm_nbits), we quantize the activations to int8 and accumulate int32 dot
 * products with AVX512-VNNI (vpdpbusd: u8 x s8 -> i32, 4 MACs / lane / instr),
 * then scale back to fp32 per K-block. Output is APPROXIMATE (activation
 * quantization noise) — this is the speed path.
 *
 * --------------------------------------------------------------------------
 * WEIGHT LAYOUT (identical to ms_qgemm_nbits / MatMulNBitsKernel.cs):
 *   - weight logically [N, K], blocks of block_size along K
 *   - n_blocks_per_row = ceil(K/block_size)
 *   - blob_size        = ceil(block_size*bits/8) bytes per (row, block)
 *   - bits in {4,8}; 4-bit packs 2 codes/byte, low nibble = even k
 *   - scales: fp32 [N * n_blocks_per_row], indexed [n*nbpr + bk]   (s_w)
 *   - zero_points: packed n-bit (same layout as weights), row stride
 *       ceil(n_blocks_per_row*bits/8) bytes; NULL => symmetric zp = 1<<(bits-1)
 *   - dequant:  w[k] = (code - zp) * s_w
 *
 * --------------------------------------------------------------------------
 * ACTIVATION QUANT GRANULARITY:  per (row m, K-block).
 *     s_a(m,bk) = max|a[m, block]| / 127 ;  aq = round(a / s_a) in [-127, 127].
 *
 * ZERO-POINT / SIGN CONVENTION:
 *     - weight code centered & signed:  wc = code - zp   (clamped [-127,127])
 *     - activation offset to unsigned:  au = aq + 128  in [1, 255]
 *   Identity (per block):
 *     sum_k aq*wc = (sum_k au*wc) - 128 * (sum_k wc)
 *   First term = vpdpbusd accumulator; second = per-block compensation using the
 *   precomputed sum of signed weight codes. fp contribution per block:
 *     (s_a * s_w) * (dot_au_wc - 128*sum_wc).
 *
 * --------------------------------------------------------------------------
 * PERFORMANCE STRUCTURE (this rewrite):
 *   - Activations pre-quantized to u8 once (Phase 1), shared across all N tiles.
 *   - Weights B-packed per N-tile (NR=8 columns) into a per-column K-contiguous
 *     int8 layout packB[j*padK + k] (one-time pack, BLIS style). The 4-bit fast
 *     path unpacks straight into packB AND computes the per-block sum_wc
 *     compensation term in the SAME pass (no second read of the weights — this is
 *     the dominant cost at M=1 decode). packB is tiny (NR*padK) and stays in cache
 *     across all N tiles a thread owns, so the microkernel reads it hot.
 *   - Decode (M=1) microkernel: NR=8 columns, 8 int32 vector accumulators; one
 *     activation load feeds all 8 weight columns. The per-block horizontal reduce
 *     of the 8 columns is fused into one 8-wide transpose-reduce (replaces 8
 *     separate _mm512_reduce_add), then scaled in fp32 vector lanes.
 *   - Prefill (M>1) microkernel: MR=4 x NR=8 register tile (held as two NR=4
 *     halves to fit 32 zmm). Each weight load is reused across MR=4 activation
 *     rows, each activation load across NR=8 columns (raises arithmetic intensity).
 *   - bs=32 PAIRING: two adjacent K-blocks (bk, bk+1) occupy the low/high 256-bit
 *     halves of one zmm, so a single full-width vpdpbusd covers both blocks; their
 *     dots are recovered by reducing the two halves separately. This doubles VNNI
 *     utilization vs. the half-width masked path that bs=32 would otherwise force.
 *   - OpenMP over N tiles (disjoint columns -> no races). 64-byte aligned scratch
 *     via _mm_malloc/_mm_free wrapped in RAII (AlignedBuf) -> no leaks.
 */
#include "../ms_kernels.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static inline int w4a8_unpack(const uint8_t* data, int base_byte, int index, int bits) {
    if (bits == 8) return data[base_byte + index];
    int byte_off = base_byte + (index >> 1);
    int shift = (index & 1) * 4;
    return (data[byte_off] >> shift) & 0x0F;
}

#if defined(__AVX512VNNI__)

/* Reduce 8 __m512i int32 accumulators to an 8-lane __m256i of horizontal sums.
 * Standard 8x8 reduce: pairwise hadd-style folding using shuffles. Each input
 * holds 16 int32 partial products; output lane j = sum of all 16 in acc[j]. */
static inline __m256i w4a8_reduce8(const __m512i* acc) {
    /* First fold each 512 -> 256 by adding high/low 256 halves. */
    __m256i r[8];
    for (int j = 0; j < 8; ++j) {
        __m256i lo = _mm512_castsi512_si256(acc[j]);
        __m256i hi = _mm512_extracti64x4_epi64(acc[j], 1);
        r[j] = _mm256_add_epi32(lo, hi);              /* 8 int32 each */
    }
    /* hadd pairs: (0,1),(2,3),(4,5),(6,7) -> 4x 256 each with 8 int32 */
    __m256i a01 = _mm256_hadd_epi32(r[0], r[1]);
    __m256i a23 = _mm256_hadd_epi32(r[2], r[3]);
    __m256i a45 = _mm256_hadd_epi32(r[4], r[5]);
    __m256i a67 = _mm256_hadd_epi32(r[6], r[7]);
    __m256i b0 = _mm256_hadd_epi32(a01, a23);  /* lanes interleave across 128 */
    __m256i b1 = _mm256_hadd_epi32(a45, a67);
    /* b0 holds sums for cols 0..3 split across its two 128-bit halves; combine. */
    __m256i lo = _mm256_permute2x128_si256(b0, b1, 0x20);
    __m256i hi = _mm256_permute2x128_si256(b0, b1, 0x31);
    return _mm256_add_epi32(lo, hi);  /* lane j = sum for column j (0..7) */
}

/* Horizontal sum of one __m256i (8 int32). */
static inline int32_t w4a8_hsum256(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1,0,3,2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1)));
    return _mm_cvtsi128_si32(s);
}

/* Horizontal-reduce 8 __m256i (each 8 int32) -> __m256i lane j = sum of col j. */
static inline __m256i w4a8_hsum8x256(const __m256i* r) {
    __m256i a01 = _mm256_hadd_epi32(r[0], r[1]);
    __m256i a23 = _mm256_hadd_epi32(r[2], r[3]);
    __m256i a45 = _mm256_hadd_epi32(r[4], r[5]);
    __m256i a67 = _mm256_hadd_epi32(r[6], r[7]);
    __m256i b0 = _mm256_hadd_epi32(a01, a23);
    __m256i b1 = _mm256_hadd_epi32(a45, a67);
    __m256i lo = _mm256_permute2x128_si256(b0, b1, 0x20);
    __m256i hi = _mm256_permute2x128_si256(b0, b1, 0x31);
    return _mm256_add_epi32(lo, hi);
}

/* Paired-block reduce of 8 accumulators: lanes 0-7 of each acc[j] belong to
 * block bk (-> *plo lane j), lanes 8-15 to block bk+1 (-> *phi lane j). The two
 * 256-bit halves are reduced separately (NOT summed), recovering both blocks'
 * dots from a single full-width vpdpbusd. */
static inline void w4a8_reduce8_pair(const __m512i* acc, __m256i* plo, __m256i* phi) {
    __m256i lo[8], hi[8];
    for (int j = 0; j < 8; ++j) {
        lo[j] = _mm512_castsi512_si256(acc[j]);
        hi[j] = _mm512_extracti64x4_epi64(acc[j], 1);
    }
    *plo = w4a8_hsum8x256(lo);
    *phi = w4a8_hsum8x256(hi);
}

/* Paired-block reduce of 4 accumulators -> two __m128i (4 lanes each). */
static inline void w4a8_reduce4_pair(__m512i a0, __m512i a1, __m512i a2, __m512i a3,
                                     __m128i* plo, __m128i* phi) {
    __m256i lo[8] = { _mm512_castsi512_si256(a0), _mm512_castsi512_si256(a1),
                      _mm512_castsi512_si256(a2), _mm512_castsi512_si256(a3),
                      _mm256_setzero_si256(), _mm256_setzero_si256(),
                      _mm256_setzero_si256(), _mm256_setzero_si256() };
    __m256i hi[8] = { _mm512_extracti64x4_epi64(a0,1), _mm512_extracti64x4_epi64(a1,1),
                      _mm512_extracti64x4_epi64(a2,1), _mm512_extracti64x4_epi64(a3,1),
                      _mm256_setzero_si256(), _mm256_setzero_si256(),
                      _mm256_setzero_si256(), _mm256_setzero_si256() };
    *plo = _mm256_castsi256_si128(w4a8_hsum8x256(lo));
    *phi = _mm256_castsi256_si128(w4a8_hsum8x256(hi));
}

/* Reduce 4 __m512i int32 accumulators to a 4-lane __m128i of horizontal sums. */
static inline __m128i w4a8_reduce4(__m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    __m256i r0 = _mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti64x4_epi64(a0, 1));
    __m256i r1 = _mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti64x4_epi64(a1, 1));
    __m256i r2 = _mm256_add_epi32(_mm512_castsi512_si256(a2), _mm512_extracti64x4_epi64(a2, 1));
    __m256i r3 = _mm256_add_epi32(_mm512_castsi512_si256(a3), _mm512_extracti64x4_epi64(a3, 1));
    __m256i h01 = _mm256_hadd_epi32(r0, r1);
    __m256i h23 = _mm256_hadd_epi32(r2, r3);
    __m256i b = _mm256_hadd_epi32(h01, h23);   /* lanes: cols split across 128 halves */
    __m128i lo = _mm256_castsi256_si128(b);
    __m128i hi = _mm256_extracti128_si256(b, 1);
    return _mm_add_epi32(lo, hi);              /* lane j = sum for column j (0..3) */
}

/* Vectorized unpack of a contiguous run of 4-bit codes into signed int8
 * (wc = code - zp) in TRUE linear k order. nbytes packed -> 2*nbytes codes.
 * nbytes must be a multiple of 32. */
static inline void w4a8_unpack4_lin(const uint8_t* src, int8_t* dst, int nbytes, int zp) {
    const __m512i lo_mask = _mm512_set1_epi8(0x0F);
    const __m512i zpv     = _mm512_set1_epi8((char)zp);
    const __m512i perm = _mm512_set_epi8(
        95,31,94,30,93,29,92,28,91,27,90,26,89,25,88,24,
        87,23,86,22,85,21,84,20,83,19,82,18,81,17,80,16,
        79,15,78,14,77,13,76,12,75,11,74,10,73, 9,72, 8,
        71, 7,70, 6,69, 5,68, 4,67, 3,66, 2,65, 1,64, 0);
    int byte = 0;
    for (; byte + 32 <= nbytes; byte += 32) {
        __m256i p = _mm256_loadu_si256((const __m256i*)(src + byte));
        __m512i packed = _mm512_castsi256_si512(p);
        __m512i lo = _mm512_and_si512(packed, lo_mask);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(packed, 4), lo_mask);
        __m512i lin = _mm512_permutex2var_epi8(lo, perm, hi);
        lin = _mm512_sub_epi8(lin, zpv);
        _mm512_storeu_si512((void*)(dst + byte * 2), lin);
    }
}

static inline int32_t w4a8_dpbusd(const uint8_t* au, const int8_t* wc, int len) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    int k = 0;
    for (; k + 128 <= len; k += 128) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const void*)(au + k)),
                                          _mm512_loadu_si512((const void*)(wc + k)));
        acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const void*)(au + k + 64)),
                                          _mm512_loadu_si512((const void*)(wc + k + 64)));
    }
    for (; k + 64 <= len; k += 64) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const void*)(au + k)),
                                          _mm512_loadu_si512((const void*)(wc + k)));
    }
    acc0 = _mm512_add_epi32(acc0, acc1);
    int32_t sum = _mm512_reduce_add_epi32(acc0);
    for (; k < len; ++k) sum += (int32_t)au[k] * (int32_t)wc[k];
    return sum;
}
#else
static inline int32_t w4a8_dpbusd(const uint8_t* au, const int8_t* wc, int len) {
    int32_t sum = 0;
    for (int k = 0; k < len; ++k) sum += (int32_t)au[k] * (int32_t)wc[k];
    return sum;
}
#endif

/* ---- aligned scratch helpers (no leaks) ---- */
template <typename T>
struct AlignedBuf {
    T* p = nullptr;
    AlignedBuf() = default;
    explicit AlignedBuf(size_t n) { if (n) p = (T*)_mm_malloc(n * sizeof(T), 64); }
    ~AlignedBuf() { if (p) _mm_free(p); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    T* get() { return p; }
};

extern "C" void ms_qgemm_w4a8(const float* a, const uint8_t* bq, const float* scales,
                              const uint8_t* zero_points, float* y,
                              int M, int N, int K, int bits, int block_size) {
    const int nbpr        = (K + block_size - 1) / block_size;
    const int blob_size   = (block_size * bits + 7) / 8;
    const int b_row_bytes = nbpr * blob_size;
    const int zp_row_bytes = (nbpr * bits + 7) / 8;
    const int default_zp  = 1 << (bits - 1);
    const int padK        = nbpr * block_size;       /* K rounded up to blocks */

    /* ---- Phase 1: quantize activations to unsigned int8 (au = aq + 128),
     * per (row m, K-block). Layout au[m*padK + k]; tail (k>=K) stays 128 (aq 0). */
    AlignedBuf<uint8_t> au_buf((size_t)M * padK);
    AlignedBuf<float>   sa_buf((size_t)M * nbpr);
    uint8_t* au = au_buf.get();
    float*   sa = sa_buf.get();
    std::memset(au, 128, (size_t)M * padK);

    #pragma omp parallel for schedule(static) if(M > 1)
    for (int m = 0; m < M; ++m) {
        const float* arow = a + (size_t)m * K;
        uint8_t* aurow = au + (size_t)m * padK;
        float*   sarow = sa + (size_t)m * nbpr;
        for (int bk = 0; bk < nbpr; ++bk) {
            const int k0 = bk * block_size;
            const int k1 = std::min(k0 + block_size, K);
            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) { float v = std::fabs(arow[k]); if (v > amax) amax = v; }
            float s = amax / 127.0f;
            sarow[bk] = s;
            float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
            for (int k = k0; k < k1; ++k) {
                int q = (int)lrintf(arow[k] * inv);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                aurow[k] = (uint8_t)(q + 128);
            }
        }
    }

    /* Register-block widths. NR weight columns per pack/microkernel tile. */
    const int NR = 8;

#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
    const bool can_fast_unpack = (bits == 4) && (!zero_points)
                                 && ((K % 64) == 0) && ((K % block_size) == 0);
#else
    const bool can_fast_unpack = false;
#endif

    /* ---- Phase 2: parallel over N tiles. Each thread packs NR weight columns
     * into a per-column K-contiguous int8 layout packB[j*padK + k] (one-time
     * B-pack), computes per-(bk,nr) sum_wc + tile scales, then runs the
     * register-blocked microkernel across all M rows reusing the pack.
     *
     * bs==32 PAIRING: two adjacent K-blocks (bk, bk+1) occupy the low/high
     * 256-bit halves of one zmm, so a single vpdpbusd covers both blocks at full
     * width; their dots are recovered by reducing the two halves separately. This
     * doubles VNNI utilization vs. the half-width masked path at bs=32. */
    const bool bs32 = (block_size == 32);

    #pragma omp parallel
    {
        AlignedBuf<int8_t>  packB_buf((size_t)NR * padK);   /* [j][padK] */
        AlignedBuf<int32_t> sumwc_buf((size_t)NR * nbpr);   /* [bk][nr]  */
        AlignedBuf<float>   packS_buf((size_t)NR * nbpr);   /* [bk][nr]  */
        int8_t*  packB = packB_buf.get();
        int32_t* sumwc = sumwc_buf.get();
        float*   packS = packS_buf.get();

        #pragma omp for schedule(static)
        for (int n0 = 0; n0 < N; n0 += NR) {
            const int nr_cnt = std::min(NR, N - n0);

            /* ---- pack the nr_cnt weight columns ---- */
            for (int j = 0; j < nr_cnt; ++j) {
                const int n = n0 + j;
                const int b_row_base  = n * b_row_bytes;
                const int zp_row_base = n * zp_row_bytes;
                const float* scol = scales + (size_t)n * nbpr;
                for (int bk = 0; bk < nbpr; ++bk) packS[bk * NR + j] = scol[bk];
                int8_t* dcol = packB + (size_t)j * padK;   /* K-contiguous column */
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
                if (can_fast_unpack) {
                    /* Fused unpack + per-block sum_wc in ONE pass over the weights
                     * (avoids a second full read for the compensation term — the
                     * dominant cost at M=1 decode). Each unpacked 64-code zmm covers
                     * 2 blocks at bs=32 (lo/hi halves) or half a block at bs=128. */
                    const __m512i lo_mask = _mm512_set1_epi8(0x0F);
                    const __m512i zpv     = _mm512_set1_epi8((char)default_zp);
                    const __m512i ones    = _mm512_set1_epi8(1);
                    const __m512i perm = _mm512_set_epi8(
                        95,31,94,30,93,29,92,28,91,27,90,26,89,25,88,24,
                        87,23,86,22,85,21,84,20,83,19,82,18,81,17,80,16,
                        79,15,78,14,77,13,76,12,75,11,74,10,73, 9,72, 8,
                        71, 7,70, 6,69, 5,68, 4,67, 3,66, 2,65, 1,64, 0);
                    const uint8_t* src = bq + b_row_base;
                    if (bs32) {
                        int bk = 0;
                        for (int byte = 0; byte + 32 <= K/2; byte += 32, bk += 2) {
                            __m256i p = _mm256_loadu_si256((const __m256i*)(src + byte));
                            __m512i packed = _mm512_castsi256_si512(p);
                            __m512i loN = _mm512_and_si512(packed, lo_mask);
                            __m512i hiN = _mm512_and_si512(_mm512_srli_epi16(packed,4), lo_mask);
                            __m512i lin = _mm512_permutex2var_epi8(loN, perm, hiN);
                            lin = _mm512_sub_epi8(lin, zpv);
                            _mm512_storeu_si512((void*)(dcol + bk*32), lin);
                            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, lin);
                            __m256i alo = _mm512_castsi512_si256(acc);
                            __m256i ahi = _mm512_extracti64x4_epi64(acc, 1);
                            sumwc[bk*NR + j]     = w4a8_hsum256(alo);
                            sumwc[(bk+1)*NR + j] = w4a8_hsum256(ahi);
                        }
                    } else {
                        /* bs=128: accumulate two 64-code halves per block */
                        const int half_per_blk = block_size / 64;  /* =2 */
                        int bk = 0, sub = 0; __m512i acc = _mm512_setzero_si512();
                        for (int byte = 0; byte + 32 <= K/2; byte += 32) {
                            __m256i p = _mm256_loadu_si256((const __m256i*)(src + byte));
                            __m512i packed = _mm512_castsi256_si512(p);
                            __m512i loN = _mm512_and_si512(packed, lo_mask);
                            __m512i hiN = _mm512_and_si512(_mm512_srli_epi16(packed,4), lo_mask);
                            __m512i lin = _mm512_permutex2var_epi8(loN, perm, hiN);
                            lin = _mm512_sub_epi8(lin, zpv);
                            _mm512_storeu_si512((void*)(dcol + byte*2), lin);
                            acc = _mm512_dpbusd_epi32(acc, ones, lin);
                            if (++sub == half_per_blk) {
                                sumwc[bk*NR + j] = _mm512_reduce_add_epi32(acc);
                                acc = _mm512_setzero_si512(); sub = 0; ++bk;
                            }
                        }
                    }
                } else
#endif
                {
                    for (int bk = 0; bk < nbpr; ++bk) {
                        int zp = zero_points ? w4a8_unpack(zero_points, zp_row_base, bk, bits)
                                             : default_zp;
                        const int blob_base = b_row_base + bk * blob_size;
                        const int k0 = bk * block_size;
                        const int k1 = std::min(k0 + block_size, K);
                        int s = 0, k = k0;
                        for (; k < k1; ++k) {
                            int v = w4a8_unpack(bq, blob_base, k - k0, bits) - zp;
                            if (v > 127) v = 127; else if (v < -127) v = -127;
                            dcol[k] = (int8_t)v;
                            s += v;
                        }
                        for (; k < k0 + block_size; ++k) dcol[k] = 0;
                        sumwc[bk * NR + j] = s;
                    }
                }
            }
            /* zero unused columns of the tile so full-NR kernels read clean data */
            for (int j = nr_cnt; j < NR; ++j) {
                std::memset(packB + (size_t)j * padK, 0, padK);
                for (int bk = 0; bk < nbpr; ++bk) {
                    sumwc[bk * NR + j] = 0;
                    packS[bk * NR + j] = 0.0f;
                }
            }

#if defined(__AVX512VNNI__)
            const bool full = (nr_cnt == NR);
            /* ============ DECODE (M=1) and remainder rows: NR=8 columns ======= */
            /* ============ PREFILL: MR=4 x NR=8 register tile ================== */
            int m = 0;
            if (full) {
                for (; m + 4 <= M; m += 4) {
                    const uint8_t* A[4] = { au + (size_t)(m+0)*padK, au + (size_t)(m+1)*padK,
                                            au + (size_t)(m+2)*padK, au + (size_t)(m+3)*padK };
                    const float* S[4] = { sa + (size_t)(m+0)*nbpr, sa + (size_t)(m+1)*nbpr,
                                          sa + (size_t)(m+2)*nbpr, sa + (size_t)(m+3)*nbpr };
                    /* per-row output accumulators split lo(cols0-3)/hi(cols4-7) */
                    __m128 y0l=_mm_setzero_ps(),y0h=_mm_setzero_ps();
                    __m128 y1l=_mm_setzero_ps(),y1h=_mm_setzero_ps();
                    __m128 y2l=_mm_setzero_ps(),y2h=_mm_setzero_ps();
                    __m128 y3l=_mm_setzero_ps(),y3h=_mm_setzero_ps();
                    for (int half = 0; half < 2; ++half) {
                        __m128 *yA = (half==0)?&y0l:&y0h, *yB=(half==0)?&y1l:&y1h,
                               *yC = (half==0)?&y2l:&y2h, *yD=(half==0)?&y3l:&y3h;
                        const int coff = half*4;
                        if (bs32) {
                            for (int bk = 0; bk < nbpr; bk += 2) {
                                const int k0 = bk * 32;
                                const bool pair = (bk + 1 < nbpr);
                                __mmask64 mk = pair ? ~(__mmask64)0 : (((__mmask64)1ULL<<32)-1);
                                const int8_t* wb = packB + (size_t)coff*padK + k0;
                                __m512i w0=_mm512_maskz_loadu_epi8(mk, wb+0*padK);
                                __m512i w1=_mm512_maskz_loadu_epi8(mk, wb+1*padK);
                                __m512i w2=_mm512_maskz_loadu_epi8(mk, wb+2*padK);
                                __m512i w3=_mm512_maskz_loadu_epi8(mk, wb+3*padK);
                                __m512i v0=_mm512_maskz_loadu_epi8(mk, A[0]+k0);
                                __m512i v1=_mm512_maskz_loadu_epi8(mk, A[1]+k0);
                                __m512i v2=_mm512_maskz_loadu_epi8(mk, A[2]+k0);
                                __m512i v3=_mm512_maskz_loadu_epi8(mk, A[3]+k0);
                                __m128i l0,h0,l1,h1,l2,h2,l3,h3;
                                w4a8_reduce4_pair(_mm512_dpbusd_epi32(_mm512_setzero_si512(),v0,w0),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v0,w1),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v0,w2),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v0,w3),&l0,&h0);
                                w4a8_reduce4_pair(_mm512_dpbusd_epi32(_mm512_setzero_si512(),v1,w0),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v1,w1),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v1,w2),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v1,w3),&l1,&h1);
                                w4a8_reduce4_pair(_mm512_dpbusd_epi32(_mm512_setzero_si512(),v2,w0),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v2,w1),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v2,w2),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v2,w3),&l2,&h2);
                                w4a8_reduce4_pair(_mm512_dpbusd_epi32(_mm512_setzero_si512(),v3,w0),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v3,w1),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v3,w2),
                                                  _mm512_dpbusd_epi32(_mm512_setzero_si512(),v3,w3),&l3,&h3);
                                __m128i swc0=_mm_slli_epi32(_mm_loadu_si128((const __m128i*)(sumwc+bk*NR+coff)),7);
                                __m128 sw0=_mm_loadu_ps(packS+bk*NR+coff);
                                *yA=_mm_add_ps(*yA,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(l0,swc0)),_mm_mul_ps(_mm_set1_ps(S[0][bk]),sw0)));
                                *yB=_mm_add_ps(*yB,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(l1,swc0)),_mm_mul_ps(_mm_set1_ps(S[1][bk]),sw0)));
                                *yC=_mm_add_ps(*yC,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(l2,swc0)),_mm_mul_ps(_mm_set1_ps(S[2][bk]),sw0)));
                                *yD=_mm_add_ps(*yD,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(l3,swc0)),_mm_mul_ps(_mm_set1_ps(S[3][bk]),sw0)));
                                if (pair) {
                                    int b1=bk+1;
                                    __m128i swc1=_mm_slli_epi32(_mm_loadu_si128((const __m128i*)(sumwc+b1*NR+coff)),7);
                                    __m128 sw1=_mm_loadu_ps(packS+b1*NR+coff);
                                    *yA=_mm_add_ps(*yA,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(h0,swc1)),_mm_mul_ps(_mm_set1_ps(S[0][b1]),sw1)));
                                    *yB=_mm_add_ps(*yB,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(h1,swc1)),_mm_mul_ps(_mm_set1_ps(S[1][b1]),sw1)));
                                    *yC=_mm_add_ps(*yC,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(h2,swc1)),_mm_mul_ps(_mm_set1_ps(S[2][b1]),sw1)));
                                    *yD=_mm_add_ps(*yD,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(h3,swc1)),_mm_mul_ps(_mm_set1_ps(S[3][b1]),sw1)));
                                }
                            }
                        } else {
                            for (int bk = 0; bk < nbpr; ++bk) {
                                const int k0 = bk * block_size;
                                const int8_t* wb = packB + (size_t)coff*padK + k0;
                                __m512i c00=_mm512_setzero_si512(),c01=_mm512_setzero_si512(),
                                        c02=_mm512_setzero_si512(),c03=_mm512_setzero_si512();
                                __m512i c10=_mm512_setzero_si512(),c11=_mm512_setzero_si512(),
                                        c12=_mm512_setzero_si512(),c13=_mm512_setzero_si512();
                                __m512i c20=_mm512_setzero_si512(),c21=_mm512_setzero_si512(),
                                        c22=_mm512_setzero_si512(),c23=_mm512_setzero_si512();
                                __m512i c30=_mm512_setzero_si512(),c31=_mm512_setzero_si512(),
                                        c32=_mm512_setzero_si512(),c33=_mm512_setzero_si512();
                                for (int k = 0; k < block_size; k += 64) {
                                    __m512i w0=_mm512_loadu_si512((const void*)(wb+0*padK+k));
                                    __m512i w1=_mm512_loadu_si512((const void*)(wb+1*padK+k));
                                    __m512i w2=_mm512_loadu_si512((const void*)(wb+2*padK+k));
                                    __m512i w3=_mm512_loadu_si512((const void*)(wb+3*padK+k));
                                    __m512i v0=_mm512_loadu_si512((const void*)(A[0]+k0+k));
                                    __m512i v1=_mm512_loadu_si512((const void*)(A[1]+k0+k));
                                    __m512i v2=_mm512_loadu_si512((const void*)(A[2]+k0+k));
                                    __m512i v3=_mm512_loadu_si512((const void*)(A[3]+k0+k));
                                    c00=_mm512_dpbusd_epi32(c00,v0,w0);c01=_mm512_dpbusd_epi32(c01,v0,w1);
                                    c02=_mm512_dpbusd_epi32(c02,v0,w2);c03=_mm512_dpbusd_epi32(c03,v0,w3);
                                    c10=_mm512_dpbusd_epi32(c10,v1,w0);c11=_mm512_dpbusd_epi32(c11,v1,w1);
                                    c12=_mm512_dpbusd_epi32(c12,v1,w2);c13=_mm512_dpbusd_epi32(c13,v1,w3);
                                    c20=_mm512_dpbusd_epi32(c20,v2,w0);c21=_mm512_dpbusd_epi32(c21,v2,w1);
                                    c22=_mm512_dpbusd_epi32(c22,v2,w2);c23=_mm512_dpbusd_epi32(c23,v2,w3);
                                    c30=_mm512_dpbusd_epi32(c30,v3,w0);c31=_mm512_dpbusd_epi32(c31,v3,w1);
                                    c32=_mm512_dpbusd_epi32(c32,v3,w2);c33=_mm512_dpbusd_epi32(c33,v3,w3);
                                }
                                __m128i d0=w4a8_reduce4(c00,c01,c02,c03);
                                __m128i d1=w4a8_reduce4(c10,c11,c12,c13);
                                __m128i d2=w4a8_reduce4(c20,c21,c22,c23);
                                __m128i d3=w4a8_reduce4(c30,c31,c32,c33);
                                __m128i swc=_mm_slli_epi32(_mm_loadu_si128((const __m128i*)(sumwc+bk*NR+coff)),7);
                                __m128 sw=_mm_loadu_ps(packS+bk*NR+coff);
                                *yA=_mm_add_ps(*yA,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(d0,swc)),_mm_mul_ps(_mm_set1_ps(S[0][bk]),sw)));
                                *yB=_mm_add_ps(*yB,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(d1,swc)),_mm_mul_ps(_mm_set1_ps(S[1][bk]),sw)));
                                *yC=_mm_add_ps(*yC,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(d2,swc)),_mm_mul_ps(_mm_set1_ps(S[2][bk]),sw)));
                                *yD=_mm_add_ps(*yD,_mm_mul_ps(_mm_cvtepi32_ps(_mm_sub_epi32(d3,swc)),_mm_mul_ps(_mm_set1_ps(S[3][bk]),sw)));
                            }
                        }
                    }
                    _mm_storeu_ps(y+(size_t)(m+0)*N+n0,   y0l);
                    _mm_storeu_ps(y+(size_t)(m+0)*N+n0+4, y0h);
                    _mm_storeu_ps(y+(size_t)(m+1)*N+n0,   y1l);
                    _mm_storeu_ps(y+(size_t)(m+1)*N+n0+4, y1h);
                    _mm_storeu_ps(y+(size_t)(m+2)*N+n0,   y2l);
                    _mm_storeu_ps(y+(size_t)(m+2)*N+n0+4, y2h);
                    _mm_storeu_ps(y+(size_t)(m+3)*N+n0,   y3l);
                    _mm_storeu_ps(y+(size_t)(m+3)*N+n0+4, y3h);
                }
            }
            /* ---- remaining rows (M=1 decode, and M%4 tail): NR=8, 1 row ---- */
            for (; m < M && full; ++m) {
                const uint8_t* aurow = au + (size_t)m * padK;
                const float*   sarow = sa + (size_t)m * nbpr;
                __m256 yacc = _mm256_setzero_ps();
                if (bs32) {
                    for (int bk = 0; bk < nbpr; bk += 2) {
                        const int k0 = bk * 32;
                        const bool pair = (bk + 1 < nbpr);
                        __mmask64 mk = pair ? ~(__mmask64)0 : (((__mmask64)1ULL<<32)-1);
                        __m512i av = _mm512_maskz_loadu_epi8(mk, aurow + k0);
                        __m512i acc[NR];
                        const int8_t* wcol = packB + k0;
                        for (int j = 0; j < NR; ++j)
                            acc[j] = _mm512_dpbusd_epi32(_mm512_setzero_si512(), av,
                                        _mm512_maskz_loadu_epi8(mk, wcol + (size_t)j*padK));
                        __m256i lo, hi;
                        w4a8_reduce8_pair(acc, &lo, &hi);
                        /* block bk (lo) */
                        {
                            __m256i swc=_mm256_loadu_si256((const __m256i*)(sumwc+bk*NR));
                            __m256i aq=_mm256_sub_epi32(lo,_mm256_slli_epi32(swc,7));
                            __m256 sw=_mm256_loadu_ps(packS+bk*NR);
                            yacc=_mm256_add_ps(yacc,_mm256_mul_ps(_mm256_cvtepi32_ps(aq),
                                  _mm256_mul_ps(_mm256_set1_ps(sarow[bk]),sw)));
                        }
                        if (pair) {
                            int bk1=bk+1;
                            __m256i swc=_mm256_loadu_si256((const __m256i*)(sumwc+bk1*NR));
                            __m256i aq=_mm256_sub_epi32(hi,_mm256_slli_epi32(swc,7));
                            __m256 sw=_mm256_loadu_ps(packS+bk1*NR);
                            yacc=_mm256_add_ps(yacc,_mm256_mul_ps(_mm256_cvtepi32_ps(aq),
                                  _mm256_mul_ps(_mm256_set1_ps(sarow[bk1]),sw)));
                        }
                    }
                } else {
                    for (int bk = 0; bk < nbpr; ++bk) {
                        const int k0 = bk * block_size;
                        __m512i acc[NR];
                        for (int j = 0; j < NR; ++j) acc[j] = _mm512_setzero_si512();
                        for (int k = 0; k < block_size; k += 64) {
                            __m512i av = _mm512_loadu_si512((const void*)(aurow + k0 + k));
                            for (int j = 0; j < NR; ++j)
                                acc[j] = _mm512_dpbusd_epi32(acc[j], av,
                                    _mm512_loadu_si512((const void*)(packB + (size_t)j*padK + k0 + k)));
                        }
                        __m256i dot = w4a8_reduce8(acc);
                        __m256i swc = _mm256_loadu_si256((const __m256i*)(sumwc + bk*NR));
                        __m256i aq  = _mm256_sub_epi32(dot, _mm256_slli_epi32(swc, 7));
                        __m256 sw   = _mm256_loadu_ps(packS + bk*NR);
                        yacc = _mm256_add_ps(yacc, _mm256_mul_ps(_mm256_cvtepi32_ps(aq),
                                _mm256_mul_ps(_mm256_set1_ps(sarow[bk]), sw)));
                    }
                }
                _mm256_storeu_ps(y + (size_t)m * N + n0, yacc);
            }
            if (!full) {
#else
            {
#endif
                /* tail tile (<NR columns) or no-AVX512: per-column scalar reduce */
                for (int mm = 0; mm < M; ++mm) {
                    const uint8_t* aurow = au + (size_t)mm * padK;
                    const float*   sarow = sa + (size_t)mm * nbpr;
                    for (int j = 0; j < nr_cnt; ++j) {
                        const int n = n0 + j;
                        const int8_t* col = packB + (size_t)j * padK;
                        float acc = 0.0f;
                        for (int bk = 0; bk < nbpr; ++bk) {
                            const int k0 = bk * block_size;
                            int32_t dot = w4a8_dpbusd(aurow + k0, col + k0, block_size);
                            int32_t aqwc = dot - 128 * sumwc[bk * NR + j];
                            acc += sarow[bk] * scales[n * nbpr + bk] * (float)aqwc;
                        }
                        y[(size_t)mm * N + n] = acc;
                    }
                }
            }
        }
    }
}
