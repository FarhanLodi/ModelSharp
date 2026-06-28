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
 * ACTIVATION QUANT GRANULARITY:  per (row m, K-block) — i.e. the activation is
 *   re-quantized for every weight block boundary, so each VNNI accumulation runs
 *   over exactly one block with a single matching (s_a, s_w) pair. This is more
 *   accurate than a single per-row scale (a fat outlier in one block no longer
 *   crushes the resolution of every other block) and it makes the rescale fall
 *   out naturally at block granularity. Cost is tiny (a few extra max/round per
 *   block) and the activation int8 buffer is shared across all N columns.
 *     s_a(m,bk) = max|a[m, block]| / 127 ;  aq = round(a / s_a) in [-127, 127].
 *
 * ZERO-POINT / SIGN CONVENTION:
 *   vpdpbusd needs operand A unsigned, operand B signed. We make:
 *     - weight code centered & signed:  wc = code - zp        (fits int8:
 *         4-bit code 0..15, zp 0..15  -> wc in [-15,15];
 *         8-bit code 0..255, zp 0..255 in practice |wc|<=127 for real models,
 *         clamped to [-127,127] to stay valid for vpdpbusd's signed operand)
 *     - activation offset to unsigned:  au = aq + 128  in [1, 255]
 *   Identity used (per block):
 *     sum_k aq*wc = sum_k (au-128)*wc = (sum_k au*wc) - 128 * (sum_k wc)
 *   The first term is the vpdpbusd accumulator; the second is a per-block
 *   compensation using the precomputed sum of signed weight codes. Each block's
 *   fp contribution is then  (s_a * s_w) * (dot_au_wc - 128*sum_wc).
 *
 * Parallelism: OpenMP over output columns N (each thread owns distinct n; the
 * shared int8 activation buffer is read-only after the quant phase -> no races).
 * No leaks: std::vector + _mm_malloc/_mm_free for aligned scratch.
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
/* int32 dot of unsigned-int8 au[0..len) x signed-int8 wc[0..len), len<=block. */
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

/* Vectorized unpack of a contiguous run of 4-bit codes into signed int8
 * (wc = code - zp) in TRUE linear k order. `nbytes` packed bytes -> 2*nbytes codes.
 * Uses vpermt2b to de-interleave the even (low-nibble) / odd (high-nibble) streams
 * back to linear order across lanes. nbytes must be a multiple of 32. */
static inline void w4a8_unpack4_lin(const uint8_t* src, int8_t* dst, int nbytes, int zp) {
    const __m512i lo_mask = _mm512_set1_epi8(0x0F);
    const __m512i zpv     = _mm512_set1_epi8((char)zp);
    /* permute indices to interleave two 32-byte streams (even,odd) into 64 linear
     * bytes: out[2i]=even[i], out[2i+1]=odd[i]. lo holds even in bytes 0..31,
     * hi holds odd; vpermt2b selects from {lo(0..63), hi(64..127)}. */
    const __m512i perm = _mm512_set_epi8(
        95,31,94,30,93,29,92,28,91,27,90,26,89,25,88,24,
        87,23,86,22,85,21,84,20,83,19,82,18,81,17,80,16,
        79,15,78,14,77,13,76,12,75,11,74,10,73, 9,72, 8,
        71, 7,70, 6,69, 5,68, 4,67, 3,66, 2,65, 1,64, 0);
    int byte = 0;
    for (; byte + 32 <= nbytes; byte += 32) {
        __m256i p = _mm256_loadu_si256((const __m256i*)(src + byte)); /* 32 bytes */
        __m512i packed = _mm512_castsi256_si512(p);
        __m512i lo = _mm512_and_si512(packed, lo_mask);                       /* even */
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(packed, 4), lo_mask); /* odd  */
        __m512i lin = _mm512_permutex2var_epi8(lo, perm, hi);  /* 64 linear codes */
        lin = _mm512_sub_epi8(lin, zpv);
        _mm512_storeu_si512((void*)(dst + byte * 2), lin);
    }
}
#else
static inline int32_t w4a8_dpbusd(const uint8_t* au, const int8_t* wc, int len) {
    int32_t sum = 0;
    for (int k = 0; k < len; ++k) sum += (int32_t)au[k] * (int32_t)wc[k];
    return sum;
}
#endif

extern "C" void ms_qgemm_w4a8(const float* a, const uint8_t* bq, const float* scales,
                              const uint8_t* zero_points, float* y,
                              int M, int N, int K, int bits, int block_size) {
    const int nbpr        = (K + block_size - 1) / block_size;
    const int blob_size   = (block_size * bits + 7) / 8;
    const int b_row_bytes = nbpr * blob_size;
    const int zp_row_bytes = (nbpr * bits + 7) / 8;
    const int default_zp  = 1 << (bits - 1);

    /* ---- Phase 1: quantize activations to unsigned int8 (au = aq + 128),
     * per (row m, K-block). Layout: aq_u8[m * (nbpr*block_size) + ...] padded so
     * each block starts at a block_size boundary (lets blocks read full vectors
     * without crossing block edges). Also store per-block s_a. ------------- */
    const int padK = nbpr * block_size;                 /* K rounded up to blocks */
    std::vector<uint8_t> au((size_t)M * padK, (uint8_t)128); /* zero-quant => 128 */
    std::vector<float>   sa((size_t)M * nbpr, 0.0f);

    #pragma omp parallel for schedule(static) if(M > 1)
    for (int m = 0; m < M; ++m) {
        const float* arow = a + (size_t)m * K;
        uint8_t* aurow = au.data() + (size_t)m * padK;
        float*   sarow = sa.data() + (size_t)m * nbpr;
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
            /* tail of last block (k>=K) already 128 (== aq 0) from init */
        }
    }

    /* ---- Phase 2: per output column n, build signed weight codes once
     * (wc = code - zp, padded to block boundaries), compute per-block sum_wc for
     * the zero-point compensation, then VNNI-dot against every activation row. */
    #pragma omp parallel
    {
        std::vector<int8_t> wc((size_t)padK);
        std::vector<int32_t> sum_wc((size_t)nbpr);

        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            const int b_row_base     = n * b_row_bytes;
            const int scale_row_base = n * nbpr;
            const int zp_row_base    = n * zp_row_bytes;

            /* unpack weight row n into signed codes wc[] + per-block sum_wc.
             * Fast path: 4-bit symmetric, K a multiple of 64, contiguous packed
             * row -> one SIMD pass (vpermt2b de-interleave) + SIMD block sums.
             * Otherwise scalar (handles asym, 8-bit, ragged tails). */
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
            /* Fast unpack works on whole-row 64-code granularity, independent of
             * block_size; only needs 4-bit, symmetric, K%64==0, and a contiguous
             * packed row (true when K%block_size==0 so there is no per-block pad). */
            const bool fast4 = (bits == 4) && (!zero_points) && ((K % 64) == 0)
                               && ((K % block_size) == 0);
            if (fast4) {
                w4a8_unpack4_lin(bq + b_row_base, wc.data(), K / 2, default_zp);
                /* per-block sum of signed codes via dpbusd(ones, wc).
                 * block_size>=32 multiple of 32: handle 64- and 32-chunks. */
                const __m512i ones = _mm512_set1_epi8(1);
                for (int bk = 0; bk < nbpr; ++bk) {
                    const int k0 = bk * block_size;
                    __m512i acc = _mm512_setzero_si512();
                    int k = k0;
                    for (; k + 64 <= k0 + block_size; k += 64)
                        acc = _mm512_dpbusd_epi32(acc, ones,
                                _mm512_loadu_si512((const void*)(wc.data() + k)));
                    if (k < k0 + block_size) { /* <64 tail (e.g. bs=32) */
                        int rem = (k0 + block_size) - k;
                        __mmask64 mk = ((__mmask64)1ULL << rem) - 1ULL;
                        __m512i wv = _mm512_maskz_loadu_epi8(mk, wc.data() + k);
                        acc = _mm512_dpbusd_epi32(acc, ones, wv);
                    }
                    sum_wc[bk] = _mm512_reduce_add_epi32(acc);
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
                    int s = 0;
                    int k = k0;
                    for (; k < k1; ++k) {
                        int code = w4a8_unpack(bq, blob_base, k - k0, bits);
                        int v = code - zp;
                        if (v > 127) v = 127; else if (v < -127) v = -127;
                        wc[k] = (int8_t)v;
                        s += v;
                    }
                    for (; k < k0 + block_size; ++k) wc[k] = 0; /* pad tail */
                    sum_wc[bk] = s;
                }
            }

            /* y[m,n] = sum_bk (s_a * s_w) * (dpbusd(au, wc) - 128*sum_wc) */
            for (int m = 0; m < M; ++m) {
                const uint8_t* aurow = au.data() + (size_t)m * padK;
                const float*   sarow = sa.data() + (size_t)m * nbpr;
                float acc = 0.0f;
                for (int bk = 0; bk < nbpr; ++bk) {
                    const int k0 = bk * block_size;
                    const int len = std::min(block_size, K - k0); /* but au padded; use block_size */
                    /* au tail is 128 (==aq 0) and wc tail is 0, so running full
                     * block_size is safe and gives identical result. */
                    int32_t dot = w4a8_dpbusd(aurow + k0, wc.data() + k0, block_size);
                    int32_t aq_wc = dot - 128 * sum_wc[bk];
                    float s_w = scales[scale_row_base + bk];
                    acc += sarow[bk] * s_w * (float)aq_wc;
                    (void)len;
                }
                y[(size_t)m * N + n] = acc;
            }
        }
    }
}
