/*
 * qgemm.cpp — native quantized matmul for ModelSharp.
 *
 * Implements ms_qgemm_nbits (ONNX MatMulNBits semantics) with an AVX-512 FMA
 * dequant-and-dot path (W{4,8} x A32), and a bonus AVX512-VNNI int8 microkernel
 * (vpdpbusd) used only to measure raw integer-dot throughput.
 *
 * Semantics matched from src/ModelSharp/Cpu/Kernels/Quantize/MatMulNBitsKernel.cs:
 *   - weight is logically [N, K]
 *   - blocks of block_size along K; n_blocks_per_row = ceil(K/block_size)
 *   - blob_size = ceil(block_size*bits/8) bytes per (row, block)
 *   - bits in {4,8}; 4-bit packs 2 weights/byte, low nibble = even k
 *   - scales: fp32 [N * n_blocks_per_row], indexed [n*nbpr + b]
 *   - zero_points: packed n-bit, same layout as weights, row stride
 *       ceil(n_blocks_per_row*bits/8) bytes; NULL => symmetric zp = 1<<(bits-1)
 *   - y[m,n] = sum_k a[m,k] * ((q - zp) * scale)
 */
#include "../ms_kernels.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>

/* ------------------------------------------------------------------ helpers */

static inline int unpack_nbit(const uint8_t* data, int base_byte, int index, int bits) {
    if (bits == 8) return data[base_byte + index];
    /* bits == 4: even index -> low nibble, odd -> high nibble */
    int byte_off = base_byte + (index >> 1);
    int shift = (index & 1) * 4;
    return (data[byte_off] >> shift) & 0x0F;
}

/* Horizontal sum of an __m512. */
static inline float hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

/* ----------------------------------------------------- fp32 dequant + dot */
/*
 * For one output column n, dequantize the weight row into scratch w[K] then dot
 * against every activation row. This matches the managed kernel's structure and
 * arithmetic exactly (q, zp, scale all combine as (q - zp) * scale in fp32).
 */
extern "C" void ms_qgemm_nbits(const float* a, const uint8_t* bq, const float* scales,
                               const uint8_t* zero_points, float* y,
                               int M, int N, int K, int bits, int block_size) {
    const int n_blocks_per_row = (K + block_size - 1) / block_size;
    const int blob_size        = (block_size * bits + 7) / 8;
    const int b_row_bytes      = n_blocks_per_row * blob_size;
    const int zp_row_bytes     = (n_blocks_per_row * bits + 7) / 8;
    const float default_zp     = (float)(1 << (bits - 1));

    #pragma omp parallel
    {
        /* thread-local scratch weight row (length K) */
        float* w = (float*)_mm_malloc((size_t)K * sizeof(float), 64);

        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            const int b_row_base    = n * b_row_bytes;
            const int scale_row_base = n * n_blocks_per_row;
            const int zp_row_base   = n * zp_row_bytes;

            /* ---- dequantize weight row n into w[0..K) ---- */
            for (int bk = 0; bk < n_blocks_per_row; ++bk) {
                const float scale = scales[scale_row_base + bk];
                float zp;
                if (zero_points)
                    zp = (float)unpack_nbit(zero_points, zp_row_base, bk, bits);
                else
                    zp = default_zp;

                const int blob_base = b_row_base + bk * blob_size;
                const int k_start = bk * block_size;
                const int k_end   = (k_start + block_size < K) ? (k_start + block_size) : K;
                for (int k = k_start; k < k_end; ++k) {
                    const int in_block = k - k_start;
                    const int q = unpack_nbit(bq, blob_base, in_block, bits);
                    w[k] = ((float)q - zp) * scale;
                }
            }

            /* ---- y[m,n] = dot(a[m,:], w[:]) for every row m, AVX-512 FMA ---- */
            for (int m = 0; m < M; ++m) {
                const float* arow = a + (size_t)m * K;
                __m512 acc0 = _mm512_setzero_ps();
                __m512 acc1 = _mm512_setzero_ps();
                __m512 acc2 = _mm512_setzero_ps();
                __m512 acc3 = _mm512_setzero_ps();
                int k = 0;
                for (; k + 64 <= K; k += 64) {
                    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k),      _mm512_loadu_ps(w + k),      acc0);
                    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k + 16), _mm512_loadu_ps(w + k + 16), acc1);
                    acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k + 32), _mm512_loadu_ps(w + k + 32), acc2);
                    acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k + 48), _mm512_loadu_ps(w + k + 48), acc3);
                }
                for (; k + 16 <= K; k += 16) {
                    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k), _mm512_loadu_ps(w + k), acc0);
                }
                acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
                float sum = hsum512(acc0);
                for (; k < K; ++k) sum += arow[k] * w[k];
                y[(size_t)m * N + n] = sum;
            }
        }

        _mm_free(w);
    }
}

/* =========================================================================
 * BONUS / capability demo: AVX512-VNNI int8 microkernel (vpdpbusd).
 *
 * This is NOT part of the parity contract and is numerically approximate vs fp.
 * It measures the raw integer-dot throughput ceiling of the machine: given an
 * unsigned-int8 activation row and signed-int8 weights, it computes the int32
 * dot product using _mm512_dpbusd_epi32 (u8 x s8 -> i32, 4 MACs/lane/instr).
 *
 * ms_vnni_i8_dot:   single dot of length K (returns i32).
 * ms_vnni_i8_gemv:  N independent dots (weight [N,K] s8, act [K] u8) -> y[N] i32.
 * ========================================================================= */

extern "C" int32_t ms_vnni_i8_dot(const uint8_t* a_u8, const int8_t* b_s8, int K) {
#if defined(__AVX512VNNI__)
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();
    int k = 0;
    for (; k + 256 <= K; k += 256) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const void*)(a_u8 + k)),       _mm512_loadu_si512((const void*)(b_s8 + k)));
        acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const void*)(a_u8 + k + 64)),  _mm512_loadu_si512((const void*)(b_s8 + k + 64)));
        acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512((const void*)(a_u8 + k + 128)), _mm512_loadu_si512((const void*)(b_s8 + k + 128)));
        acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512((const void*)(a_u8 + k + 192)), _mm512_loadu_si512((const void*)(b_s8 + k + 192)));
    }
    for (; k + 64 <= K; k += 64) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const void*)(a_u8 + k)), _mm512_loadu_si512((const void*)(b_s8 + k)));
    }
    acc0 = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
    int32_t sum = _mm512_reduce_add_epi32(acc0);
    for (; k < K; ++k) sum += (int32_t)a_u8[k] * (int32_t)b_s8[k];
    return sum;
#else
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) sum += (int32_t)a_u8[k] * (int32_t)b_s8[k];
    return sum;
#endif
}

extern "C" void ms_vnni_i8_gemv(const uint8_t* a_u8, const int8_t* b_s8, int32_t* y,
                                int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        y[n] = ms_vnni_i8_dot(a_u8, b_s8 + (size_t)n * K, K);
    }
}
