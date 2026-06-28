/*
 * qgemm.cpp — native quantized matmul for ModelSharp.
 *
 * Implements ms_qgemm_nbits (ONNX MatMulNBits semantics) with a dequant-and-dot
 * path (W{4,8} x A32), plus a bonus AVX512-VNNI int8 microkernel (vpdpbusd) used
 * only to measure raw integer-dot throughput.
 *
 * PORTABILITY: the dot product over the dequantized weight row has an AVX-512 and
 * a portable AVX2 implementation, each in a target-attributed function; the row
 * loop dispatches once on mscpu::has_avx512(). The VNNI int8 path runs the
 * vpdpbusd kernel when the host supports VNNI and a scalar fallback otherwise
 * (gated on mscpu::has_vnni()).
 *
 * Semantics matched from src/ModelSharp/Cpu/Kernels/Quantize/MatMulNBitsKernel.cs:
 *   - weight is logically [N, K]; blocks of block_size along K
 *   - blob_size = ceil(block_size*bits/8) bytes per (row, block)
 *   - bits in {4,8}; 4-bit packs 2 weights/byte, low nibble = even k
 *   - scales fp32 [N * nbpr], zero_points packed n-bit or NULL (sym zp=1<<(bits-1))
 *   - y[m,n] = sum_k a[m,k] * ((q - zp) * scale)
 */
#include "../ms_kernels.h"
#include "cpu_features.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>

static inline int unpack_nbit(const uint8_t* data, int base_byte, int index, int bits) {
    if (bits == 8) return data[base_byte + index];
    int byte_off = base_byte + (index >> 1);
    int shift = (index & 1) * 4;
    return (data[byte_off] >> shift) & 0x0F;
}

/* ---- dequantized-row dot products: AVX-512 and AVX2 variants ---- */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static float dot_row_avx512(const float* arow, const float* w, int K) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 64 <= K; k += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k),      _mm512_loadu_ps(w + k),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k + 16), _mm512_loadu_ps(w + k + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k + 32), _mm512_loadu_ps(w + k + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k + 48), _mm512_loadu_ps(w + k + 48), acc3);
    }
    for (; k + 16 <= K; k += 16)
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(arow + k), _mm512_loadu_ps(w + k), acc0);
    acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    float sum = _mm512_reduce_add_ps(acc0);
    for (; k < K; ++k) sum += arow[k] * w[k];
    return sum;
}

static float dot_row_avx2(const float* arow, const float* w, int K) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(arow + k),      _mm256_loadu_ps(w + k),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(arow + k + 8),  _mm256_loadu_ps(w + k + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(arow + k + 16), _mm256_loadu_ps(w + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(arow + k + 24), _mm256_loadu_ps(w + k + 24), acc3);
    }
    for (; k + 8 <= K; k += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(arow + k), _mm256_loadu_ps(w + k), acc0);
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float sum = _mm_cvtss_f32(s);
    for (; k < K; ++k) sum += arow[k] * w[k];
    return sum;
}

extern "C" void ms_qgemm_nbits(const float* a, const uint8_t* bq, const float* scales,
                               const uint8_t* zero_points, float* y,
                               int M, int N, int K, int bits, int block_size) {
    const int n_blocks_per_row = (K + block_size - 1) / block_size;
    const int blob_size        = (block_size * bits + 7) / 8;
    const int b_row_bytes      = n_blocks_per_row * blob_size;
    const int zp_row_bytes     = (n_blocks_per_row * bits + 7) / 8;
    const float default_zp     = (float)(1 << (bits - 1));
    const bool use_avx512      = mscpu::has_avx512();

    #pragma omp parallel
    {
        float* w = (float*)_mm_malloc((size_t)K * sizeof(float), 64);

        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            const int b_row_base     = n * b_row_bytes;
            const int scale_row_base = n * n_blocks_per_row;
            const int zp_row_base    = n * zp_row_bytes;

            for (int bk = 0; bk < n_blocks_per_row; ++bk) {
                const float scale = scales[scale_row_base + bk];
                float zp = zero_points ? (float)unpack_nbit(zero_points, zp_row_base, bk, bits)
                                       : default_zp;
                const int blob_base = b_row_base + bk * blob_size;
                const int k_start = bk * block_size;
                const int k_end   = (k_start + block_size < K) ? (k_start + block_size) : K;
                for (int k = k_start; k < k_end; ++k) {
                    const int q = unpack_nbit(bq, blob_base, k - k_start, bits);
                    w[k] = ((float)q - zp) * scale;
                }
            }

            for (int m = 0; m < M; ++m) {
                const float* arow = a + (size_t)m * K;
                y[(size_t)m * N + n] = use_avx512 ? dot_row_avx512(arow, w, K)
                                                  : dot_row_avx2(arow, w, K);
            }
        }
        _mm_free(w);
    }
}

/* =========================================================================
 * BONUS / capability demo: AVX512-VNNI int8 microkernel (vpdpbusd).
 * Runtime-gated: uses VNNI when the host has it, scalar otherwise.
 * ========================================================================= */

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))
static int32_t vnni_i8_dot_avx512(const uint8_t* a_u8, const int8_t* b_s8, int K) {
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
    for (; k + 64 <= K; k += 64)
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const void*)(a_u8 + k)), _mm512_loadu_si512((const void*)(b_s8 + k)));
    acc0 = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
    int32_t sum = _mm512_reduce_add_epi32(acc0);
    for (; k < K; ++k) sum += (int32_t)a_u8[k] * (int32_t)b_s8[k];
    return sum;
}

static int32_t vnni_i8_dot_scalar(const uint8_t* a_u8, const int8_t* b_s8, int K) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) sum += (int32_t)a_u8[k] * (int32_t)b_s8[k];
    return sum;
}

extern "C" int32_t ms_vnni_i8_dot(const uint8_t* a_u8, const int8_t* b_s8, int K) {
    return mscpu::has_vnni() ? vnni_i8_dot_avx512(a_u8, b_s8, K)
                             : vnni_i8_dot_scalar(a_u8, b_s8, K);
}

extern "C" void ms_vnni_i8_gemv(const uint8_t* a_u8, const int8_t* b_s8, int32_t* y,
                                int N, int K) {
    const bool vnni = mscpu::has_vnni();
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const int8_t* b = b_s8 + (size_t)n * K;
        y[n] = vnni ? vnni_i8_dot_avx512(a_u8, b, K) : vnni_i8_dot_scalar(a_u8, b, K);
    }
}
