/*
 * ModelSharp native kernels — C ABI.
 *
 * This header is the *integration contract*: every kernel .cpp under native/src
 * implements one or more of these exported symbols, and the C# P/Invoke layer
 * (bench/ModelSharp.Bench/NativeKernels.cs and, optionally, the core engine)
 * declares matching [DllImport]s. Keep signatures stable; add new functions
 * rather than changing existing ones.
 *
 * Conventions:
 *   - All float buffers are 32-bit IEEE, row-major, contiguous.
 *   - Pointers are caller-owned; kernels never allocate caller-visible memory.
 *   - Output buffers are fully overwritten (no read-modify-write unless stated).
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- capability / info ---------------------------------------------------- */
const char* ms_build_info(void);   /* compiler + flags string */
int         ms_has_avx512(void);   /* 1 if built with AVX-512 */
int         ms_has_vnni(void);     /* 1 if built with AVX512-VNNI */

/* ---- fp32 GEMM ------------------------------------------------------------ */
/* C[M,N] = A[M,K] * B[K,N], all row-major. C is overwritten. */
void ms_sgemm_f32(const float* A, const float* B, float* C,
                  int M, int N, int K);

/* ---- quantized matmul (ONNX MatMulNBits semantics) ------------------------ */
/* a:      [M,K] fp32 activations
 * bq:     packed n-bit weight codes for a logical [N,K] weight (row n, col k);
 *         blocks of `block_size` along K, `bits` in {4,8}; 4-bit packs 2/byte
 *         (low nibble = even k). blob layout matches MatMulNBitsKernel.cs.
 * scales: fp32, one per (row n, block) -> N * ceil(K/block_size) entries
 * zero_points: packed same `bits`, one per (row,block), or NULL for symmetric
 *         (zp = 1<<(bits-1)).
 * y:      [M,N] fp32, y[m,n] = sum_k a[m,k] * (deq w[n,k]). Overwritten.
 */
void ms_qgemm_nbits(const float* a, const uint8_t* bq, const float* scales,
                    const uint8_t* zero_points, float* y,
                    int M, int N, int K, int bits, int block_size);

/* ---- W4A8 VNNI quant matmul (llama.cpp-class path) ------------------------ */
/* Same weight layout as ms_qgemm_nbits, but activations are quantized to int8
 * internally and consumed by AVX512-VNNI (vpdpbusd) directly, instead of
 * dequantizing weights to fp32. Output y[M,N] fp32 is APPROXIMATE (activation
 * quantization error) — the speed path for quantized LLM inference. */
void ms_qgemm_w4a8(const float* a, const uint8_t* bq, const float* scales,
                   const uint8_t* zero_points, float* y,
                   int M, int N, int K, int bits, int block_size);

/* ---- fused scaled-dot-product attention ----------------------------------- */
/* q: [BH, Sq, D], k: [BH, Sk, D], v: [BH, Sk, D], out: [BH, Sq, D].
 * BH = batch*heads folded. scores scaled by `scale`. causal!=0 masks j>i. */
void ms_attention_f32(const float* q, const float* k, const float* v, float* out,
                      int BH, int Sq, int Sk, int D, float scale, int causal);

/* ---- conv2d (NCHW), im2col + sgemm ---------------------------------------- */
/* x:[N,Cin,H,W] w:[Cout,Cin/groups,KH,KW] bias:[Cout] or NULL. y:[N,Cout,Ho,Wo]. */
void ms_conv2d_f32(const float* x, const float* w, const float* bias, float* y,
                   int N, int Cin, int H, int W,
                   int Cout, int KH, int KW,
                   int strideH, int strideW, int padH, int padW,
                   int dilH, int dilW, int groups);

#ifdef __cplusplus
}
#endif
