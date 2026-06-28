/*
 * ModelSharp native fp32 GEMM — GotoBLAS/BLIS-style, AVX-512 tuned for znver4.
 *
 *   C[M,N] = A[M,K] * B[K,N]   (row-major, contiguous, C fully overwritten)
 *
 * Design:
 *   - Microkernel MR=8 x NR=32 = 16 zmm accumulators (NR = 2 * 16-lane vectors).
 *     Each k-step: load 2 B vectors (32 contiguous N), broadcast 8 A scalars,
 *     issue 16 FMAs. B vectors feed 8 accs each, A broadcasts feed 2 each.
 *   - Cache blocking (BLIS loops): NC -> KC -> MC. B panel (KC x NC) packed once
 *     and reused across MC blocks (L3-resident on X3D); A panel (MC x KC) packed
 *     per loop and L2-resident.
 *   - Packing zero-pads ragged edges so the microkernel always runs full MRxNR
 *     tiles; only the valid output region is written back.
 *   - OpenMP parallelism over the 2D (MR-row-panel x NR-col-panel) tile grid:
 *     distinct threads own distinct MRxNR tiles of C, so no write races. This
 *     gives parallelism in BOTH dimensions, so low-M / GEMV LLM shapes (where
 *     M/MR yields only 1-2 row panels) still saturate all cores by splitting
 *     across N. Each thread gets its own packed-A scratch and reuses it across
 *     consecutive tiles that share the same row panel.
 */
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace {

// Microkernel register tile.
constexpr int MR = 8;
constexpr int NR = 32;          // 2 zmm lanes of 16 floats

// Cache blocking parameters (tuned for znver4: 32KB L1d, 1MB L2/core, big L3).
#ifndef MS_KC
#define MS_KC 256
#endif
#ifndef MS_NC
#define MS_NC 4096
#endif
constexpr int KC = MS_KC;       // shared K dimension per block (A & B panels)
constexpr int NC = MS_NC;       // cols of B per L3 block (X3D: stays in L3)

// ---- Packing ---------------------------------------------------------------
// Pack a KC-tall, MC-wide slice of A (row-major, lda=K) into MR-row panels:
//   layout: [ mp tiles ][ k ][ mr=MR ]  (column-major within each MR strip)
// Zero-pads the trailing partial MR strip.
static void pack_A(const float* A, int lda, int mc, int kc, float* Ap) {
    int mp = 0;
    for (int i = 0; i < mc; i += MR) {
        int mr = std::min(MR, mc - i);
        const float* Asrc = A + (size_t)i * lda;
        float* dst = Ap + (size_t)mp * KC * MR;   // fixed panel stride
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

// Pack a KC-tall, NC-wide slice of B (row-major, ldb=N) into NR-col panels:
//   layout: [ np tiles ][ k ][ nr=NR ]  (row-major within each NR strip)
// Zero-pads the trailing partial NR strip.
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

// ---- Microkernel: 8x32, full MRxNR tile from packed panels -----------------
// Ap: kc * MR (col-major MR strips), Bp: kc * NR (row-major NR strips).
// Writes to a contiguous MRxNR scratch (ldc = NR) which the caller stores out.
static inline void micro_8x32(int kc, const float* Ap, const float* Bp,
                              float* Cbuf) {
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
        // Prefetch B rows ahead so the next k-steps' B is already in L1.
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

// Fused microkernel for the common full-tile (mr==MR, nr==NR) case: accumulate
// in registers and write straight to C, skipping the Cbuf bounce. `first` picks
// beta=0 (overwrite) vs beta=1 (accumulate the next KC block).
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

// Store a (possibly ragged) mr x nr block from the MRxNR scratch into C.
static inline void store_C(float* C, int ldc, int mr, int nr,
                           const float* Cbuf, bool first) {
    if (first) {
        // C overwritten (beta = 0): first KC block writes directly.
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
        // Subsequent KC blocks accumulate.
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

} // namespace

extern "C" void ms_sgemm_f32(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {                       // empty contraction => C = 0
        for (int i = 0; i < M; ++i)
            std::memset(C + (size_t)i * N, 0, sizeof(float) * (size_t)N);
        return;
    }

    const int lda = K, ldb = N, ldc = N;

    // Shared packed-B panel (one KC x NC block, reused across MC and threads).
    // Panels are stored with fixed stride KC*NR so addressing is uniform.
    float* Bp = (float*)_mm_malloc(sizeof(float) * (size_t)KC * NC, 64);

    // proc_bind(spread) pins threads across the physical cores even when the
    // caller has not set OMP_PROC_BIND/OMP_PLACES. For this FMA-bound kernel one
    // thread per physical core already saturates the FP units (SMT siblings add
    // ~1%), so spreading avoids two threads contending one core's FMA ports.
    // Falls back to the runtime default harmlessly if binding is unsupported.
    #pragma omp parallel proc_bind(spread)
    {
        // Per-thread scratch: one packed MRxKC A strip + one MRxNR C tile.
        float* Ap   = (float*)_mm_malloc(sizeof(float) * (size_t)MR * KC, 64);
        float* Cbuf = (float*)_mm_malloc(sizeof(float) * MR * NR, 64);

        // jc loop: column panels of width up to NC.
        for (int jc = 0; jc < N; jc += NC) {
            int nc = std::min(NC, N - jc);
            int npanels = (nc + NR - 1) / NR;

            // pc loop: K blocks of height up to KC.
            for (int pc = 0; pc < K; pc += KC) {
                int kc = std::min(KC, K - pc);
                bool first = (pc == 0);

                // Cooperatively pack the KCxNC B slice into NR strips (fixed
                // KC*NR stride). Implicit barrier => Bp ready for all threads.
                #pragma omp for schedule(static)
                for (int jp = 0; jp < npanels; ++jp) {
                    int j = jp * NR;
                    int nr = std::min(NR, nc - j);
                    pack_B(B + (size_t)pc * ldb + jc + j, ldb, kc, nr,
                           Bp + (size_t)jp * KC * NR);
                }

                // Compute loop. Two parallelization regimes, chosen by how much
                // parallelism the M dimension alone offers:
                //
                //   (a) Plenty of row panels (mpanels >= nthreads): parallelize
                //       over MR-row panels only. Each thread packs its A row
                //       strip ONCE and reuses it across all N panels -> best
                //       A-packing amortization and locality. This is the fast
                //       path for square / large-M GEMM.
                //
                //   (b) Few row panels (low-M / GEMV, mpanels < nthreads):
                //       parallelize over the flattened (mpanels x npanels) tile
                //       grid so the many N column panels keep every core busy.
                //       A is re-packed per row panel touched (cached so that
                //       consecutive same-row tiles on a thread reuse it). The
                //       op is memory-bandwidth-bound here, so the extra A packing
                //       is cheap relative to the bandwidth win from using all
                //       cores.
                //
                // In both regimes each thread writes disjoint MRxNR C tiles
                // (no write races) and Bp is shared read-only (L3-resident).
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
                // implicit barrier after omp for => safe to repack Bp next pc.
            }
        }

        _mm_free(Ap);
        _mm_free(Cbuf);
    }

    _mm_free(Bp);
}
