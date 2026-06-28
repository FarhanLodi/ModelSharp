/*
 * cpu_features.h — runtime x86 ISA detection + dispatch helpers for ModelSharp.
 *
 * The library is COMPILED with a portable baseline (-mavx2 -mfma -mf16c) so the
 * resulting libms_kernels.so loads and runs on any x86-64 CPU since ~2015. The
 * AVX-512 (and AVX512-VNNI) hot paths are still compiled — each lives in a
 * function carrying its own __attribute__((target(...))) — but are only EXECUTED
 * when the runtime CPU actually supports them. Detection is cached on first use.
 *
 * A debug/verification override is honored: if the environment variable
 * MODELSHARP_FORCE_NOAVX512 is set (to anything), the AVX-512 / VNNI paths are
 * reported unavailable so the AVX2/scalar fallback runs even on this host. This
 * lets the self-tests exercise (and prove parity of) the portable path.
 */
#pragma once
#include <cstdlib>

namespace mscpu {

// Force-disable AVX-512 (and VNNI) at runtime when MODELSHARP_FORCE_NOAVX512 is
// set. Evaluated once.
inline bool force_no_avx512() {
    static const bool forced = (std::getenv("MODELSHARP_FORCE_NOAVX512") != nullptr);
    return forced;
}

// True iff the running CPU supports the AVX-512 subset the kernels need
// (F + BW + VL + DQ), and the override is not set. Cached.
inline bool has_avx512() {
    static const bool ok = []() {
        if (std::getenv("MODELSHARP_FORCE_NOAVX512") != nullptr) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f")
            && __builtin_cpu_supports("avx512bw")
            && __builtin_cpu_supports("avx512vl")
            && __builtin_cpu_supports("avx512dq");
    }();
    return ok;
}

// True iff the running CPU supports AVX512-VNNI (implies the F/BW/VL/DQ subset on
// every shipping part) and the override is not set. Cached.
inline bool has_vnni() {
    static const bool ok = []() {
        if (std::getenv("MODELSHARP_FORCE_NOAVX512") != nullptr) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512vnni")
            && __builtin_cpu_supports("avx512f")
            && __builtin_cpu_supports("avx512bw")
            && __builtin_cpu_supports("avx512vl")
            && __builtin_cpu_supports("avx512dq");
    }();
    return ok;
}

// True iff the running CPU supports AVX512-VBMI (used by the fused 4-bit unpack
// fast path in qgemm_w4a8). Cached. Honors the override.
inline bool has_vbmi() {
    static const bool ok = []() {
        if (std::getenv("MODELSHARP_FORCE_NOAVX512") != nullptr) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512vbmi") && has_vnni();
    }();
    return ok;
}

} // namespace mscpu
