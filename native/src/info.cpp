/* Capability/info exports. Owned by the scaffold (not a compute kernel).
 *
 * ms_has_avx512 / ms_has_vnni perform RUNTIME CPU detection (not compile-time
 * #ifdef) so the C# gate reflects the actual host: the .so is built with a
 * portable -mavx2 baseline but contains AVX-512 hot paths that are only used
 * when the running CPU supports them (see cpu_features.h). */
#include "../ms_kernels.h"
#include "cpu_features.h"

extern "C" const char* ms_build_info(void) {
#if defined(__clang__)
    return "ms_kernels (clang, " __DATE__ ")";
#elif defined(__GNUC__)
    return "ms_kernels (g++ " __VERSION__ ", portable avx2 baseline + runtime avx512 dispatch)";
#else
    return "ms_kernels (unknown compiler)";
#endif
}

extern "C" int ms_has_avx512(void) {
    return mscpu::has_avx512() ? 1 : 0;
}

extern "C" int ms_has_vnni(void) {
    return mscpu::has_vnni() ? 1 : 0;
}
