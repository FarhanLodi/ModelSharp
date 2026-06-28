/* Capability/info exports. Owned by the scaffold (not a compute kernel). */
#include "../ms_kernels.h"

extern "C" const char* ms_build_info(void) {
#if defined(__clang__)
    return "ms_kernels (clang, " __DATE__ ")";
#elif defined(__GNUC__)
    return "ms_kernels (g++ " __VERSION__ ")";
#else
    return "ms_kernels (unknown compiler)";
#endif
}

extern "C" int ms_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

extern "C" int ms_has_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}
