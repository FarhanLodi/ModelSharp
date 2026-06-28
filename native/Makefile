# ModelSharp native kernels build.
#
#   make           -> build/libms_kernels.so   (the shared lib C# P/Invokes)
#   make tests     -> build/test_* standalone self-tests (parity + GFLOP/s)
#   make run-tests -> build and run every self-test
#   make clean
#
# PORTABLE baseline: -mavx2 -mfma -mf16c is supported on essentially every
# x86-64 CPU since ~2015, so libms_kernels.so loads and runs anywhere without
# SIGILL. The AVX-512 / AVX512-VNNI hot paths are still compiled in (each kernel
# carries its own __attribute__((target(...))) ) and are selected at RUNTIME via
# __builtin_cpu_supports (see src/cpu_features.h), preserving full AVX-512 perf on
# hosts that have it (e.g. znver4) while staying safe on older CPUs.

CXX      ?= g++
CXXFLAGS ?= -O3 -mavx2 -mfma -mf16c -fopenmp -fPIC -std=c++17 -Wall -funroll-loops -ffast-math
LDFLAGS  ?= -shared -fopenmp

SRC      := $(wildcard src/*.cpp)
OBJ      := $(patsubst src/%.cpp,build/%.o,$(SRC))
LIB      := build/libms_kernels.so

TEST_SRC := $(wildcard test/*_test.cpp)
TEST_BIN := $(patsubst test/%_test.cpp,build/test_%,$(TEST_SRC))

.PHONY: all tests run-tests clean
all: $(LIB)

$(LIB): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^
	@echo "built $@"

build/%.o: src/%.cpp ms_kernels.h | build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Each self-test is linked against the kernel objects directly (no .so needed).
build/test_%: test/%_test.cpp $(OBJ) | build
	$(CXX) $(CXXFLAGS) -o $@ $< $(filter build/%.o,$(OBJ))

tests: $(TEST_BIN)

run-tests: tests
	@for t in $(TEST_BIN); do echo "=== $$t ==="; $$t || exit 1; done

build:
	@mkdir -p build

clean:
	rm -rf build/*.o $(LIB) $(TEST_BIN)
