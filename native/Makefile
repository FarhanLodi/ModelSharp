# ModelSharp native kernels build.
#
#   make           -> build/libms_kernels.so   (the shared lib C# P/Invokes)
#   make tests     -> build/test_* standalone self-tests (parity + GFLOP/s)
#   make run-tests -> build and run every self-test
#   make clean
#
# -march=native targets this host's exact ISA (znver4 here => AVX-512 + VNNI).
# For a portable build, replace with: -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mfma

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -mfma -fopenmp -fPIC -std=c++17 -Wall -funroll-loops -ffast-math
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
