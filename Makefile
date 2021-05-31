
MKL_PATH=/home/buddhika/Installs/mkl/mkl/
MKL_INCLUDE=$(MKL_PATH)/include
MKL_LIB=$(MKL_PATH)/lib/intel64

all: 
	g++ -O2 -std=c++11 -fopenmp -march=core-avx2 -DMKL_ILP64 -m64 -I$(MKL_INCLUDE) -o dgemm  -L$(MKL_LIB) -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl matmul.cc

dgemm: dgemm.cc
	g++ -O2 -std=c++11 -fopenmp -march=core-avx2 -DMKL_ILP64 -m64 -I$(MKL_INCLUDE) -o dgemm  -L$(MKL_LIB) -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl dgemm.cc

dgemm-O3: dgemm.cc
	g++ -O3 -std=c++11 -fopenmp -march=core-avx2 -DMKL_ILP64 -m64 -I$(MKL_INCLUDE) -o dgemm  -L$(MKL_LIB) -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl dgemm.cc


simd:
	g++ -O3 -ggdb -march=core-avx2 -o simd test-simd.cc

clean:
	rm -f matmul simd dgemm
