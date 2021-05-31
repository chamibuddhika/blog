/*
 * dgemm -- A stage-wise optimized matrix multiplication kernel.
 *
 * Copyright (c) 2020 Buddhika Chamith
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

/*
 * Functions implemented in this file are
 *
 * dgemm_* : Manually optimized kernel for matrix multiplication
 *    C = alpha * A * B + beta * C (with alpha=1 and beta=1)
 * dgemm_blas : Same operation implemented via Intel Math Kernel Library.
 *
 * All arrays are assumed to be square with all dimensions being a power of two
 * and a multiple of cache line size (64) in order to keep implementations
 * simple.
 */

#include <ctype.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <random>
#include <string>

#include "clock.h"
#include "mkl.h"
#include "mkl_cblas.h"

int N = 2048;  // Matrix dimensions.
int B = 256;   // Cache block size.

// Input and output arrays.
struct Arrays {
  double* a;
  double* b;
  double* c_b;  // Output array of BLAS dgemm.
  double* c_m;  // Output array of manual dgemm.
};

/*
 * util functions.
 */

Arrays* init_arrays() {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(5.0, 2.0);

  int sz = N * N;
  double* a = reinterpret_cast<double*>(aligned_alloc(64, sizeof(double) * sz));
  double* b = reinterpret_cast<double*>(aligned_alloc(64, sizeof(double) * sz));
  double* c_b =
      reinterpret_cast<double*>(aligned_alloc(64, sizeof(double) * sz));
  double* c_m =
      reinterpret_cast<double*>(aligned_alloc(64, sizeof(double) * sz));

  // Init a.
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[N * i + j] = distribution(generator);
    }
  }

  // Init b.
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      b[N * i + j] = distribution(generator);
    }
  }

  // Init outputs.
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double val = distribution(generator);
      c_b[N * i + j] = val;
      c_m[N * i + j] = val;
    }
  }

  Arrays* arr = new Arrays;
  arr->a = a;
  arr->b = b;
  arr->c_b = c_b;
  arr->c_m = c_m;

  return arr;
}

// Verify the result of manual dgemm with BLAS dgemm.
void verify(Arrays* arrs) {
  double* c_m = arrs->c_m;
  double* c_b = arrs->c_b;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double d = arrs->c_b[N * i + j];
      double m = arrs->c_m[N * i + j];
      // printf("c_b[%d][%d], c_m[%d][%d] : %lf, %lf\n", i, j, i, j, d, m);
      if (!(fabs(d - m) <=
            ((fabs(d) > fabs(m) ? fabs(m) : fabs(d)) * 0.000001))) {
        std::cout << "ERROR: Invalid result\n";
        exit(-1);
      }
    }
  }
}

/*
 * dgemm functions.
 */

void dgemm_blas(Arrays* arrs) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, arrs->a, N,
              arrs->b, N, 1, arrs->c_b, N);
}

void dgemm_naive(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {

        // c_m[i][j] = a[i][k] * b[k][j]
        c_m[N * i + j] += a[N * i + k] * b[N * k + j];
      }
    }
  }
}

void dgemm_tile(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      for (int k = 0; k < N; k += B) {

        // Block multiplication.
        for (int ib = 0; ib < B; ib++) {
          for (int jb = 0; jb < B; jb++) {
            for (int kb = 0; kb < B; kb++) {

              c_m[N * (i + ib) + (j + jb)] +=
                  a[N * (i + ib) + (k + kb)] * b[N * (k + kb) + (j + jb)];
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      for (int k = 0; k < N; k += B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          for (int kb = 0; kb < B; kb++) {
            for (int jb = 0; jb < B; jb++) {
              c_m[N * (i + ib) + (j + jb)] +=
                  a[N * (i + ib) + (k + kb)] * b[N * (k + kb) + (j + jb)];
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B, c_m += N * B, a += N * B) {
    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c_m block row start.
          double* a_blk = a + N * ib;      // a block row start.

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            double* b_blk = b + N * kb;  // b block row start.
            for (int jb = j; jb < JEND; jb++) {
              c_m_blk[jb] += a_blk[k + kb] * b_blk[jb];
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist_prefetch(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B, c_m += N * B, a += N * B) {
    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c block start.
          double* a_blk = a + N * ib;      // a block start.

          double* a_blk_nxt = a + N * (ib + 1) + k;  // a prefetch cache line.
          // Prefetch next row of a block non temporally.
          _mm_prefetch(&a_blk_nxt, _MM_HINT_NTA);

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            double* b_blk = b + N * kb;                // b block start.
            double* b_blk_nxt = b + N * (kb + 1) + j;  // b prefetch cache line.

            // Prefetch next row of b block to all caches (L1 and up) in the
            // cache hierarchy.
            _mm_prefetch(&b_blk_nxt, _MM_HINT_T0);
            for (int jb = j; jb < JEND; jb++) {
              c_m_blk[jb] += a_blk[k + kb] * b_blk[jb];
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist_prefetch_simd(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B, c_m += N * B, a += N * B) {
    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c block row start.
          double* a_blk = a + N * ib;      // a block row start.

          double* a_blk_nxt = a + N * (ib + 1) + k;  // a prefetch cache line.
          // Prefetch next row of a block non temporally.
          _mm_prefetch(&a_blk_nxt, _MM_HINT_NTA);

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            __m256d a_k = _mm256_broadcast_sd(&a_blk[k + kb]);
            double* b_blk = b + N * kb;                // b block row start.
            double* b_blk_nxt = b + N * (kb + 1) + j;  // b prefetch cache line.

            // Prefetch next row of b block to L2 and up in the
            // cache hierarchy.
            _mm_prefetch(&b_blk_nxt, _MM_HINT_T1);

            for (int jb = j; jb < JEND; jb += 4) {
              __m256d b_j = _mm256_load_pd(&b_blk[jb]);
              __m256d c_m_j = _mm256_load_pd(&c_m_blk[jb]);
              c_m_j = _mm256_fmadd_pd(a_k, b_j, c_m_j);
              _mm256_store_pd(&c_m_blk[jb], c_m_j);
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist_simd(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B, c_m += N * B, a += N * B) {
    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c block row start.
          double* a_blk = a + N * ib;      // a block row start.

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            __m256d a_k = _mm256_broadcast_sd(&a_blk[k + kb]);
            double* b_blk = b + N * kb;  // b block row start.

            for (int jb = j; jb < JEND; jb += 4) {
              __m256d b_j = _mm256_load_pd(&b_blk[jb]);
              __m256d c_m_j = _mm256_load_pd(&c_m_blk[jb]);
              c_m_j = _mm256_fmadd_pd(a_k, b_j, c_m_j);
              _mm256_store_pd(&c_m_blk[jb], c_m_j);
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist_prefetch_simd_unroll(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B, c_m += N * B, a += N * B) {
    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c block row start.
          double* a_blk = a + N * ib;      // a block row start.

          double* a_blk_nxt = a + N * (ib + 1) + k;  // a prefetch cache line.
          // Prefetch next row of a block non temporally.
          _mm_prefetch(&a_blk_nxt, _MM_HINT_NTA);

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            __m256d a_k = _mm256_broadcast_sd(&a_blk[k + kb]);
            double* b_blk = b + N * kb;                // b block row start.
            double* b_blk_nxt = b + N * (kb + 1) + j;  // b prefetch cache line.

            // Prefetch next row of b block to all caches L2 and up in the
            // cache hierarchy.
            _mm_prefetch(&b_blk_nxt, _MM_HINT_T1);

            for (int jb = j; jb < JEND; jb += 16) {
              register int base = jb;
              __m256d b_j = _mm256_load_pd(&b_blk[base]);
              __m256d c_m_j = _mm256_load_pd(&c_m_blk[base]);

              register int base4 = base + 4;
              __m256d b_j_1 = _mm256_load_pd(&b_blk[base4]);
              __m256d c_m_j_1 = _mm256_load_pd(&c_m_blk[base4]);

              c_m_j = _mm256_fmadd_pd(a_k, b_j, c_m_j);
              _mm256_store_pd(&c_m_blk[base], c_m_j);

              c_m_j_1 = _mm256_fmadd_pd(a_k, b_j_1, c_m_j_1);
              _mm256_store_pd(&c_m_blk[base4], c_m_j_1);

              register int base8 = base + 8;
              __m256d b_j_2 = _mm256_load_pd(&b_blk[base8]);
              __m256d c_m_j_2 = _mm256_load_pd(&c_m_blk[base8]);

              register int base12 = base + 12;
              __m256d b_j_3 = _mm256_load_pd(&b_blk[base12]);
              __m256d c_m_j_3 = _mm256_load_pd(&c_m_blk[base12]);

              c_m_j_2 = _mm256_fmadd_pd(a_k, b_j_2, c_m_j_2);
              _mm256_store_pd(&c_m_blk[base8], c_m_j_2);

              c_m_j_3 = _mm256_fmadd_pd(a_k, b_j_3, c_m_j_3);
              _mm256_store_pd(&c_m_blk[base12], c_m_j_3);
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist_simd_unroll(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

  for (int i = 0; i < N; i += B, c_m += N * B, a += N * B) {
    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c block row start.
          double* a_blk = a + N * ib;      // a block row start.

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            __m256d a_k = _mm256_broadcast_sd(&a_blk[k + kb]);
            double* b_blk = b + N * kb;  // b block row start.

            for (int jb = j; jb < JEND; jb += 16) {
              register int base = jb;
              __m256d b_j = _mm256_load_pd(&b_blk[base]);
              __m256d c_m_j = _mm256_load_pd(&c_m_blk[base]);

              register int base4 = base + 4;
              __m256d b_j_1 = _mm256_load_pd(&b_blk[base4]);
              __m256d c_m_j_1 = _mm256_load_pd(&c_m_blk[base4]);

              c_m_j = _mm256_fmadd_pd(a_k, b_j, c_m_j);
              _mm256_store_pd(&c_m_blk[base], c_m_j);

              c_m_j_1 = _mm256_fmadd_pd(a_k, b_j_1, c_m_j_1);
              _mm256_store_pd(&c_m_blk[base4], c_m_j_1);

              register int base8 = base + 8;
              __m256d b_j_2 = _mm256_load_pd(&b_blk[base8]);
              __m256d c_m_j_2 = _mm256_load_pd(&c_m_blk[base8]);

              register int base12 = base + 12;
              __m256d b_j_3 = _mm256_load_pd(&b_blk[base12]);
              __m256d c_m_j_3 = _mm256_load_pd(&c_m_blk[base12]);

              c_m_j_2 = _mm256_fmadd_pd(a_k, b_j_2, c_m_j_2);
              _mm256_store_pd(&c_m_blk[base8], c_m_j_2);

              c_m_j_3 = _mm256_fmadd_pd(a_k, b_j_3, c_m_j_3);
              _mm256_store_pd(&c_m_blk[base12], c_m_j_3);
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist_prefetch_simd_unroll_par(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

#pragma omp parallel for
  for (int i = 0; i < N; i += B) {
    double* c_m = arrs->c_m + N * i;
    double* a = arrs->a + N * i;
    double* b = nullptr;

    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c block row start.
          double* a_blk = a + N * ib;      // a block row start.

          double* a_blk_nxt = a + N * (ib + 1) + k;  // a prefetch cache line.
          // Prefetch next row of a block non temporally.
          _mm_prefetch(&a_blk_nxt, _MM_HINT_NTA);

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            __m256d a_k = _mm256_broadcast_sd(&a_blk[k + kb]);
            double* b_blk = b + N * kb;                // b block row start.
            double* b_blk_nxt = b + N * (kb + 1) + j;  // b prefetch cache line.

            // Prefetch next row of b block to all caches L2 and up in the
            // cache hierarchy.
            _mm_prefetch(&b_blk_nxt, _MM_HINT_T1);

            for (int jb = j; jb < JEND; jb += 16) {
              register int base = jb;
              __m256d b_j = _mm256_load_pd(&b_blk[base]);
              __m256d c_m_j = _mm256_load_pd(&c_m_blk[base]);

              register int base4 = base + 4;
              __m256d b_j_1 = _mm256_load_pd(&b_blk[base4]);
              __m256d c_m_j_1 = _mm256_load_pd(&c_m_blk[base4]);

              c_m_j = _mm256_fmadd_pd(a_k, b_j, c_m_j);
              _mm256_store_pd(&c_m_blk[base], c_m_j);

              c_m_j_1 = _mm256_fmadd_pd(a_k, b_j_1, c_m_j_1);
              _mm256_store_pd(&c_m_blk[base4], c_m_j_1);

              register int base8 = base + 8;
              __m256d b_j_2 = _mm256_load_pd(&b_blk[base8]);
              __m256d c_m_j_2 = _mm256_load_pd(&c_m_blk[base8]);

              register int base12 = base + 12;
              __m256d b_j_3 = _mm256_load_pd(&b_blk[base12]);
              __m256d c_m_j_3 = _mm256_load_pd(&c_m_blk[base12]);

              c_m_j_2 = _mm256_fmadd_pd(a_k, b_j_2, c_m_j_2);
              _mm256_store_pd(&c_m_blk[base8], c_m_j_2);

              c_m_j_3 = _mm256_fmadd_pd(a_k, b_j_3, c_m_j_3);
              _mm256_store_pd(&c_m_blk[base12], c_m_j_3);
            }
          }
        }
      }
    }
  }
}

void dgemm_tile_ichange_hoist_simd_unroll_par(Arrays* arrs) {
  double* a = arrs->a;
  double* b = arrs->b;
  double* c_m = arrs->c_m;

#pragma omp parallel for
  for (int i = 0; i < N; i += B) {
    double* c_m = arrs->c_m + N * i;
    double* a = arrs->a + N * i;
    double* b = nullptr;

    for (int j = 0; j < N; j += B) {
      b = arrs->b;
      for (int k = 0; k < N; k += B, b += N * B) {

        // Do the block multiplication.
        for (int ib = 0; ib < B; ib++) {
          double* c_m_blk = c_m + N * ib;  // c block row start.
          double* a_blk = a + N * ib;      // a block row start.

          int JEND = j + B;
          for (int kb = 0; kb < B; kb++) {
            __m256d a_k = _mm256_broadcast_sd(&a_blk[k + kb]);
            double* b_blk = b + N * kb;  // b block row start.

            for (int jb = j; jb < JEND; jb += 16) {
              register int base = jb;
              __m256d b_j = _mm256_load_pd(&b_blk[base]);
              __m256d c_m_j = _mm256_load_pd(&c_m_blk[base]);

              register int base4 = base + 4;
              __m256d b_j_1 = _mm256_load_pd(&b_blk[base4]);
              __m256d c_m_j_1 = _mm256_load_pd(&c_m_blk[base4]);

              c_m_j = _mm256_fmadd_pd(a_k, b_j, c_m_j);
              _mm256_store_pd(&c_m_blk[base], c_m_j);

              c_m_j_1 = _mm256_fmadd_pd(a_k, b_j_1, c_m_j_1);
              _mm256_store_pd(&c_m_blk[base4], c_m_j_1);

              register int base8 = base + 8;
              __m256d b_j_2 = _mm256_load_pd(&b_blk[base8]);
              __m256d c_m_j_2 = _mm256_load_pd(&c_m_blk[base8]);

              register int base12 = base + 12;
              __m256d b_j_3 = _mm256_load_pd(&b_blk[base12]);
              __m256d c_m_j_3 = _mm256_load_pd(&c_m_blk[base12]);

              c_m_j_2 = _mm256_fmadd_pd(a_k, b_j_2, c_m_j_2);
              _mm256_store_pd(&c_m_blk[base8], c_m_j_2);

              c_m_j_3 = _mm256_fmadd_pd(a_k, b_j_3, c_m_j_3);
              _mm256_store_pd(&c_m_blk[base12], c_m_j_3);
            }
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  std::string impl = "naive";  // blas | naive | tile | tile_ichange |
                               // tile_ichange_hoist |
                               // tile_ichange_hoist_prefetch |
                               // tile_ichange_hoist_simd |
                               // tile_ichange_hoist_prefetch_simd |
                               // tile_ichange_hoist_prefetch_simd_unroll |
                               // tile_ichange_hoist_prefetch_simd_unroll_par
  bool is_compare = false;

  if (argc == 2) {
    impl = std::string(argv[1]);
  } else if (argc > 2) {
    int i = 1;
    while (i < argc) {
      char* arg = argv[i];
      if (strcmp(arg, "-c") == 0) {
        is_compare = true;
      } else if (strcmp(arg, "-b") == 0) {
        B = atoi(argv[i + 1]);
        i += 2;
        continue;
      } else if (strcmp(arg, "-n") == 0) {
        int sz = atoi(argv[i + 1]);
        N = sz;
        i += 2;
        continue;
      } else {
        impl = std::string(arg);
      }
      i += 1;
    }
  }

  bool is_blas = false;
  void (*impl_ptr)(Arrays*) = nullptr;
  if (impl == "blas") {
    is_blas = true;
  } else if (impl == "naive") {
    impl_ptr = dgemm_naive;
  } else if (impl == "tile") {
    impl_ptr = dgemm_tile;
  } else if (impl == "tile_ichange") {
    impl_ptr = dgemm_tile_ichange;
  } else if (impl == "tile_ichange_hoist") {
    impl_ptr = dgemm_tile_ichange_hoist;
  } else if (impl == "tile_ichange_hoist_prefetch") {
    impl_ptr = dgemm_tile_ichange_hoist_prefetch;
  } else if (impl == "tile_ichange_hoist_simd") {
    impl_ptr = dgemm_tile_ichange_hoist_simd;
  } else if (impl == "tile_ichange_hoist_prefetch_simd") {
    impl_ptr = dgemm_tile_ichange_hoist_prefetch_simd;
  } else if (impl == "tile_ichange_hoist_simd_unroll") {
    impl_ptr = dgemm_tile_ichange_hoist_simd_unroll;
  } else if (impl == "tile_ichange_hoist_prefetch_simd_unroll") {
    impl_ptr = dgemm_tile_ichange_hoist_prefetch_simd_unroll;
  } else if (impl == "tile_ichange_hoist_prefetch_simd_unroll_par") {
    impl_ptr = dgemm_tile_ichange_hoist_prefetch_simd_unroll_par;
  } else if (impl == "tile_ichange_hoist_simd_unroll_par") {
    impl_ptr = dgemm_tile_ichange_hoist_simd_unroll_par;
  }

  std::cout << "------ Parameters -----\n";
  std::cout << " Implementation   : " << impl << "\n";

  if (impl.rfind("tile", 0) == 0) {
    std::cout << " Block(Tile) size : " << B << "\n";
  }
  std::cout << " Matrix dimension : " << N << " x " << N << "\n";
  std::cout << "\n";

  std::cout << "------ Hardware Configuration -----\n";
  std::cout << " L1D Size : " << sysconf(_SC_LEVEL1_DCACHE_SIZE) << "\n";
  std::cout << " L2 Size  : " << sysconf(_SC_LEVEL2_CACHE_SIZE) << "\n";
  std::cout << " L3 Size  : " << sysconf(_SC_LEVEL3_CACHE_SIZE) << "\n";
  std::cout << " L1D Line Size : " << sysconf(_SC_LEVEL1_DCACHE_LINESIZE)
            << "\n\n";

  std::cout << "---- Initializing arrays...\n";
  Arrays* arrs = init_arrays();
  std::cout << " Array initialization done.\n\n";

  double dgemm_time = 0.0;

  Ticks t;
  if (is_compare || is_blas) {
    std::cout << "---- Running BLAS dgemm...\n";

    mkl_set_num_threads(1);
    Tic(&t);
    dgemm_blas(arrs);
    Toc(&t);

    dgemm_time = t.Elapsed(Unit::SECS);

    std::cout << " BLAS dgemm done.\n";
    std::cout << "    Elapsed (s) : " << dgemm_time << "\n\n";
  }

  double impl_time = 0.0;
  if (impl_ptr) {
    std::cout << "---- Running '" << impl << "' implementation...\n";
    Tic(&t);
    impl_ptr(arrs);
    Toc(&t);

    impl_time = t.Elapsed(Unit::SECS);
    std::cout << " '" << impl << "' implmentation done.\n";
    std::cout << "    Elapsed (s) : " << impl_time << "\n\n";
  }

  if (is_compare) {
    std::cout << "---- Verifying results...\n";
    verify(arrs);
    std::cout << "  Verification successful.\n\n";

    std::cout << "|>> Performance headroom (impl/blas) : "
              << impl_time / dgemm_time << "\n\n";
  }

  return 0;
}
