//
// Created by chaohu on 2021/04/25.
//

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "../common/utils.h"

/**
 * We solve SpMV with vector method.
 * In this method, wavefront can be divided into several groups (wavefront must be divided with no remainder).
 * (e.g. groups size can only be 1, 2,4,8,16,32,64 if \tparam WF_SIZE is 64).
 * Here, one group of threads are called a "vector".
 * Then, each vector can process one row of matrix A,
 * which also means one wavefront with multiple vectors can compute multiple rows.
 *
 * @tparam VECTOR_SIZE threads in one vector
 * @tparam WF_SIZE threads in one wavefront
 * @tparam T type of data in matrix A, vector x, vector y and alpha, beta.
 * @param m rows in matrix A
 * @param alpha alpha value
 * @param beta beta value
 * @param row_offset row offset array of csr matrix A
 * @param csr_col_ind col index of csr matrix A
 * @param csr_val matrix A in csr format
 * @param x vector x
 * @param y vector y
 * @return
 */
template <int VECTOR_SIZE, int WF_SIZE, typename T>
__global__ void spmv_vector_row_kernel(int m, const T alpha, const T beta, const int *row_offset, const int *csr_col_ind,
                                      const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int vector_thread_id = global_thread_id % VECTOR_SIZE; // local thread id in current vector
  const int vector_id = global_thread_id / VECTOR_SIZE;        // global vector id
  const int vector_num = gridDim.x * blockDim.x / VECTOR_SIZE; // total vectors on device

  for (int row = vector_id; row < m; row += vector_num) {
    const int row_start = row_offset[row];
    const int row_end = row_offset[row + 1];
    T sum = static_cast<T>(0);

    for (int i = row_start + vector_thread_id; i < row_end; i += VECTOR_SIZE) {
      asm_v_fma_f64(csr_val[i], device_ldg(x + csr_col_ind[i]), sum);
    }

    // reduce inside a vector
    for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, VECTOR_SIZE);
    }

    if (vector_thread_id == 0) {
      y[row] = device_fma(beta, y[row], alpha * sum);
    }
  }
}

#define VECTOR_KERNEL_WRAPPER(N)                                                                                        \
  (spmv_vector_row_kernel<N, 64, double>)<<<256, 256>>>(m, alpha, beta, rowptr, colindex, value, x, y)

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y) {
  const int avg_eles_per_row = rowptr[m] / m;

  if (avg_eles_per_row <= 4) {
    VECTOR_KERNEL_WRAPPER(2);
  } else if (avg_eles_per_row <= 8) {
    VECTOR_KERNEL_WRAPPER(4);
  } else if (avg_eles_per_row <= 16) {
    VECTOR_KERNEL_WRAPPER(8);
  } else if (avg_eles_per_row <= 32) {
    VECTOR_KERNEL_WRAPPER(16);
  } else if (avg_eles_per_row <= 64) {
    VECTOR_KERNEL_WRAPPER(32);
  } else {
    VECTOR_KERNEL_WRAPPER(64);
  }
}
