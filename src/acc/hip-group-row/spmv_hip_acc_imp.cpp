//
// Created by chaohu on 2021/04/25.
//

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

/**
 * We solve SpMV with group method.
 * In this method, wavefront can be divided into several groups (wavefront must be divided with no remainder).
 * (e.g. groups size can only be 1, 2,4,8,16,32,64 if \tparam WF_SIZE is 64).
 * Then, each group can process one row of matrix A,
 * which also means one wavefront with multiple groups can compute multiple rows.
 *
 * @tparam GROUP_SIZE threads in on group
 * @tparam WF_SIZE threads in one wavefront
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
template <int GROUP_SIZE, int WF_SIZE>
__global__ void spmv_group_row_kernel(int m, const int alpha, const int beta, const int *row_offset,
                                      const int *csr_col_ind, const double *csr_val, const double *x, double *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int group_thread_id = global_thread_id % GROUP_SIZE; // local thread id in current group
  const int group_id = global_thread_id / GROUP_SIZE;        // global group id
  const int group_num = gridDim.x * blockDim.x / GROUP_SIZE; // total groups on device

  int row = group_id;
  for (row = group_id; row < m; row += group_num) {
    const int row_start = row_offset[row];
    const int row_end = row_offset[row + 1];
    double sum = 0;

    for (int i = row_start + group_thread_id; i < row_end; i += GROUP_SIZE) {
      sum += csr_val[i] * x[csr_col_ind[i]];
    }

    // reduce inside a group
    for (int i = GROUP_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, GROUP_SIZE);
    }

    if (group_thread_id == 0) {
      y[row] = alpha * sum + beta * y[row];
    }
  }
}

#define GROUP_KERNEL_WRAPPER(N)                                                                                        \
  (spmv_group_row_kernel<N, 64>)<<<256, 256>>>(m, alpha, beta, rowptr, colindex, value, x, y)

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y) {
  const int avg_eles_per_row = rowptr[m] / m;

  if (avg_eles_per_row <= 4) {
    GROUP_KERNEL_WRAPPER(2);
  } else if (avg_eles_per_row <= 8) {
    GROUP_KERNEL_WRAPPER(4);
  } else if (avg_eles_per_row <= 16) {
    GROUP_KERNEL_WRAPPER(8);
  } else if (avg_eles_per_row <= 32) {
    GROUP_KERNEL_WRAPPER(16);
  } else if (avg_eles_per_row <= 64) {
    GROUP_KERNEL_WRAPPER(32);
  } else {
    GROUP_KERNEL_WRAPPER(64);
  }
}
