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
 * @param size rows in matrix A
 * @param alpha alpha value
 * @param beta beta value
 * @param ia row offset array of csr matrix A
 * @param ja col index of csr matrix A
 * @param va matrix A in csr format
 * @param x vector x
 * @param y vector y
 * @return
 */
template <int GROUP_SIZE, int WF_SIZE>
__global__ void spmv_group_row_kernel(int trans, const int alpha, const int beta, const int *ia, const int *ja,
                                      const double *va, const double *x, double *y, int size) {

  int globalThreadId = threadIdx.x + blockDim.x * blockIdx.x;
  int groupThreadId = globalThreadId % GROUP_SIZE;           // local thread id in current group
  const int group_id = globalThreadId / GROUP_SIZE;          // global group id
  const int group_num = gridDim.x * blockDim.x / GROUP_SIZE; // total groups on device
  int rowStart, rowEnd;
  double sum;

  int row = group_id;
  for (row = group_id; row < size; row += group_num) {
    rowStart = ia[row];
    rowEnd = ia[row + 1];
    sum = 0;

    for (int i = rowStart + groupThreadId; i < rowEnd; i += GROUP_SIZE) {
      sum += va[i] * x[ja[i]];
    }

    // reduce inside a group
    for (int i = GROUP_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, GROUP_SIZE);
    }

    if (groupThreadId == 0) {
      y[row] = alpha * sum + beta * y[row];
    }
  }
}

#define GROUP_KERNEL_WRAPPER(N)                                                                                        \
  (spmv_group_row_kernel<N, 64>)<<<256, 256>>>(trans, alpha, beta, rowptr, colindex, value, x, y, m)

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y) {

  const int mean_eles_per_row = rowptr[m] / m;

  if (mean_eles_per_row <= 4) {
    GROUP_KERNEL_WRAPPER(2);
  } else if (mean_eles_per_row <= 8) {
    GROUP_KERNEL_WRAPPER(4);
  } else if (mean_eles_per_row <= 16) {
    GROUP_KERNEL_WRAPPER(8);
  } else if (mean_eles_per_row <= 32) {
    GROUP_KERNEL_WRAPPER(16);
  } else if (mean_eles_per_row <= 64) {
    GROUP_KERNEL_WRAPPER(32);
  } else {
    GROUP_KERNEL_WRAPPER(64);
  }
}
