//
// Created by chaohu on 2021/04/25.
//
// spmv_csr_pcsr_kernel version
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#include "../common/utils.h"

#define WF_SIZE 64
#define BLOCK_SIZE 256
#define GRID_SIZE 256

template <int THREADS_PER_VECTOR>
__global__ void spmv_light_kernel(int trans, const int alpha, const int beta, int *hip_row_counter, const int *ia,
                                  const int *ja, const double *va, const double *x, double *y, int size) {

  int i;
  double sum;
  int row;
  int row_start, row_end;
  const int land_id = threadIdx.x % THREADS_PER_VECTOR;
  const int wf_land_id = threadIdx.x & (WF_SIZE - 1);
  const int wf_vec_id = wf_land_id / THREADS_PER_VECTOR;

  if (wf_land_id == 0) {
    row = atomicAdd(hip_row_counter, WF_SIZE / THREADS_PER_VECTOR);
  }
  row = __shfl(row, 0, WF_SIZE) + wf_vec_id;
  __syncthreads();

  while (row < size) {
    row_start = ia[row];
    row_end = ia[row + 1];
    sum = 0;

    for (i = row_start + land_id; i < row_end; i += THREADS_PER_VECTOR) {
      sum += va[i] * x[ja[i]];
    }

    // reduction
    for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
      __syncthreads();
    }

    if (land_id == 0) {
      y[row] = device_fma(beta, y[row], alpha * sum);
    }

    if (wf_land_id == 0) {
      row = atomicAdd(hip_row_counter, WF_SIZE / THREADS_PER_VECTOR);
    }
    row = __shfl(row, 0, WF_SIZE) + wf_vec_id;
    __syncthreads();
  }
}

#define LIGHT_KERNEL_CALLER(N)                                                                                         \
  ((spmv_light_kernel<N>) <<<GRID_SIZE, BLOCK_SIZE>>>                                                                  \
   (trans, alpha, beta, hip_row_counter, rowptr, colindex, value, x, y, m))

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y) {

  int *hip_row_counter;
  hipMalloc((void **)&hip_row_counter, sizeof(int));
  const int mean_eles_per_row = (rowptr[m] - rowptr[0]) / m;
  hipMemset(hip_row_counter, 0, sizeof(int));

  if (mean_eles_per_row <= 2) {
    LIGHT_KERNEL_CALLER(1);
  } else if (mean_eles_per_row <= 4) {
    LIGHT_KERNEL_CALLER(2);
  } else if (mean_eles_per_row <= 8) {
    LIGHT_KERNEL_CALLER(4);
  } else if (mean_eles_per_row <= 16) {
    LIGHT_KERNEL_CALLER(8);
  } else if (mean_eles_per_row <= 32) {
    LIGHT_KERNEL_CALLER(16);
  } else if (mean_eles_per_row <= 64) {
    LIGHT_KERNEL_CALLER(32);
  } else {
    LIGHT_KERNEL_CALLER(64);
  }

  hipFree(hip_row_counter);
}
