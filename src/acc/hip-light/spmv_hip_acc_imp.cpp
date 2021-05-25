//
// Created by chaohu on 2021/04/??.
//
// spmv_csr_pcsr_kernel version
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#define WF_SIZE 64
#define BLOCK_SIZE 256
#define GRID_SIZE 256

template <int THREADS_PER_VECTOR>
__global__ void spmv_light_kernel(int trans, const int alpha, const int beta, int *hipRowCounter, const int *ia,
                                  const int *ja, const double *va, const double *x, double *y, int size) {

  int i;
  double sum;
  int row;
  int rowStart, rowEnd;
  int laneId = threadIdx.x % THREADS_PER_VECTOR;
  int vectorId = threadIdx.x / THREADS_PER_VECTOR;
  int warpLaneId = threadIdx.x & (WF_SIZE - 1);
  int warpVectorId = warpLaneId / THREADS_PER_VECTOR;

  if (warpLaneId == 0) {
    row = atomicAdd(hipRowCounter, WF_SIZE / THREADS_PER_VECTOR);
  }
  row = __shfl(row, 0, WF_SIZE) + warpVectorId;
  __syncthreads();

  while (row < size) {

    rowStart = ia[row];
    rowEnd = ia[row + 1];
    sum = 0;

    for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
      sum += va[i] * x[ja[i]];
    }

    for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
      __syncthreads();
    }

    if (laneId == 0) {
      y[row] = device_fma(beta, y[row], alpha * sum);
    }

    if (warpLaneId == 0) {
      row = atomicAdd(hipRowCounter, WF_SIZE / THREADS_PER_VECTOR);
    }
    row = __shfl(row, 0, WF_SIZE) + warpVectorId;
    __syncthreads();
  }
}

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y) {

  int *hipRowCounter;
  int rowCounter = 0;
  hipMalloc((void **)&hipRowCounter, sizeof(int));
  int meanElementsPerRow = (rowptr[m] - rowptr[0]) / m;
  hipMemset(hipRowCounter, 0, sizeof(int));

  if (meanElementsPerRow <= 2) {
    spmv_light_kernel<1>
        <<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, hipRowCounter, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 4) {
    spmv_light_kernel<2>
        <<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, hipRowCounter, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 8) {
    spmv_light_kernel<4>
        <<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, hipRowCounter, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 16) {
    spmv_light_kernel<8>
        <<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, hipRowCounter, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 32) {
    spmv_light_kernel<16>
        <<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, hipRowCounter, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 64) {
    spmv_light_kernel<32>
        <<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, hipRowCounter, rowptr, colindex, value, x, y, m);
  } else {
    spmv_light_kernel<64>
        <<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, hipRowCounter, rowptr, colindex, value, x, y, m);
  }

  hipFree(hipRowCounter);
}
