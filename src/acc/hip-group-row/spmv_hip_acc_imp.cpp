//
// Created by chaohu on 2021/04/??.
//
// spmv_csr_pcsr_kernel version
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#define WF_SIZE 64
#define BLOCK_SIZE 256
#define GRID_SIZE 256

template <int GROUP_SIZE>
__global__ void spmv_group_row_kernel(int trans, const int alpha, const int beta, const int *ia, const int *ja,
                                      const double *va, const double *x, double *y, int size) {

  int globalThreadId = threadIdx.x + blockDim.x * blockIdx.x;
  int groupThreadId = globalThreadId % GROUP_SIZE;
  int groupId = globalThreadId / GROUP_SIZE;
  int groupNum = gridDim.x * blockDim.x / GROUP_SIZE;
  int rowStart, rowEnd;
  double sum;

  int row = groupId;
  while (row < size) {
    rowStart = ia[row];
    rowEnd = ia[row + 1];
    sum = 0;

    for (int i = rowStart + groupThreadId; i < rowEnd; i += GROUP_SIZE) {
      sum += va[i] * x[ja[i]];
    }

    for (int i = GROUP_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, GROUP_SIZE);
    }

    if (groupThreadId == 0) {
      y[row] = alpha * sum + beta * y[row];
    }

    row += groupNum;
  }
}

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y) {

  int meanElementsPerRow = (rowptr[m] - rowptr[0]) / m;

  if (meanElementsPerRow <= 4) {
    spmv_group_row_kernel<2><<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 8) {
    spmv_group_row_kernel<4><<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 16) {
    spmv_group_row_kernel<8><<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 32) {
    spmv_group_row_kernel<16><<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, rowptr, colindex, value, x, y, m);
  } else if (meanElementsPerRow <= 64) {
    spmv_group_row_kernel<32><<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, rowptr, colindex, value, x, y, m);
  } else {
    spmv_group_row_kernel<64><<<GRID_SIZE, BLOCK_SIZE>>>(trans, alpha, beta, rowptr, colindex, value, x, y, m);
  }
}
