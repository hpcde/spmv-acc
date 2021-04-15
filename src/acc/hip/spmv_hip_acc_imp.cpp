//
// Created by genshen on 2021/4/15.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE


__global__ void device_sparse_spmv_acc(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                                   const int *colindex, const double *value, const double *x, double *y) {
  int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (thread_id == 0) {
    for (int i = 0; i < m; i++) {
      double y0 = 0;
      for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
        y0 += value[j] * x[colindex[j]];
        y[i] = alpha * y0 + beta * y[i];
    }
  }
}

void sparse_spmv(int htrans, const int halpha, const int hbeta, int hm, int hn, const int *hrowptr,
                 const int *hcolindex, const double *hvalue, const double *hx, double *hy) {
  device_sparse_spmv_acc<<<1, 256>>>(htrans, halpha, hbeta, hm, hn, hrowptr, hcolindex, hvalue, hx, hy);
}
