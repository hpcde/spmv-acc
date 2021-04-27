//
// Created by genshen on 2021/4/15.
//
//spmv_csr_scalar_kernel version 一个线程负责A一行的计算
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

#define WF_SIZE 64
#define BLOCK_SIZE 256
__global__ void device_sparse_spmv_acc(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                                   const int *colindex, const double *value, const double *x, double *y) {
  // thread id in block
  int block_thread_id = threadIdx.x;
  // thread id in wavefront
  int wf_thread_id = threadIdx.x & (WF_SIZE - 1);
  // thread id in global
  int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  // wavefront id in global
  int global_wf_id = global_thread_id / WF_SIZE;
  // number of wavefront
  int num_wf = gridDim.x * BLOCK_SIZE / WF_SIZE; 
  // share memory for store thread result
  for (int row = global_wf_id; row < m; row += num_wf) {
    int row_begin = rowptr[row];
    int row_end = rowptr[row + 1];
    double local_sum, total_sum;
    local_sum = 0;
    total_sum = 0;
    // calculate sum for all element in thread.
    for (int j = row_begin + wf_thread_id; j < row_end; j += WF_SIZE) {
      local_sum += value[j] * __ldg(x + colindex[j]);
    }
    // reduce thread sum to row sum
    total_sum += local_sum;
    __shfl_down(local_sum, 32);
    total_sum += local_sum;
    __shfl_down(local_sum, 16);
    total_sum += local_sum;
    __shfl_down(local_sum, 8); 
    total_sum += local_sum;
    __shfl_down(local_sum, 4);
    total_sum += local_sum;
    __shfl_down(local_sum, 2);
    total_sum += local_sum;
    __shfl_down(local_sum, 1);
    total_sum += local_sum;
    if (wf_thread_id == 0) {
       y[row] = alpha * total_sum + beta * y[row];
    }   
  }  
}

void sparse_spmv(int htrans, const int halpha, const int hbeta, int hm, int hn, const int *hrowptr,
                 const int *hcolindex, const double *hvalue, const double *hx, double *hy) {
  device_sparse_spmv_acc<<<64, 256>>>(htrans, halpha, hbeta, hm, hn, hrowptr, hcolindex, hvalue, hx, hy);
}
