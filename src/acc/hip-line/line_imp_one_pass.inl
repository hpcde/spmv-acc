//
// Created by reget on 2021/09/29.
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/utils.h"
#include "building_config.h"

template <int ROW_SIZE, int MAX_ROW_NNZ, typename I, typename T>
__global__ void spmv_line_one_pass_kernel(int m, const T alpha, const T beta, const int *row_offset,
                                          const int *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int block_id = blockIdx.x;                                 // global block id
  const int block_thread_num = blockDim.x;                         // threads num in a block
  const int block_thread_id = global_thread_id % block_thread_num; // local thread id in current block
  constexpr int shared_len = ROW_SIZE * MAX_ROW_NNZ;
  __shared__ T shared_val[shared_len];
  const I block_row_begin = block_id * ROW_SIZE;
  const I block_row_end = min(block_row_begin + ROW_SIZE, m);
  // load val to lds parallel
  const I block_row_idx_begin = row_offset[block_row_begin];
  const I block_row_idx_end = row_offset[block_row_end];
  for (I i = block_row_idx_begin + block_thread_id; i < block_row_idx_end; i += block_thread_num) {
    shared_val[i - block_row_idx_begin] = csr_val[i] * x[csr_col_ind[i]];
  }
  __syncthreads();
  // `ROW_SIZE` must smaller than `block_thread_num`
  const I reduce_row_id = block_row_begin + block_thread_id;
  if (reduce_row_id >= block_row_end) {
    return;
  }
  const I reduce_row_idx_begin = row_offset[reduce_row_id];
  const I reduce_row_idx_end = row_offset[reduce_row_id + 1];
  T sum = static_cast<T>(0);
  for (I i = reduce_row_idx_begin; i < reduce_row_idx_end; i++) {
    sum += shared_val[i - block_row_idx_begin];
  }
  y[reduce_row_id] = alpha * sum + y[reduce_row_id];
}

#define LINE_ONE_PASS_KERNEL_WRAPPER(N, MAX_ROW_NNZ, BLOCKS, THREADS)                                                  \
  (spmv_line_one_pass_kernel<N, MAX_ROW_NNZ, int, double>)<<<BLOCKS, THREADS>>>(m, alpha, beta, rowptr, colindex,      \
                                                                                value, x, y)
