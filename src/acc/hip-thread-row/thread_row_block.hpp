//
// Created by genshen on 2021/7/26.
//

#ifndef SPMV_ACC_THREAD_ROW_BLOCK_HPP
#define SPMV_ACC_THREAD_ROW_BLOCK_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/global_mem_ops.hpp"
#include "../common/utils.h"
#include "thread_row_config.h"

// calculate in a block
template <int N, int MAX_ROW_NNZ, int WF_SIZE, int THREADS, typename I, typename T>
__global__ void kernel_thread_row_block_level(const T alpha, const T beta, const I m, const I *__restrict__ row_ptr,
                                              const I *__restrict__ csr_col_inx, const T *__restrict__ csr_val,
                                              const T *__restrict__ x, T *__restrict__ y) {
  int t_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int global_threads_num = blockDim.x * gridDim.x;

  const int g_block_id = t_id / THREADS;
  const int tid_in_block = t_id % THREADS;

  constexpr int shared_len = N * THREADS * MAX_ROW_NNZ;
  __shared__ T _shared_val[shared_len];

  // In each loop, each thread process N rows.
  const I wf_rounds = m / THREADS + (m % THREADS == 0 ? 0 : 1);

  constexpr int N_UNROLLING_SHIFT = 1;
  for (I i = N * g_block_id; i < wf_rounds; i += N * gridDim.x) {
    // each wavefront process `N * gridDim.x` rows.
    if (i * THREADS >= m) {
      return;
    }
    const I block_row_start_id = i * THREADS;
    const I block_row_end_id = min((i + 1) * THREADS, m);

    // we have: block_row_start_id < block_row_end_id and block_row_start_id < m.
    const I reduce_row_id = min(block_row_start_id + tid_in_block, m - 1);
    const I thread_row_start = row_ptr[reduce_row_id];
    const I thread_row_end = row_ptr[reduce_row_id + 1];

    const I block_start_index = row_ptr[block_row_start_id]; // __shfl(thread_row_start, 0);
    const I block_end_index = row_ptr[block_row_end_id];     // __shfl(thread_row_end, WF_SIZE - 1);

    __syncthreads();
    for (I j = block_start_index + tid_in_block; j < block_end_index; j += THREADS) {
      const T local_val = csr_val[j] * x[csr_col_inx[j]];
      _shared_val[j - block_start_index] = local_val;
    }
    __syncthreads();

    const T y_local = __builtin_nontemporal_load(y + reduce_row_id);

    // reduction
    // The last row may be reduced and stored more than once by threads in the last wavefront,
    // but it does not matter.
    const I reduce_start_index = thread_row_start - block_start_index;
    const I reduce_end_index = thread_row_end - block_start_index;
    T sum = static_cast<T>(0);
    for (I k = reduce_start_index; k < reduce_end_index; k++) {
      sum += _shared_val[k];
    }

    const T y_result = alpha * sum + beta * y_local;
    __builtin_nontemporal_store(y_result, y + reduce_row_id);
  }
}

#endif // SPMV_ACC_THREAD_ROW_BLOCK_HPP
