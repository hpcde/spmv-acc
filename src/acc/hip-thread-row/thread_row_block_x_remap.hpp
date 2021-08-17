//
// Created by reget on 2021/8/17.
//

#ifndef SPMV_ACC_THREAD_ROW_BLOCK_X_REMAP_HPP
#define SPMV_ACC_THREAD_ROW_BLOCK_X_REMAP_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/global_mem_ops.hpp"
#include "../common/utils.h"
#include "thread_row_config.h"

// calculate in a block
template <int N, int MAX_ROW_NNZ, int WF_SIZE, int THREADS, typename I, typename T>
__global__ void kernel_thread_row_block_v2(const T alpha, const T beta, const I m, const I *__restrict__ row_ptr,
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
#ifndef THREAD_ROW_GLOBAL_LOAD_X2
    for (I j = block_start_index + tid_in_block; j < block_end_index; j += THREADS) {
      _shared_val[j - block_start_index] = csr_col_inx[j];
    }
#endif
#ifdef THREAD_ROW_GLOBAL_LOAD_X2
    {
      const int n_lds_load = block_end_index - block_start_index;
      const int unrolling_loop_end = block_start_index + ((n_lds_load >> N_UNROLLING_SHIFT) << N_UNROLLING_SHIFT);
      for (I j = block_start_index + 2 * tid_in_block; j < unrolling_loop_end; j += 2 * THREADS) {
        int_x2 int_v_x2;
        global_load_int(static_cast<const void *>(csr_col_inx + j), int_v_x2);
        _shared_val[j - block_start_index] = int_v_x2.a;
        _shared_val[j - block_start_index + 1] = int_v_x2.b;
      }
      for (I j = unrolling_loop_end + tid_in_block; j < block_end_index; j += THREADS) {
        _shared_val[j - block_start_index] = csr_col_inx[j];
      }
    }
#endif
    __syncthreads();
    // step2: remapping LDS (column index) and read x vector via the remapped column index.
    // each thread read data belonging to its row.
    T _thread_local_x_vec[MAX_ROW_NNZ];
#pragma unroll
    for (I j = 0; j < MAX_ROW_NNZ; j++) {
      const I col_addr = thread_row_start + j;
      if (col_addr < thread_row_end) {
        const I thread_col_inx = _shared_val[col_addr - block_start_index];
        _thread_local_x_vec[j] = x[thread_col_inx];
      }
    }
    __syncthreads();
    // step3: load matrix value to LDS
#ifndef THREAD_ROW_GLOBAL_LOAD_X2
    for (I j = block_start_index + tid_in_block; j < block_end_index; j += THREADS) {
      _shared_val[j - block_start_index] = csr_val[j];
    }
#endif
#ifdef THREAD_ROW_GLOBAL_LOAD_X2
    {
      const int n_lds_load = block_end_index - block_start_index;
      const int unrolling_loop_end = block_start_index + ((n_lds_load >> N_UNROLLING_SHIFT) << N_UNROLLING_SHIFT);
      for (I j = block_start_index + 2 * tid_in_block; j < unrolling_loop_end; j += 2 * THREADS) {
        dbl_x2 dbl_v_x2;
        global_load(static_cast<const void *>(csr_val + j), dbl_v_x2);
        asm volatile("s_waitcnt vmcnt(0)");
        _shared_val[j - block_start_index] = dbl_v_x2.a;
        _shared_val[j - block_start_index + 1] = dbl_v_x2.b;
      }
      for (I j = unrolling_loop_end + tid_in_block; j < block_end_index; j += THREADS) {
        _shared_val[j - block_start_index] = csr_val[j];
      }
    }
#endif
    __syncthreads();
    // step4: remapping matrix value and perform multiplication of Ax.
    const T y_local = __builtin_nontemporal_load(y + reduce_row_id);

    // reduction
    // The last row may be reduced and stored more than once by threads in the last wavefront,
    // but it does not matter.
    const I reduce_start_index = thread_row_start - block_start_index;
    const I reduce_end_index = thread_row_end - block_start_index;
    T sum = static_cast<T>(0);
    #pragma unroll
    for (I j = 0; j < MAX_ROW_NNZ; j++) {
      const I col_addr = reduce_start_index + j;
      if (col_addr < reduce_end_index) {
        const T thread_value = _shared_val[col_addr];
        sum += thread_value * _thread_local_x_vec[j];
      }
    }
    // step5: store y
    const T y_result = alpha * sum + beta * y_local;
    __builtin_nontemporal_store(y_result, y + reduce_row_id);
  }
}

#endif // SPMV_ACC_THREAD_ROW_BLOCK_HPP
