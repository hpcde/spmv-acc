//
// Created by genshen on 2021/7/25.
//

#ifndef SPMV_ACC_VECTOR_ROW_ADAPTIVE_HPP
#define SPMV_ACC_VECTOR_ROW_ADAPTIVE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/utils.h"

template <int VECTOR_SIZE, int WF_SIZE, typename I, typename T>
__device__ __forceinline__ void adaptive_vec_kernel_core(const int b_wf_id, const int b_thread_id, const int b_threads,
                                                         const I block_start_row, const I block_end_row, const T alpha,
                                                         const T beta, const I *row_offset, const I *csr_col_ind,
                                                         const T *csr_val, const T *x, T *y) {
  const int b_vec_id = b_thread_id / VECTOR_SIZE;
  const int b_vec_num = b_threads / VECTOR_SIZE;
  const int vector_thread_id = b_thread_id % VECTOR_SIZE;

  for (int row = block_start_row + b_vec_id; row < block_end_row; row += b_vec_num) {
    const int row_start = row_offset[row];
    const int row_end = row_offset[row + 1];
    T sum = static_cast<T>(0);

    for (int i = row_start + vector_thread_id; i < row_end; i += VECTOR_SIZE) {
      asm_v_fma_f64(csr_val[i], device_ldg(x + csr_col_ind[i]), sum);
    }

    // preload y value.
    T y_local = static_cast<T>(0);
    if (vector_thread_id == 0) {
      y_local = y[row];
    }

    // reduce inside a vector
    for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, VECTOR_SIZE);
    }

    if (vector_thread_id == 0) {
      y[row] = device_fma(beta, y_local, alpha * sum);
    }
  }
}

#define ADAPTIVE_VECTOR_KERNEL_CORE_WRAPPER(N)                                                                         \
  adaptive_vec_kernel_core<N, WF_SIZE, int, T>(b_wf_id, b_thread_id, b_threads, block_row_start, block_row_end, alpha, \
                                               beta, row_offset, csr_col_ind, csr_val, x, y);

template <int DATA_BLOCKS, int WF_SIZE, typename T>
__global__ void adaptive_vector_row_kernel(int m, const T alpha, const T beta, const int *row_offset,
                                           const int *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int wf_thread_id = global_thread_id % WF_SIZE; // local thread id in current wavefront
  const int wf_id = global_thread_id / WF_SIZE;        // global wavefront id
  const int wf_num = gridDim.x * blockDim.x / WF_SIZE; // total wavefront on device

  // make sure wf_num is DATA_BLOCKS*k.
  const int b_wf_num = wf_num / DATA_BLOCKS; // wavefront number for the data block
  const int data_block_i = wf_id / b_wf_num; // index of current data block (ith data block)
  const int b_wf_id = wf_id % b_wf_num;      // relative wavefront id in the data block
  const int b_thread_id = global_thread_id - data_block_i * b_wf_num * WF_SIZE; // relative thread id in the data block.
  const int b_threads = b_wf_num * WF_SIZE;                                     // threads for current data block.

  const int rows_cur_block = m / DATA_BLOCKS + (data_block_i < m % DATA_BLOCKS ? 1 : 0);
  const int block_row_start =
      data_block_i * (m / DATA_BLOCKS) + (data_block_i < m % DATA_BLOCKS ? data_block_i : m % DATA_BLOCKS);
  const int block_row_end = block_row_start + rows_cur_block;
  const int data_block_row_start = row_offset[block_row_start];
  const int data_block_row_end = row_offset[block_row_end];

  // apply adaptive vector size.
  const int avg_nnz = (data_block_row_end - data_block_row_start) / (block_row_end - block_row_start);
  if (avg_nnz <= 4) {
    ADAPTIVE_VECTOR_KERNEL_CORE_WRAPPER(2);
  } else if (avg_nnz <= 8) {
    ADAPTIVE_VECTOR_KERNEL_CORE_WRAPPER(4);
  } else if (avg_nnz <= 16) {
    ADAPTIVE_VECTOR_KERNEL_CORE_WRAPPER(8);
  } else if (avg_nnz <= 32) {
    ADAPTIVE_VECTOR_KERNEL_CORE_WRAPPER(16);
  } else if (avg_nnz <= 64) {
    ADAPTIVE_VECTOR_KERNEL_CORE_WRAPPER(32);
  } else {
    ADAPTIVE_VECTOR_KERNEL_CORE_WRAPPER(64);
  }
}

#define ADAPTIVE_VECTOR_KERNEL_WRAPPER(N)                                                                              \
  (adaptive_vector_row_kernel<N, 64, double>)<<<512, 256>>>(m, alpha, beta, rowptr, colindex, value, x, y);

#endif // SPMV_ACC_VECTOR_ROW_ADAPTIVE_HPP
