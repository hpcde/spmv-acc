//
// Created by genshen on 2021/07/06.
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/utils.h"

/**
 * We solve SpMV with flat method.
 *
 * @tparam ROW_SIZE the max nnz in one row.
 * @tparam WF_SIZE threads in one wavefront
 * @tparam BLOCKS total blocks on one GPU (blocks in one grid).
 * @tparam I type of data in matrix index
 * @tparam T type of data in matrix A, vector x, vector y and alpha, beta.
 * @param m rows in matrix A
 * @param alpha alpha value
 * @param beta beta value
 * @param row_offset row offset array of csr matrix A
 * @param csr_col_ind col index of csr matrix A
 * @param csr_val matrix A in csr format
 * @param x vector x
 * @param y vector y
 * @return
 */
template <int WF_SIZE, int R, int BLOCKS, int THREADS, typename I, typename T>
__global__ void spmv_flat_kernel(int m, const T alpha, const T beta, const I *__restrict__ row_offset,
                                 const I *__restrict__ break_points, const I *__restrict__ csr_col_ind,
                                 const T *__restrict__ csr_val, const T *__restrict__ x, T *__restrict__ y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  constexpr int global_threads_num = BLOCKS * THREADS;

  const int wf_id_in_block = blockIdx.x / WF_SIZE;         // wavefront id in block
  const int block_id = blockIdx.x;                         // block id
  const int threads_in_block = blockDim.x;                 // threads in one block
  const int tid_in_block = threadIdx.x % threads_in_block; // thread id in one block

  constexpr unsigned int shared_len = THREADS * R; // 64 * 1024 / (BLOCKS / 64) / sizeof(T); // nnz per block
  __shared__ T shared_val[shared_len];
  constexpr int nnz_per_block = THREADS * R;
  const I last_element_index = row_offset[m];

  I bp_index = block_id;
  // each block process THREADS * R elements.
  for (int k = nnz_per_block * block_id; k < last_element_index; k += BLOCKS * nnz_per_block) {
    for (I i = tid_in_block; i < nnz_per_block; i += THREADS) {
      const I index = min(k + i, last_element_index - 1);
      shared_val[i] = csr_val[index] * x[csr_col_ind[index]];
    }
    __syncthreads();

    // reduce via LDS.
    const I reduce_start_row_id = min(break_points[bp_index], m);
    I reduce_end_row_id = min(break_points[bp_index + 1], m);
    // if it is the last block
    if (reduce_end_row_id == 0) {
      reduce_end_row_id = m;
    }
    // if start of the next block cuts some row.
    // or some row cross multiple blocks.
    if (row_offset[reduce_end_row_id] % nnz_per_block != 0 || reduce_end_row_id == reduce_start_row_id) {
      reduce_end_row_id = min(reduce_end_row_id + 1, m); // make sure `reduce_end_row_id` is not large than m
    }

    I reduce_row_id = reduce_start_row_id + tid_in_block;
    if (reduce_row_id < reduce_end_row_id) {
      T sum = static_cast<T>(0);
      // what if it has a very long row? which means `reduce_start_row_id == reduce_end_row_id`.
      const I reduce_start_inx = max(0, row_offset[reduce_row_id] - bp_index * nnz_per_block);
      const I reduce_end_inx = min(nnz_per_block, row_offset[reduce_row_id + 1] - bp_index * nnz_per_block);
      for (int i = reduce_start_inx; i < reduce_end_inx; i++) {
        sum += shared_val[i];
      }
      atomicAdd(y + reduce_row_id, alpha * sum);
      // y[reduce_row_id] = device_fma(beta, y[reduce_row_id], alpha * sum);
    }
    __syncthreads();
    bp_index += BLOCKS;
  }
}

template <int BREAK_STRIDE, int BLOCKS, typename I>
__global__ void pre_calc_break_point(const I *__restrict__ row_ptr, const I m, I *__restrict__ break_points,
                                     const I bp_len) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int global_threads_num = blockDim.x * gridDim.x;

  constexpr I break_stride = BREAK_STRIDE;
  if (global_thread_id == 0) {
    break_points[0] = 0; // start row of the block 0 and the first round.
  }

  for (int i = global_thread_id; i < m; i += global_threads_num) {
    // for first element of row i and row i+1, they belong to different blocks.
    if (row_ptr[i] / break_stride != row_ptr[i + 1] / break_stride) { // fixme: step may be not 1
      // record the row id of the first element in the block.
      // note: a row can cross multiple blocks.
      for (int j = row_ptr[i] / break_stride + 1; j <= row_ptr[i + 1] / break_stride; j++) {
        break_points[j] = i;
      }
      if (row_ptr[i + 1] % break_stride == 0) {
        break_points[row_ptr[i + 1] / break_stride] += 1;
      }
    }
  }
}

#define FLAT_KERNEL_WRAPPER(R, BLOCKS, THREADS)                                                                        \
  (spmv_flat_kernel<64, R, BLOCKS, THREADS, int, double>)<<<BLOCKS, THREADS>>>(m, alpha, beta, rowptr, break_points,   \
                                                                               colindex, value, x, y)
