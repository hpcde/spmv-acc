//
// Created by genshen on 2025/2/20.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/utils.h"
#include "../hip-line-enhance/line_enhance_reduce.hpp"
#include "building_config.h"

template <int REDUCE_OPTION, int WF_SIZE, int VEC_SIZE, int MAX_ROWS_PER_BLOCK, int R, int THREADS, typename I,
          typename T>
__device__ __forceinline__ void line_enhance_plus(const I m, const I bp[], const T alpha, const T beta,
                                                  const I *__restrict__ row_offset, const I *__restrict__ csr_col_ind,
                                                  const T *__restrict__ csr_val, const T *__restrict__ x,
                                                  T *__restrict__ y, int g_bid, I block_row_begin, I block_row_end);

template <int REDUCE_OPTION, int WF_SIZE, int MIN_NNZ_PER_BLOCK, int R, int THREADS, typename I, typename T>
__device__ __forceinline__ void
line_enhance_plus_shared_block(const I m, const T alpha, const T beta, const I *__restrict__ row_offset,
                               const I *__restrict__ csr_col_ind, const T *__restrict__ csr_val,
                               const T *__restrict__ x, T *__restrict__ y, int g_bid, const I block_row_begin,
                               const I block_row_end, const I first_row_block_id);

/**
 * The csr-adaptive plus kernel.
 */
template <int REDUCE_OPTION, int WF_SIZE, int VEC_SIZE, int MAX_ROWS_PER_BLOCK, int MIN_NNZ_PER_BLOCK, int R,
          int THREADS, typename I, typename T>
__global__ void line_enhance_plus_kernel(const I m, const I bp[], const I row_1st_block_id[], const T alpha,
                                         const T beta, const I *__restrict__ row_offset,
                                         const I *__restrict__ csr_col_ind, const T *__restrict__ csr_val,
                                         const T *__restrict__ x, T *__restrict__ y) {
  const int g_bid = blockIdx.x; // global block id

  const I block_row_begin = bp[g_bid];
  const I block_row_end = min(bp[g_bid + 1], m);

  const I first_row_block_id_with_bit_flag = row_1st_block_id[block_row_begin];
  const I first_row_block_id = first_row_block_id_with_bit_flag / 2;
  if ((first_row_block_id_with_bit_flag & 1) == 0) {
    line_enhance_plus<REDUCE_OPTION, WF_SIZE, VEC_SIZE, MAX_ROWS_PER_BLOCK, R, THREADS, I, T>(
        m, bp, alpha, beta, row_offset, csr_col_ind, csr_val, x, y, g_bid, block_row_begin, block_row_end);
  } else { // it is very long row
    line_enhance_plus_shared_block<REDUCE_OPTION, WF_SIZE, MIN_NNZ_PER_BLOCK, R, THREADS, I, T>(
        m, alpha, beta, row_offset, csr_col_ind, csr_val, x, y, g_bid, block_row_begin, block_row_end,
        first_row_block_id);
  }
}

template <int REDUCE_OPTION, int WF_SIZE, int MIN_NNZ_PER_BLOCK, int R, int THREADS, typename I, typename T>
__device__ __forceinline__ void
line_enhance_plus_shared_block(const I m, const T alpha, const T beta, const I *__restrict__ row_offset,
                               const I *__restrict__ csr_col_ind, const T *__restrict__ csr_val,
                               const T *__restrict__ x, T *__restrict__ y, int g_bid, const I block_row_begin,
                               const I block_row_end, const I first_row_block_id) {
  const int g_tid = threadIdx.x + blockDim.x * blockIdx.x; // global thread id
  const int tid_in_block = g_tid % THREADS;                // local thread id in current block

  const bool is_last_row_bolck = (block_row_begin != block_row_end);
  const I block_row_idx_start = row_offset[block_row_begin] + (g_bid - first_row_block_id) * MIN_NNZ_PER_BLOCK;
  // block_row_begin may equals to block_row_end if a raw spaning multiple rows.
  // we calculate `block_row_idx_end` with `max`.
  const I block_row_idx_end = max(block_row_idx_start + MIN_NNZ_PER_BLOCK, row_offset[block_row_end]);

  T sum = static_cast<T>(0);
  const int rounds = (block_row_idx_end - block_row_idx_start) / (R * THREADS) +
                     ((block_row_idx_end - block_row_idx_start) % (R * THREADS) == 0 ? 0 : 1);
  for (int r = 0; r < rounds; r++) {
    // start and end data index in each round
    const I block_round_inx_start = block_row_idx_start + r * R * THREADS;
    const I block_round_inx_end = min(block_round_inx_start + R * THREADS, block_row_idx_end);
    I i = block_round_inx_start + tid_in_block;

    __syncthreads();
// in each inner loop, it processes R*THREADS element at max
#pragma unroll
    for (int k = 0; k < R; k++) {
      if (i < block_row_idx_end) {
        const T tmp = csr_val[i] * x[csr_col_ind[i]];
        sum += tmp; // dirrerent from line-enhance method, we sum to the result directly.
      }
      i += THREADS;
    }
    __syncthreads();
  }

  // performance reduce inside a block
  // 1: recuce inside  wavefront first.
  line_enhance_vec_local_shift<I, T, WF_SIZE>(sum);

  // 2: reduce among wavefronts inside a block.
  constexpr int wavefronts_num = THREADS / WF_SIZE;
  __shared__ T shared_val[wavefronts_num];
  const int wf_id_in_block = tid_in_block / WF_SIZE;
  const int tid_in_wf = tid_in_block % WF_SIZE;
  if (tid_in_wf == 0) {
    shared_val[wf_id_in_block] = sum;
  }
  __syncthreads();
  // use a wavefront to reduce from LDS is encough
  T reduced_sum = 0;
  if (wf_id_in_block == 0 && tid_in_wf < wavefronts_num) {
    reduced_sum = shared_val[tid_in_wf];
  }
  // number of `wavefronts_num` threads as a vector for reduction.
  line_enhance_vec_local_shift<I, T, wavefronts_num>(reduced_sum);
  if (tid_in_block == 0) {
    reduced_sum = alpha * reduced_sum; // alpha * Ax
    if (is_last_row_bolck) {
      // process beta*y in the last block of this row.
      // Note: y[i] can be non-zero, we need add a new kernel to set beta*y before this kernel.
      // reduced_sum += beta * y[block_row_begin];
    }
    atomicAdd(y + block_row_begin, reduced_sum); // alpha * Ax + beta * y
  }
}

template <int REDUCE_OPTION, int WF_SIZE, int VEC_SIZE, int MAX_ROWS_PER_BLOCK, int R, int THREADS, typename I,
          typename T>
__device__ __forceinline__ void line_enhance_plus(const I m, const I bp[], const T alpha, const T beta,
                                                  const I *__restrict__ row_offset, const I *__restrict__ csr_col_ind,
                                                  const T *__restrict__ csr_val, const T *__restrict__ x,
                                                  T *__restrict__ y, int g_bid, I block_row_begin, I block_row_end) {
  const int g_tid = threadIdx.x + blockDim.x * blockIdx.x; // global thread id
  const int tid_in_block = g_tid % THREADS;                // local thread id in current block

  constexpr int shared_len = THREADS * R;
  __shared__ T shared_val[shared_len];
  // __shared__ T shared_index[THREADS];

  const I block_row_idx_start = row_offset[block_row_begin];
  const I block_row_idx_end = row_offset[block_row_end];

  // vector reduce, if VEC_SIZE is set to 1, it will be direct reduction.
  const I vec_id_in_block = g_tid / VEC_SIZE % (THREADS / VEC_SIZE);
  const I tid_in_vec = g_tid % VEC_SIZE;
  // load reduce row bound
  const I reduce_row_id = block_row_begin + vec_id_in_block;
  I reduce_row_idx_begin = 0;
  I reduce_row_idx_end = 0;
  if (reduce_row_id < block_row_end) {
    reduce_row_idx_begin = row_offset[reduce_row_id];
    reduce_row_idx_end = row_offset[reduce_row_id + 1];
  }

  T sum = static_cast<T>(0);
  const int rounds = (block_row_idx_end - block_row_idx_start) / (R * THREADS) +
                     ((block_row_idx_end - block_row_idx_start) % (R * THREADS) == 0 ? 0 : 1);
  for (int r = 0; r < rounds; r++) {
    // start and end data index in each round
    const I block_round_inx_start = block_row_idx_start + r * R * THREADS;
    const I block_round_inx_end = min(block_round_inx_start + R * THREADS, block_row_idx_end);
    I i = block_round_inx_start + tid_in_block;

    __syncthreads();
// in each inner loop, it processes R*THREADS element at max
#pragma unroll
    for (int k = 0; k < R; k++) {
      if (i < block_row_idx_end) {
        const T tmp = csr_val[i] * x[csr_col_ind[i]];
        shared_val[i - block_round_inx_start] = tmp;
      }
      i += THREADS;
    }

    __syncthreads();
    // reduce
    if (REDUCE_OPTION == LE_REDUCE_OPTION_DIRECT) {
      line_enhance_direct_reduce<I, T>(reduce_row_id, block_row_end, reduce_row_idx_begin, reduce_row_idx_end,
                                       block_round_inx_start, block_round_inx_end, shared_val, sum);
    }
    if (REDUCE_OPTION == LE_REDUCE_OPTION_VEC || REDUCE_OPTION == LE_REDUCE_OPTION_VEC_MEM_COALESCING) {
      sum += line_enhance_vec_reduce<I, T, VEC_SIZE>(reduce_row_id, block_row_end, reduce_row_idx_begin,
                                                     reduce_row_idx_end, block_round_inx_start, block_round_inx_end,
                                                     shared_val, tid_in_vec);
    }
    // if (REDUCE_OPTION == CSR_ADAPTICE_REDUCE_OPTION_BLOCK_ONE_ROW) {
    //   shared_index[tid_in_block] = ;
    // }
  }
  // store result
  if (REDUCE_OPTION == LE_REDUCE_OPTION_DIRECT) {
    if (reduce_row_id < block_row_end) {
      y[reduce_row_id] = alpha * sum + y[reduce_row_id];
    }
  }
  if (REDUCE_OPTION == LE_REDUCE_OPTION_VEC) {
    line_enhance_vec_local_shift<I, T, VEC_SIZE>(sum);
    if (reduce_row_id < block_row_end && tid_in_vec == 0) {
      y[reduce_row_id] = alpha * sum + y[reduce_row_id]; // todo: beta
    }
  }
  if (REDUCE_OPTION == LE_REDUCE_OPTION_VEC_MEM_COALESCING) {
    const I thread_reduce_row_id = block_row_begin + tid_in_block;
    line_enhance_vec_local_shift<I, T, VEC_SIZE>(sum);
    sum = line_enhance_vec_global_shift<I, T, THREADS / VEC_SIZE>(tid_in_block, vec_id_in_block, tid_in_vec, shared_val,
                                                                  sum);
    if (thread_reduce_row_id < block_row_end) {
      y[thread_reduce_row_id] = alpha * sum + y[thread_reduce_row_id];
    }
  }
}

#define LE_PLUS_KERNEL_WRAPPER(REDUCE, MAX_ROWS_PER_BLOCK, MIN_NNZ_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS)            \
  (line_enhance_plus_kernel<REDUCE, __WRAP_SIZE__, VEC_SIZE, MAX_ROWS_PER_BLOCK, MIN_NNZ_PER_BLOCK, R, THREADS, int,   \
                            double>)<<<BLOCKS, THREADS>>>(m, break_points, first_process_block_of_row, alpha, beta,    \
                                                          rowptr, colindex, value, x, y);
