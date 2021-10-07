//
// Created by chu genshen on 2021/10/4.
//

#ifndef SPMV_ACC_LINE_ENHANCE_REDUCE_HPP
#define SPMV_ACC_LINE_ENHANCE_REDUCE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

constexpr int LE_REDUCE_OPTION_DIRECT = 0;
constexpr int LE_REDUCE_OPTION_VEC = 1;
constexpr int LE_REDUCE_OPTION_VEC_MEM_COALESCING = 2;

constexpr int DEFAULT_LE_REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;

/**
 * direct reduction method: each thread perform reduction for one row in matrix.
 * @tparam I type of index
 * @tparam T type of float number data
 * @param reduce_row_id the row id for reduction for current thread.
 * @param block_row_end the end row id (not include) of this HIP block in calculation
 * @param reduce_row_idx_begin row_offset[\param reduce_row_id]
 * @param reduce_row_idx_end row_offset[\param reduce_row_id + 1]
 * @param block_round_inx_start start index of matrix values of current round in current HIP block.
 * @param block_round_inx_end end index of matrix values of current round in current HIP block.
 * @param shared_val LDS address
 * @param sum partial sum
 * @return
 */
template <typename I, typename T>
__device__ __forceinline__ void line_enhance_direct_reduce(const I reduce_row_id, const I block_row_end,
                                                           const I reduce_row_idx_begin, const I reduce_row_idx_end,
                                                           const I block_round_inx_start, const I block_round_inx_end,
                                                           const T *shared_val, T &sum) {
  if (reduce_row_id < block_row_end) {
    if (reduce_row_idx_begin < block_round_inx_end && reduce_row_idx_end > block_round_inx_start) {
      const I reduce_start = max(reduce_row_idx_begin, block_round_inx_start);
      const I reduce_end = min(reduce_row_idx_end, block_round_inx_end);
      for (I j = reduce_start; j < reduce_end; j++) {
        sum += shared_val[j - block_round_inx_start];
      }
    }
  }
}

/**
 * direct and vector reduction method: each vector perform reduction for one row in matrix.
 * If the vector size is set to 1, it means one thread in block reduce one row.
 * @tparam I type of index
 * @tparam T type of float number data
 * @tparam VECTOR_SIZE threads number in a vector for reduction.
 * @oaram: tid_in_vec thread id in vector.
 * @note: other parameter keep the same as device function line_enhance_direct_reduce.
 */
template <typename I, typename T, int VECTOR_SIZE>
__device__ __forceinline__ void line_enhance_vec_reduce(const I reduce_row_id, const I block_row_end,
                                                        const I reduce_row_idx_begin, const I reduce_row_idx_end,
                                                        const I block_round_inx_start, const I block_round_inx_end,
                                                        const T *shared_val, T &sum, const int tid_in_vec) {
  if (reduce_row_id < block_row_end) {
    if (reduce_row_idx_begin < block_round_inx_end && reduce_row_idx_end > block_round_inx_start) {
      T local_sum = static_cast<T>(0);
      // reduce from LDS
      const I reduce_start = max(reduce_row_idx_begin, block_round_inx_start);
      const I reduce_end = min(reduce_row_idx_end, block_round_inx_end);
      for (I j = reduce_start + tid_in_vec; j < reduce_end; j += VECTOR_SIZE) {
        local_sum += shared_val[j - block_round_inx_start];
      }
      // reduce from lanes in a vector
      if (VECTOR_SIZE > 1) { // in fact, this branch is unnecessary.
#pragma unroll
        for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
          local_sum += __shfl_down(local_sum, i, VECTOR_SIZE);
        }
      }
      sum += local_sum;
    }
  }
}

#endif // SPMV_ACC_LINE_ENHANCE_REDUCE_HPP
