//
// Created by genshen on 2021/12/4.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/utils.h"
#include "../hip-vector-row/vector_row_native.hpp" // use vector-row reduction
#include "building_config.h"
#include "line_config.h"
#include "line_imp_one_pass.inl"

/**
 * run adaptive line method.
 * In this version, it can switch between line and vector-row method.
 * In the adaptive line, each hip block is assigned equal number of rows.
 * If the nnz of those rows is larger than the LDS size that is available for this hip block,
 * vector-row method will be used, otherwise, traditional line method will be used.
 * @tparam ROW_SIZE rows number for calculation by this hip block.
 * @tparam BLOCK_LDS_SIZE LDS size for a block to store temp values.
 * @tparam THREADS threads number per block in kernel config.
 * @tparam WF_SIZE threads number in a wavefront
 * @tparam VECTOR_SIZE thread number in a vector
 * @tparam I integer index type
 * @tparam T float point type
 * @tparam NO_LDS_EXCEED True: api caller can make sure the data in a block
 *         will not exceed the LDS size `BLOCK_LDS_SIZE`.
 * @param m total rows number
 * @param alpha,beta: alpha and beta value in y=alpha*Ax+beta*y
 * @param row_offset: row offset in CSR format.
 * @param csr_col_ind column index in CSR format
 * @param csr_val values array in CSR format
 * @param x vector x in y=alpha*Ax+beta*y
 * @param y vector y in y=alpha*Ax+beta*y
 * @return
 */
template <int ROW_SIZE, int BLOCK_LDS_SIZE, int THREADS, int WF_SIZE, int VECTOR_SIZE, typename I, typename T,
          bool NO_LDS_EXCEED>
__global__ void spmv_adaptive_line_kernel(const I m, const T alpha, const T beta, const I *row_offset,
                                          const I *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int block_id = blockIdx.x;                                 // global block id
  const int block_thread_num = blockDim.x;                         // threads num in a block
  const int block_thread_id = global_thread_id % block_thread_num; // local thread id in current block
  constexpr int shared_len = BLOCK_LDS_SIZE;

  __shared__ T shared_val[shared_len];
  const I block_row_begin = block_id * ROW_SIZE;
  const I block_row_end = min(block_row_begin + ROW_SIZE, m);
  // load val to lds parallel
  const I block_row_idx_begin = row_offset[block_row_begin];
  const I block_row_idx_end = row_offset[block_row_end];
  const I n_values_load = block_row_idx_end - block_row_idx_begin;

  if (NO_LDS_EXCEED || n_values_load <= shared_len) {
    line_one_pass_kernel<I, T>(block_thread_num, block_thread_id, block_row_begin, block_row_end, block_row_idx_end,
                               block_row_idx_begin, shared_val, m, alpha, beta, row_offset, csr_col_ind, csr_val, x, y);
    // call line directly.
  } else {
    // If LDS size is exceed, call vector-row.
    const int vector_thread_id = block_thread_id % VECTOR_SIZE; // local thread id in current vector
    const int vector_id = block_thread_id / VECTOR_SIZE;        // vector id in block
    const int vector_num = blockDim.x / VECTOR_SIZE;            // total vectors in block

    const int tid_in_wf = block_thread_id % WF_SIZE;
    const int wf_id = block_thread_id / WF_SIZE;

    __shared__ T lds_y[THREADS / VECTOR_SIZE]; // todo: USE `shared_val`

    for (int row = block_row_begin + vector_id; row < block_row_end; row += vector_num) {
      const int row_start = row_offset[row];
      const int row_end = row_offset[row + 1];
      T sum = static_cast<T>(0);

      for (int i = row_start + vector_thread_id; i < row_end; i += VECTOR_SIZE) {
        asm_v_fma_f64(csr_val[i], device_ldg(x + csr_col_ind[i]), sum);
      }

      // reduce inside a vector
      for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
        sum += __shfl_down(sum, i, VECTOR_SIZE);
      }

      block_store_y_with_coalescing<THREADS, VECTOR_SIZE, WF_SIZE, T>(tid_in_wf, global_thread_id, row, m, alpha, beta,
                                                                      sum, y, y, lds_y);
    }
  }
}
