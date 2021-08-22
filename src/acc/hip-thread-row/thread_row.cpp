//
// Created by genshen on 2021/7/15.
//

#include "thread_row.h"
#include "building_config.h"
#include "thread_row_config.h"

void thread_row_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *d_row_ptr,
                            const int *d_csr_col_index, const double *d_csr_value, const double *d_x, double *d_y) {
  const int avg_nnz_per_row = d_row_ptr[m] / m;
  if (avg_nnz_per_row <= 4) {
    constexpr int MAX_ROW_NNZ = 5; // 5 is up bound.

#if defined OPT_THREAD_ROW_BLOCK_LEVEL // thread-row block level
    constexpr int BLOCKS = 112 * AVAILABLE_CU;
    (kernel_thread_row_block_level<1, MAX_ROW_NNZ, 64, 256, int, double>)<<<BLOCKS, 256>>>(
        alpha, beta, m, d_row_ptr, d_csr_col_index, d_csr_value, d_x, d_y);
#elif defined OPT_THREAD_ROW_REMAP_VEC_X_BLOCK_LEVEL // thread-row block level with x remapping
    constexpr int BLOCKS = 112 * AVAILABLE_CU;
    (kernel_thread_row_block_v2<1, MAX_ROW_NNZ, 64, 512, int, double>)<<<BLOCKS, 512>>>(
        alpha, beta, m, d_row_ptr, d_csr_col_index, d_csr_value, d_x, d_y);
#elif defined OPT_THREAD_ROW_REMAP_VEC_X             // thread-row wavefront level with x remapping
    constexpr int BLOCKS = 112 * AVAILABLE_CU;
    (kernel_thread_row_v2<1, MAX_ROW_NNZ, 64, 256, int, double>)<<<BLOCKS, 256>>>(
        alpha, beta, m, d_row_ptr, d_csr_col_index, d_csr_value, d_x, d_y);
#else                                                // thread-row wavefront level
    // fallback to normal one.
    constexpr int BLOCKS = 53 * AVAILABLE_CU;
    (kernel_thread_row<1, MAX_ROW_NNZ, 64, 256, int, double>)<<<BLOCKS, 256>>>(alpha, beta, m, d_row_ptr,
                                                                               d_csr_col_index, d_csr_value, d_x, d_y);
#endif
  } else {
    native_thread_row<<<128, 1024>>>(trans, alpha, beta, m, n, d_row_ptr, d_csr_col_index, d_csr_value, d_x, d_y);
  }
  return;
}
