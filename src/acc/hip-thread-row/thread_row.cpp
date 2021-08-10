//
// Created by genshen on 2021/7/15.
//

#include "thread_row.h"
#include "thread_row_config.h"

void thread_row_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *d_row_ptr,
                            const int *d_csr_col_index, const double *d_csr_value, const double *d_x, double *d_y) {
  const int avg_nnz_per_row = d_row_ptr[m] / m;
  if (avg_nnz_per_row <= 4) {
    constexpr int MAX_ROW_NNZ = 5; // 5 is up bound.
#ifndef OPT_THREAD_ROW_REMAP_VEC_X
    constexpr int BLOCKS = 53 * 60;
    (kernel_thread_row<1, MAX_ROW_NNZ, 64, 256, int, double>)<<<BLOCKS, 256>>>(alpha, beta, m, d_row_ptr,
                                                                               d_csr_col_index, d_csr_value, d_x, d_y);
#endif
#ifdef OPT_THREAD_ROW_REMAP_VEC_X
    constexpr int BLOCKS = 53 * 60;
    (kernel_thread_row_v2<1, MAX_ROW_NNZ, 64, 256, int, double>)<<<BLOCKS, 256>>>(
        alpha, beta, m, d_row_ptr, d_csr_col_index, d_csr_value, d_x, d_y);
#endif
  } else {
    native_thread_row<<<1, 1024>>>(trans, alpha, beta, m, n, d_row_ptr, d_csr_col_index, d_csr_value, d_x, d_y);
  }
  return;
}
