//
// Created by genshen on 2021/7/28.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "hip-flat/spmv_hip_acc_imp.h"
#include "hip-thread-row/thread_row.h"
#include "hip-vector-row/vector_row.h"

void adaptive_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *row_ptr,
                          const int *col_index, const double *value, const double *x, double *y) {
  // 0. sampling and data block dividing.
  // If it need data block dividing, just use vector method,
  // but with different vector size for different data block.
  const int bp_0 = row_ptr[m / 4];
  const int bp_1 = row_ptr[m / 2];
  const int bp_2 = row_ptr[3 * m / 4];
  const int bp_3 = row_ptr[m];

  const int avg_nnz_per_row = bp_3 / m;
  const int nnz_block_0 = bp_1 - 0;
  const int nnz_block_1 = bp_3 - bp_1;

  // 1. divided into 2 data blocks if 2 data blocks have large difference.
  if ((nnz_block_1 > nnz_block_0 && nnz_block_1 / nnz_block_0 >= 4) ||
      (nnz_block_0 > nnz_block_1 && nnz_block_0 / nnz_block_1 >= 4)) {
    // vector-row based data blocks dividing
    // use nnz as weight of each block
    adaptive_vec_row_sparse_spmv(nnz_block_0, nnz_block_1, trans, alpha, beta, m, n, row_ptr, col_index, value, x, y);
    return;
  }

  // 2. otherwise, use picker strategy by average nnz per row.
  if (avg_nnz_per_row <= 4) {
    // use thread-row strategy.
    thread_row_sparse_spmv(trans, alpha, beta, m, n, row_ptr, col_index, value, x, y);
    return;
  }

  // 3. If non-zeros number of large enough, we use flat strategy.
  // The flat strategy has a data pre-processing kernel function,
  // which can not be applied to small data set.
  if (bp_3 > (1 << 24)) { // 16,777,216
    flat_sparse_spmv(trans, alpha, beta, m, n, row_ptr, col_index, value, x, y);
    return;
  }

  // 4. use vector-row method for small data set.
  // todo: pass nnz/row.
  vec_row_sparse_spmv(trans, alpha, beta, m, n, row_ptr, col_index, value, x, y);
}
