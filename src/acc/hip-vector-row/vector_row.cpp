//
// Created by genshen on 2021/7/15.
//

#include "vector_row.h"

void vec_row_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                         const int *colindex, const double *value, const double *x, double *y) {
  //  const int avg_eles_per_row = ceil(rowptr[m] + 0.0 / m);
  const int avg_eles_per_row = rowptr[m] / m;

  if (avg_eles_per_row <= 4) {
    NATIVE_VECTOR_KERNEL_WRAPPER(2);
  } else if (avg_eles_per_row <= 8) {
    NATIVE_VECTOR_KERNEL_WRAPPER(4);
  } else if (avg_eles_per_row <= 16) {
    VECTOR_KERNEL_WRAPPER(8);
  } else if (avg_eles_per_row <= 32) {
    NATIVE_VECTOR_KERNEL_WRAPPER(16);
  } else if (avg_eles_per_row <= 64) {
    NATIVE_VECTOR_KERNEL_WRAPPER(32);
  } else {
    NATIVE_VECTOR_KERNEL_WRAPPER(64);
  }
}

void adaptive_vec_row_sparse_spmv(const int weight_block_0, const int weight_block_1, int trans, const int alpha,
                                  const int beta, int m, int n, const int *rowptr, const int *colindex,
                                  const double *value, const double *x, double *y) {
  const int avg_eles_per_row = rowptr[m] / m;
  const int bp = static_cast<int>(round((16.0 * weight_block_0) / (weight_block_0 + weight_block_1)));
  const int rescaled_bp = std::min(16 - 1, std::max(1, bp)); // up and low boundary.
  ADAPTIVE_VECTOR_KERNEL_WRAPPER(2, rescaled_bp); // N is data blocks.
}
