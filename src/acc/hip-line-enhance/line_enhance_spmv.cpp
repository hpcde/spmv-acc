//
// Created by chu genshen on 2021/10/2.
//

#include "line_enhance_spmv_imp.h"

void line_enhance_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                              const int *colindex, const double *value, const double *x, double *y) {
  constexpr int R = 2;
  constexpr int ROWS_PER_BLOCK = 32; // note: make sure ROWS_PER_BLOCK * VEC_SIZE <= THREADS_PER_BLOCK.

  constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
  constexpr int VEC_SIZE = 4; // note: if using direct reduce, VEC_SIZE must set to 1.

  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  constexpr int THREADS_PER_BLOCK = 512;
  LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
}
