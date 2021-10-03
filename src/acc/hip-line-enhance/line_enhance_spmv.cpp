//
// Created by chu genshen on 2021/10/2.
//

#include "line_enhance_spmv_imp.h"

void line_enhance_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                              const int *colindex, const double *value, const double *x, double *y) {
  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int ROWS_PER_BLOCK = 32;
  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  LINE_ENHANCE_KERNEL_WRAPPER(ROWS_PER_BLOCK, R, BLOCKS, THREADS_PER_BLOCK);
}
