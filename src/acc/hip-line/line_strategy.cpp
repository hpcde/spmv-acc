//
// Created by genshen on 2021/7/15.
//

#include "line_strategy.h"

void line_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                      const double *value, const double *x, double *y) {
  LINE_KERNEL_WRAPPER(5);
}
