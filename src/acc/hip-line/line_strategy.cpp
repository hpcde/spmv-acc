//
// Created by genshen on 2021/7/15.
//

#include "line_strategy.h"

void line_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                      const double *value, const double *x, double *y) {
  // const int avg_eles_per_row = ceil(rowptr[m] + 0.0 / m);
  const int avg_eles_per_row = rowptr[m] / m;
  // ROW_NUM * MAX_NNZ_NUM < HIP_THREAD
  constexpr int HIP_THREAD = 512;
  constexpr int R = 2;
  if (avg_eles_per_row <= 5) {
    constexpr int MAX_NNZ_NUM = 5;
    const int ROW_NUM = HIP_THREAD / MAX_NNZ_NUM * R;
    const int HIP_BLOCK = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
    LINE_ONE_PASS_KERNEL_WRAPPER(ROW_NUM, MAX_NNZ_NUM, HIP_BLOCK, HIP_THREAD);
  } else {
    constexpr int MAX_NNZ_NUM = 14;
    const int ROW_NUM = HIP_THREAD / MAX_NNZ_NUM * R;
    const int HIP_BLOCK = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
    LINE_ONE_PASS_KERNEL_WRAPPER(ROW_NUM, MAX_NNZ_NUM, HIP_BLOCK, HIP_THREAD);
  }
  // LINE_KERNEL_WRAPPER(5);
}
