//
// Created by genshen on 2021/7/15.
//

#include "spmv_hip_acc_imp.h"

void flat_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                      const double *value, const double *x, double *y) {
  constexpr int blocks = 512;
  constexpr int threads_per_block = 512;
  int *break_points;
  // the nnz is rowptr[m], in one round, it can process about `blocks * threads_per_block` nnz.
  const int total_rounds =
      rowptr[m] / (threads_per_block * blocks) + (rowptr[m] % (threads_per_block * blocks) == 0 ? 0 : 1);
  // each round and each block both have a break point.
  const int break_points_len = total_rounds * blocks + 1;
  hipMalloc((void **)&break_points, break_points_len * sizeof(int));
  hipMemset(break_points, 0, break_points_len * sizeof(int));

  constexpr int R = 2;
  (pre_calc_break_point<R * threads_per_block, blocks, int>)<<<1024, 512>>>(rowptr, m, break_points, break_points_len);
  FLAT_KERNEL_WRAPPER(R, blocks, threads_per_block);
}
