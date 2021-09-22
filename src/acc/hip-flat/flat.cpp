//
// Created by genshen on 2021/7/15.
//

#include "flat_config.h"
#include "spmv_hip_acc_imp.h"

template <int R, int blocks, int threads_per_block>
inline void flat_multi_pass_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                                        const int *colindex, const double *value, const double *x, double *y) {
  int *break_points;
  // the nnz is rowptr[m], in one round, it can process about `blocks * R * threads_per_block` nnz.
  const int total_rounds =
      rowptr[m] / (R * threads_per_block * blocks) + (rowptr[m] % (R * threads_per_block * blocks) == 0 ? 0 : 1);
  // each round and each block both have a break point.
  const int break_points_len = total_rounds * blocks + 1;
  hipMalloc((void **)&break_points, break_points_len * sizeof(int));
  hipMemset(break_points, 0, break_points_len * sizeof(int));

  (pre_calc_break_point<R * threads_per_block, blocks, int>)<<<1024, 512>>>(rowptr, m, break_points, break_points_len);
  FLAT_KERNEL_WRAPPER(R, blocks, threads_per_block);
}

template <int R, int THREADS_PER_BLOCK, int REDUCE_OPTION>
inline void flat_one_pass_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                                      const int *colindex, const double *value, const double *x, double *y) {
  // each block can process `R*THREADS_PER_BLOCK` non-zeros.
  const int HIP_BLOCKS = rowptr[m] / (R * THREADS_PER_BLOCK) + ((rowptr[m] % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);
  constexpr int TOTAL_ROUNDS = 1;
  // each round and each block both have a break point.
  const int break_points_len = TOTAL_ROUNDS * HIP_BLOCKS + 1;
  int *break_points;
  hipMalloc((void **)&break_points, break_points_len * sizeof(int));
  hipMemset(break_points, 0, break_points_len * sizeof(int));

  // template parameter `BLOCKS` is not used, thus, we can set it to 0.
  (pre_calc_break_point<R * THREADS_PER_BLOCK, 0, int>)<<<1024, 512>>>(rowptr, m, break_points, break_points_len);
  FLAT_KERNEL_ONE_PASS_WRAPPER(REDUCE_OPTION, R, HIP_BLOCKS, THREADS_PER_BLOCK);
}

void flat_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                      const double *value, const double *x, double *y) {
  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  if (FLAT_ONE_PASS) {
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;
    flat_one_pass_sparse_spmv<R, THREADS_PER_BLOCK, RED_OPT>(trans, alpha, beta, m, n, rowptr, colindex, value, x, y);
  } else {
    constexpr int blocks = 512;
    flat_multi_pass_sparse_spmv<R, blocks, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, rowptr, colindex, value, x, y);
  }
}
