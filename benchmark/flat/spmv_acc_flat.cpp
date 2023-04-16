//
// Created by reget on 2021/11/21.
// this file is copied and modified from src/acc/flat/flat.cpp for adding timer code.
//

#ifndef SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP
#define SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP

#include <iostream>

#include "spmv_acc_flat.h"
#include "../utils/benchmark_time.h"
#include "common/macros.h"
#include "hip-flat/flat_config.h"
#include "hip-flat/spmv_hip_acc_imp.h"
#include "timer.h"
#include "benchmark_config.h"

template <int R, int REDUCE_OPTION, int REDUCE_VEC_SIZE, int BLOCKS, int THREADS_PER_BLOCK>
inline void flat_multi_pass_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, int nnz,
                                        const int *rowptr, const int *colindex, const double *value, const double *x,
                                        double *y, BenchmarkTime *bmt) {
  my_timer pre_timer, calc_timer;
  pre_timer.start();
  int *break_points;
  // the nnz is rowptr[m], in one round, it can process about `blocks * R * threads_per_block` nnz.
  const int total_rounds =
      nnz / (R * THREADS_PER_BLOCK * BLOCKS) + (nnz % (R * THREADS_PER_BLOCK * BLOCKS) == 0 ? 0 : 1);
  // each round and each block both have a break point.
  const int break_points_len = total_rounds * BLOCKS + 1;
  hipMalloc((void **)&break_points, break_points_len * sizeof(int));
  hipMemset(break_points, 0, break_points_len * sizeof(int));
  (pre_calc_break_point<R * THREADS_PER_BLOCK, BLOCKS, int>)<<<1024, 512>>>(rowptr, m, break_points, break_points_len);
  lazy_device_sync();
  pre_timer.stop();

  calc_timer.start();
  FLAT_KERNEL_WRAPPER(R, REDUCE_OPTION, REDUCE_VEC_SIZE, BLOCKS, THREADS_PER_BLOCK);
  lazy_device_sync(true);
  calc_timer.stop();
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.);
  }
}

template <int R, int REDUCE_OPTION, int REDUCE_VEC_SIZE, int THREADS_PER_BLOCK>
inline void flat_one_pass_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, int nnz,
                                      const int *rowptr, const int *colindex, const double *value, const double *x,
                                      double *y, BenchmarkTime *bmt) {
  my_timer pre_timer, calc_timer;
  pre_timer.start();
  // each block can process `R*THREADS_PER_BLOCK` non-zeros.
  const int HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);
  constexpr int TOTAL_ROUNDS = 1;
  // each round and each block both have a break point.
  const int break_points_len = TOTAL_ROUNDS * HIP_BLOCKS + 1;
  int *break_points;
  hipMalloc((void **)&break_points, break_points_len * sizeof(int));
  hipMemset(break_points, 0, break_points_len * sizeof(int));

  // template parameter `BLOCKS` is not used, thus, we can set it to 0.
  (pre_calc_break_point<R * THREADS_PER_BLOCK, 0, int>)<<<1024, 512>>>(rowptr, m, break_points, break_points_len);
  lazy_device_sync();
  pre_timer.stop();
  calc_timer.start();
  FLAT_KERNEL_ONE_PASS_WRAPPER(R, REDUCE_OPTION, REDUCE_VEC_SIZE, HIP_BLOCKS, THREADS_PER_BLOCK);
  lazy_device_sync(true);
  calc_timer.stop();
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, 0.);
  }
}

void adaptive_flat_sparse_spmv(const int nnz_block_0, const int nnz_block_1, int trans, const int alpha, const int beta,
                               const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
  VAR_FROM_CSR_DESC(d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  // for non one-pass.
  if (!FLAT_ONE_PASS) {
    constexpr int blocks = 512;
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;
    flat_multi_pass_sparse_spmv<R, RED_OPT, 2, blocks, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                          colindex, value, x, y, bmt);
    return;
  }

  if (!FLAT_ONE_PASS_ADAPTIVE) {
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 4;
    flat_one_pass_sparse_spmv<R, RED_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr, colindex,
                                                                       value, x, y, bmt);
    return;
  }

  // divide the matrix into 2 parts and get the max nnz per row of each part.
  int avg_block_nnz_max = std::max(2 * nnz_block_0 / m, 2 * nnz_block_1 / m);
  if (avg_block_nnz_max <= 32) {
    // use single thread to reduce if average nnz of row is less than 32.
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_DIRECT;
    constexpr int VEC_SIZE = 1;
    flat_one_pass_sparse_spmv<R, REDUCE_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                          colindex, value, x, y, bmt);
  } else if (avg_block_nnz_max <= 64) {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    // use more threads (4 threads) in vector to reduce if average nnz of row is larger than 32.
    constexpr int VEC_SIZE = 4;
    flat_one_pass_sparse_spmv<R, REDUCE_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                          colindex, value, x, y, bmt);
  } else {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    // use more threads (16 threads) in vector to reduce if average nnz of row is larger than 64.
    constexpr int VEC_SIZE = 16;
    flat_one_pass_sparse_spmv<R, REDUCE_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                          colindex, value, x, y, bmt);
  }
}

void flat_sparse_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                      const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
  // divide the matrix into 2 blocks and calculate nnz for each block.
  const int m = h_csr_desc.rows;
  const int bp_1 = h_csr_desc.row_ptr[m / 2];
  const int bp_2 = h_csr_desc.row_ptr[m];

  const int nnz_block_0 = bp_1 - 0;
  const int nnz_block_1 = bp_2 - bp_1;
  adaptive_flat_sparse_spmv(nnz_block_0, nnz_block_1, trans, alpha, beta, d_csr_desc, x, y, bmt);
}

#endif // SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP