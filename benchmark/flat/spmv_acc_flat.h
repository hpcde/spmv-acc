//
// Created by genshen on 2021/11/26.
// this file is copied and modified from src/acc/flat/flat.cpp for adding timer code.
//

#ifndef SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_H
#define SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_H

#include <api/types.h>

#include "../utils/benchmark_time.h"


void adaptive_flat_sparse_spmv(const int nnz_block_0, const int nnz_block_1, int trans, const int alpha, const int beta,
                               const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt);

void flat_sparse_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                      const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt);

void segment_sum_flat_sparse_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                                  const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt);

#endif // SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP