#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_HPP
#define SPMV_ACC_BENCHMARK_MERGE_PATH_HPP

#include <iostream>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "csr_spmv.hpp"
#include "merge-path/merge_path_spmv.h"
#include "merge-path/merge_path_config.h"
#include "utils.hpp"
#include "utils/benchmark_time.h"

struct MergePathSpMV : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    merge_path_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return false; }
};

struct MergePathSingleBlockUpdateSpMV : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    merge_path_spmv<Binary, SingleBlock>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return false; }
};

struct MergePathLookBackUpdateSpMV : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    merge_path_spmv<Binary, LookBack>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return false; }
};

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_HPP
