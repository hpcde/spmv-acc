//
// Created by reget on 2021/11/19.
//

#ifndef SPMV_ACC_BENCHMARK_HOLA_HPP
#define SPMV_ACC_BENCHMARK_HOLA_HPP

#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "hola/spmv.h"
#include "utils.hpp"
#include "utils/benchmark_time.h"

struct HolaSpMV : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    try {
      spmv(trans, h_csr_desc.rows, h_csr_desc.cols, h_csr_desc.nnz, d_csr_desc, x, y, bmt);
    } catch (const std::runtime_error &error) {
      throw error;
    }
  }
  bool verify_beta_y() { return false; }
};

#endif // SPMV_ACC_BENCHMARK_HOLA_HPP
