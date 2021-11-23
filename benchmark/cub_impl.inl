#define CUB_STDERR

#include <iostream>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "cub/spmv.h"
#include "utils.hpp"
#include "utils/benchmark_time.h"

struct CubDeviceSpMV : CsrSpMV<CubDeviceSpMV> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    spmv(trans, h_csr_desc, d_csr_desc, x, y, bmt);
    return false;
  }
};