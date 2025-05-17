#ifndef SPMV_ACC_BENCHMARK_ACSR_SPMV
#define SPMV_ACC_BENCHMARK_ACSR_SPMV

#include "../utils/benchmark_time.h"
#include "api/types.h"

void acsr(int trans, const double alpha, const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc, const double *x,
          double *y, BenchmarkTime *bmt);

#endif
