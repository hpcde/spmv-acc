#ifndef SPMV_ACC_BENCHMARK_CUB_SPMV
#define SPMV_ACC_BENCHMARK_CUB_SPMV
#include "api/types.h"

void spmv(int trans, const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc, const double *x,
          double *y);

#endif