#ifndef SPMV_ACC_BENCHMARK_HOLA_HIP_H
#define SPMV_ACC_BENCHMARK_HOLA_HIP_H

#include "../utils/benchmark_time.h"
#include "api/types.h"

void spmv(int trans, int rows, int cols, int nnz, const csr_desc<int, double> d_csr_desc, const double *x, double *y,
          BenchmarkTime *bmt);

#endif // SPMV_ACC_BENCHMARK_HOLA_HIP_H
