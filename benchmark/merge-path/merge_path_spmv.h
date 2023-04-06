#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H

#include "../utils/benchmark_time.h"
#include "api/types.h"

void merge_path_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt);

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H