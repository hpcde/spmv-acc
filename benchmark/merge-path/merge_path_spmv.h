#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H

#include "../utils/benchmark_time.h"
#include "api/types.h"
#include "merge_path_config.h"

template <int REDUCTION_ALGORITHM = Binary, int UPDATE_ALGORITHM = SingleBlock>
void merge_path_spmv(int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt);

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H