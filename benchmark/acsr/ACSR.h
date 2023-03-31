#ifndef SPMV_ACC_BENCHMARK_ACSR_SPMV
#define SPMV_ACC_BENCHMARK_ACSR_SPMV

#include "../utils/benchmark_time.h"
#include "api/types.h"


template<class T>
void acsr_driver(T *values, int *col_idx, int *row_off, T *x, T *y, int m, int n, int nnz,BenchmarkTime *bmt);
void acsr(int trans, const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc, const double *x,
          double *y, BenchmarkTime *bmt);



#endif