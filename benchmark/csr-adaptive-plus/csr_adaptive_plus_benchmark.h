//
// Created by genshen on 2025/3/11.
//

#ifndef SPMV_ACC_BENCHMARK_SPMV_ACC_CSR_ADAPTIVE_PLUS_H
#define SPMV_ACC_BENCHMARK_SPMV_ACC_CSR_ADAPTIVE_PLUS_H

#include <api/types.h>

#include "../utils/benchmark_time.h"
#include "benchmark_config.h"

template <typename I, typename T>
void csr_adaptive_plus_sparse_spmv_with_profile(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                                const csr_desc<I, T> d_csr_desc, const T *x, T *y, my_timer &pre_timer,
                                                my_timer &calc_timer, my_timer &destroy_timer);

#endif // SPMV_ACC_BENCHMARK_SPMV_ACC_CSR_ADAPTIVE_PLUS_H
