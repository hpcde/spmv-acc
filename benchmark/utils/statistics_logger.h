//
// Created by genshen on 2021/11/23.
//

#ifndef SPMV_ACC_STATISTICS_LOGGER_H
#define SPMV_ACC_STATISTICS_LOGGER_H

#include <string>

#include "utils/benchmark_time.h"
#include "verification.h"

namespace statistics {
    void print_statistics_header();

    template<typename T>
    void print_statistics(std::string mtx_name, std::string strategy_name, int rows, int cols, int nnz,
                          BenchmarkTime bmt, const VerifyResult<int, T> verify_result);
}

#endif //SPMV_ACC_STATISTICS_LOGGER_H
