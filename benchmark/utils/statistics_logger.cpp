//
// Created by genshen on 2021/11/23.
//

#include <iostream>
#include <string>

#include "statistics_logger.h"

namespace statistics {
    void print_statistics_header() {
        std::cout << "PERFORMANCE,"
                  << "matrix name,"
                  << "strategy name,"
                  << "rows,"
                  << "cols,"
                  << "nnz,"
                  << "nnz/row,"
                  << "GB/s(calc_time),"
                  << "GFLOPS(calc_time),"
                  << "GB/s(total_time),"
                  << "GFLOPS(total_time),"
                  << "mid pre cost,"
                  << "mid calc cost,"
                  << "mid destroy cost,"
                  << "mid total cost" << std::endl;
    }

    /**
     * @tparam T type of data
     * @param m rows
     * @param n cols
     * @param nnz number of non-zeros
     * @param time time in us
     */
    template<typename T>
    void print_statistics(std::string mtx_name, std::string strategy_name, int rows, int cols, int nnz,
                          BenchmarkTime bmt) {
        double mem_bytes = static_cast<double>(sizeof(T) * (2 * rows + nnz) + sizeof(int) * (rows + 1 + nnz));

        double calc_time_bandwidth = (mem_bytes + 0.0) / (1024 * 1024 * 1024) / (bmt.calc_time_use / 1e3 / 1e3);
        double calc_time_gflops = static_cast<double>(2 * nnz) / bmt.calc_time_use / 1e3;

        double total_time_bandwidth = (mem_bytes + 0.0) / (1024 * 1024 * 1024) / (bmt.total_time_use / 1e3 / 1e3);
        double total_time_gflops = static_cast<double>(2 * nnz) / bmt.total_time_use / 1e3;

        std::cout << "PERFORMANCE," << mtx_name << "," << strategy_name << "," << rows << "," << cols << "," << nnz
                  << ","
                  << (nnz + 0.0) / rows << "," << calc_time_bandwidth << "," << calc_time_gflops << ","
                  << total_time_bandwidth << "," << total_time_gflops << "," << bmt.pre_time_use << ","
                  << bmt.calc_time_use
                  << "," << bmt.destroy_time_use << "," << bmt.total_time_use << std::endl;
    }

    template void print_statistics<float>(std::string mtx_name, std::string strategy_name, int rows, int cols, int nnz,
                                          BenchmarkTime bmt);

    template void print_statistics<double>(std::string mtx_name, std::string strategy_name, int rows, int cols, int nnz,
                                           BenchmarkTime bmt);
} // namespace statistics
