//
// Created by reget on 2021/11/21.
//

#ifndef SPMV_ACC_BENCHMARK_BENCHMARK_TIME_HPP
#define SPMV_ACC_BENCHMARK_BENCHMARK_TIME_HPP

#include <algorithm>

#include "timer.h"

constexpr int BENCHMARK_ARRAY_SIZE = 3;

struct BenchmarkTime {
  // us
  double pre_time_use;
  double calc_time_use;
  double destroy_time_use;
  double total_time_use;
  BenchmarkTime() : pre_time_use(0), calc_time_use(0), destroy_time_use(0), total_time_use(0) {}
  BenchmarkTime(double _pre_time_use, double _calc_time_use, double _destroy_time_use)
      : pre_time_use(_pre_time_use), calc_time_use(_calc_time_use), destroy_time_use(_destroy_time_use) {
    update_total_time();
  }
  BenchmarkTime &operator=(const BenchmarkTime &_benchmark_time) {
    pre_time_use = _benchmark_time.pre_time_use;
    calc_time_use = _benchmark_time.calc_time_use;
    destroy_time_use = _benchmark_time.destroy_time_use;
    total_time_use = _benchmark_time.total_time_use;
    return *this;
  }
  BenchmarkTime(const BenchmarkTime &_benchmark_time) { *this = _benchmark_time; }
  void update_total_time();
  void set_time(double _pre_time_use, double _calc_time_use, double _destroy_time_use);
};

struct BenchmarkTimeArray {
  BenchmarkTime bmt_array[BENCHMARK_ARRAY_SIZE];
  size_t index;
  BenchmarkTimeArray() : index(0) {}
  void append(BenchmarkTime bmt);
  BenchmarkTime get_mid_time();
};

#endif // SPMV_ACC_BENCHMARK_BENCHMARK_TIME_HPP
