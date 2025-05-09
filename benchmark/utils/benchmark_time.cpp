#include "benchmark_time.h"

void BenchmarkTime::set_time(double _pre_time_use, double _calc_time_use, double _calc2_time_use,
                             double _destroy_time_use) {
  pre_time_use = _pre_time_use;
  calc_time_use = _calc_time_use;
  calc2_time_use = _calc2_time_use;
  destroy_time_use = _destroy_time_use;
  update_total_time();
}

void BenchmarkTime::update_total_time() {
  total_time_use = pre_time_use + calc_time_use + calc2_time_use + destroy_time_use;
}

void BenchmarkTimeArray::append(BenchmarkTime bmt) {
  if (index >= BENCHMARK_ARRAY_SIZE) {
    return;
  }
  bmt_array[index++] = bmt;
}

BenchmarkTime BenchmarkTimeArray::get_mid_time() {
  BenchmarkTime mid_bmt;
  int mid_index = index / 2;
  if (index == 0) {
    return mid_bmt;
  }
  std::sort(bmt_array, bmt_array + index, [](const BenchmarkTime &lbt, const BenchmarkTime &rbt) -> bool {
    return lbt.total_time_use > rbt.total_time_use;
  });
  if (index % 2 == 0) {
    mid_bmt.pre_time_use = (bmt_array[mid_index - 1].pre_time_use + bmt_array[mid_index].pre_time_use) / 2.0;
    mid_bmt.calc_time_use = (bmt_array[mid_index - 1].calc_time_use + bmt_array[mid_index].calc_time_use) / 2.0;
    mid_bmt.calc2_time_use = (bmt_array[mid_index - 1].calc2_time_use + bmt_array[mid_index].calc2_time_use) / 2.0;
    mid_bmt.destroy_time_use =
        (bmt_array[mid_index - 1].destroy_time_use + bmt_array[mid_index].destroy_time_use) / 2.0;
    mid_bmt.update_total_time();
  } else {
    mid_bmt = bmt_array[mid_index];
  }
  return mid_bmt;
}