#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_PARTITION_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_PARTITION_H

#include "merge_path_config.h"
#include "merge_path_utils.h"

template <typename I, int BLOCK_THREAD_NUM, int ITEMS_PER_BLOCK>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    partition(I const *__restrict__ row_ptr, int m, int count, int *__restrict__ S) {
  const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_thread_num = blockDim.x * gridDim.x;
  for (int idx = global_thread_id; idx < count; idx += global_thread_num) {
    // binary search
    const I target = idx * ITEMS_PER_BLOCK;
    S[idx] = largest_less_equal_binary_search(0, m + 1, row_ptr, target);
  }
}

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_PARTITION_H