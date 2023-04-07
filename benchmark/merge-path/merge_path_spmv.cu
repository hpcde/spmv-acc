#include "../utils/benchmark_time.h"
#include "api/types.h"
#include "merge_path_config.h"
#include "merge_path_partition.h"
#include "merge_path_reduction.h"
#include "merge_path_spmv.h"
#include "merge_path_update.h"
#include "merge_path_utils.h"
#include <common/platforms/cuda/cuda_utils.hpp>
#include <cuda_runtime.h>
#include <utility>

template <int REDUCTION_ALGORITHM, int UPDATE_ALGORITHM>
void merge_path_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
  my_timer pre_timer, calc_timer, destroy_timer;
  pre_timer.start();

  constexpr int BLOCK_THREAD_NUM = 256;
  constexpr int ITEMS_PER_THREAD = 1;
  constexpr int ITEMS_PER_BLOCK = BLOCK_THREAD_NUM * ITEMS_PER_THREAD;
  const int GlobalBlockNum = (h_csr_desc.nnz + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
  using I = int;
  using T = double;

  void *temp_storage = nullptr;
  char *remain_storage = nullptr;
  int temp_storage_bytes = 0;
  temp_storage_bytes += ALIGN_256_BYTES((GlobalBlockNum + 1) * sizeof(int));
  temp_storage_bytes += ALIGN_256_BYTES(GlobalBlockNum * sizeof(KeyValuePair<int, T>));
  temp_storage_bytes += update_temp_storage_bytes<256>(GlobalBlockNum, typename UpdateTrait<UPDATE_ALGORITHM>::type{});
  cudaMalloc(&temp_storage, temp_storage_bytes);
  remain_storage = reinterpret_cast<char *>(temp_storage);
  int *S = reinterpret_cast<int *>(remain_storage);
  remain_storage += ALIGN_256_BYTES((GlobalBlockNum + 1) * sizeof(int));
  KeyValuePair<int, T> *r = reinterpret_cast<KeyValuePair<int, T> *>(remain_storage);
  remain_storage += ALIGN_256_BYTES(GlobalBlockNum * sizeof(KeyValuePair<int, T>));

  // step 1: partition
  partition<int, 256, ITEMS_PER_BLOCK><<<512, 256>>>(d_csr_desc.row_ptr, h_csr_desc.rows, GlobalBlockNum + 1, S);
  cudaDeviceSynchronize();
  pre_timer.stop();
  // step 2: reduction
  calc_timer.start();
  reduction<T, I, BLOCK_THREAD_NUM, ITEMS_PER_THREAD><<<GlobalBlockNum, BLOCK_THREAD_NUM>>>(
      alpha, h_csr_desc.nnz, S, r, d_csr_desc.row_ptr, d_csr_desc.col_index, d_csr_desc.values, x, y,
      typename ReductionTrait<REDUCTION_ALGORITHM>::type{});
  // step 3: update
  update<T, 256>(r, h_csr_desc.rows, GlobalBlockNum, y, reinterpret_cast<void *>(remain_storage),
                 typename UpdateTrait<UPDATE_ALGORITHM>::type{});
  remain_storage += update_temp_storage_bytes<256>(GlobalBlockNum, typename UpdateTrait<UPDATE_ALGORITHM>::type{});
  cudaDeviceSynchronize();
  calc_timer.stop();
  destroy_timer.start();
  cudaFree(temp_storage);
  destroy_timer.stop();
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, destroy_timer.time_use);
  }
}

// template instantiation
template void merge_path_spmv<Linear, SingleBlock>(int trans, const int alpha, const int beta,
                                                   const csr_desc<int, double> h_csr_desc,
                                                   const csr_desc<int, double> d_csr_desc, const double *x, double *y,
                                                   BenchmarkTime *bmt);

template void merge_path_spmv<Linear, LookBack>(int trans, const int alpha, const int beta,
                                                const csr_desc<int, double> h_csr_desc,
                                                const csr_desc<int, double> d_csr_desc, const double *x, double *y,
                                                BenchmarkTime *bmt);

template void merge_path_spmv<Binary, SingleBlock>(int trans, const int alpha, const int beta,
                                                   const csr_desc<int, double> h_csr_desc,
                                                   const csr_desc<int, double> d_csr_desc, const double *x, double *y,
                                                   BenchmarkTime *bmt);

template void merge_path_spmv<Binary, LookBack>(int trans, const int alpha, const int beta,
                                                const csr_desc<int, double> h_csr_desc,
                                                const csr_desc<int, double> d_csr_desc, const double *x, double *y,
                                                BenchmarkTime *bmt);