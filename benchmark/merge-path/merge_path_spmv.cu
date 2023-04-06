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

  char *temp_storage = nullptr;
  int temp_storage_bytes = 0;
  temp_storage_bytes += ALIGN_256_BYTES((GlobalBlockNum + 1) * sizeof(int));
  temp_storage_bytes += ALIGN_256_BYTES(GlobalBlockNum * sizeof(KeyValuePair<int, T>));
#ifdef UPDATE_LOOK_BACK
  const int UpdateBlockNum = (GlobalBlockNum + 255) / 256;
  temp_storage_bytes += ALIGN_256_BYTES(sizeof(int));
  temp_storage_bytes += ALIGN_256_BYTES(UpdateBlockNum * sizeof(PartStateType));
#endif
  cudaMalloc((void **)&temp_storage, temp_storage_bytes);

  int *S = reinterpret_cast<int *>(temp_storage);
  temp_storage += ALIGN_256_BYTES((GlobalBlockNum + 1) * sizeof(int));
  KeyValuePair<int, T> *r = reinterpret_cast<KeyValuePair<int, T> *>(temp_storage);
  ;
  temp_storage += ALIGN_256_BYTES(GlobalBlockNum * sizeof(KeyValuePair<int, T>));

  // step 1: partition
  partition<int, 256, ITEMS_PER_BLOCK><<<512, 256>>>(d_csr_desc.row_ptr, h_csr_desc.rows, GlobalBlockNum + 1, S);
  cudaDeviceSynchronize();
  pre_timer.stop();
  // step 2: reduction
  calc_timer.start();
  reduction<T, I, BLOCK_THREAD_NUM, ITEMS_PER_THREAD><<<GlobalBlockNum, BLOCK_THREAD_NUM>>>(
      alpha, h_csr_desc.nnz, S, r, d_csr_desc.row_ptr, d_csr_desc.col_index, d_csr_desc.values, x, y);
  // step 3: update
#ifdef UPDATE_SINGLE_BLOCK
  { single_block_update<T, 256><<<1, 256>>>(r, h_csr_desc.rows, GlobalBlockNum, y); }
#endif
#ifdef UPDATE_LOOK_BACK
  {
    cudaMemset(temp_storage, 0, ALIGN_256_BYTES(sizeof(int)) + ALIGN_256_BYTES(UpdateBlockNum * sizeof(PartStateType)));
    int *dblock_id = reinterpret_cast<int *>(temp_storage);
    temp_storage += ALIGN_256_BYTES(sizeof(int));
    PartStateType *dpart_state_type = reinterpret_cast<PartStateType *>(temp_storage);
    temp_storage += ALIGN_256_BYTES(UpdateBlockNum * sizeof(PartStateType));
    look_back_update<T, 256>
        <<<UpdateBlockNum, 256>>>(r, h_csr_desc.rows, GlobalBlockNum, y, dblock_id, dpart_state_type);
  }
#endif
  cudaDeviceSynchronize();
  calc_timer.stop();
  destroy_timer.start();
  cudaFree(r);
  cudaFree(S);
  destroy_timer.stop();
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, destroy_timer.time_use);
  }
}