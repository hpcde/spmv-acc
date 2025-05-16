#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H

#include "merge_path_config.h"
#include "merge_path_utils.h"
#include <cub/block/block_scan.cuh>

template <typename T, typename BlockScanT, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD> struct ReductionTempStorage {
  union Reuseable {
    typename BlockScanT::TempStorage for_scan;
    KeyValuePair<int, T> for_pair[BLOCK_THREAD_NUM * ITEMS_PER_THREAD];
  } reuse;
};

template <typename T, typename I, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD = 1>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    reduction(const T alpha, int nnz, int *__restrict__ S, KeyValuePair<int, T> *__restrict__ r,
              const I *__restrict__ rowptr, const I *__restrict__ colindex, const T *__restrict__ value,
              const T *__restrict__ x, T *__restrict__ y, LinearSearchType) {
  static_assert(ITEMS_PER_THREAD <= 32, "ITEMS_PER_THREAD must less equal than 32.");
  const int global_block_id = blockIdx.x;
  const int block_thread_id = threadIdx.x;
  const int block_row_begin = S[global_block_id];
  const int block_row_end = S[global_block_id + 1];
  const int block_idx_begin = global_block_id * BLOCK_THREAD_NUM * ITEMS_PER_THREAD;

  using BlockScanT = cub::BlockScan<KeyValuePair<int, T>, BLOCK_THREAD_NUM>;

  __shared__ char temp_storage[sizeof(ReductionTempStorage<T, BlockScanT, BLOCK_THREAD_NUM, ITEMS_PER_THREAD>)];
  KeyValuePair<int, T> *temp_storage_for_pair = reinterpret_cast<KeyValuePair<int, T> *>(temp_storage);
  typename BlockScanT::TempStorage *temp_storage_for_scan =
      reinterpret_cast<typename BlockScanT::TempStorage *>(temp_storage);
  using ReduceOpT = ReduceByKeyOp;
  ReduceOpT op;
  KeyValuePair<int, T> pair[ITEMS_PER_THREAD];
  unsigned int is_last = 0;
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    KeyValuePair<int, T> tmp_pair = {block_row_end, static_cast<T>(0)};
    const int idx = block_idx_begin + BLOCK_THREAD_NUM * i + block_thread_id;
    // linear search
    int next_row_idx;
    for (int row = block_row_begin + 1; row <= block_row_end; ++row) {
      next_row_idx = rowptr[row];
      if (idx < next_row_idx) {
        tmp_pair.key = row - 1;
        break;
      }
    }
    if (idx < nnz) {
      tmp_pair.val = alpha * value[idx] * x[colindex[idx]];
    }
    temp_storage_for_pair[idx - block_idx_begin] = tmp_pair;
  }
  __syncthreads();
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int offset = ITEMS_PER_THREAD * block_thread_id + i;
    pair[i] = temp_storage_for_pair[offset];
    is_last |= (block_idx_begin + offset + 1 == rowptr[pair[i].key + 1]) << i;
  }
  __syncthreads();
  BlockScanT(*temp_storage_for_scan).InclusiveScan<ITEMS_PER_THREAD>(pair, pair, op);
  __syncthreads();
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (is_last & 0b1) {
      y[pair[i].key] = pair[i].val;
    }
    if (block_thread_id == BLOCK_THREAD_NUM - 1 && i == ITEMS_PER_THREAD - 1) {
      if (is_last & 0b1) {
        pair[i].val = 0;
      }
      r[global_block_id] = pair[i];
    }
    is_last >>= 1;
  }
}

template <typename T, typename I, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD = 1>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    reduction(const T alpha, int nnz, int *__restrict__ S, KeyValuePair<int, T> *__restrict__ r,
              const I *__restrict__ rowptr, const I *__restrict__ colindex, const T *__restrict__ value,
              const T *__restrict__ x, T *__restrict__ y, BinarySearchType) {
  static_assert(ITEMS_PER_THREAD <= 32, "ITEMS_PER_THREAD must less equal than 32.");
  const int global_block_id = blockIdx.x;
  const int block_thread_id = threadIdx.x;
  const int block_row_begin = S[global_block_id];
  const int block_row_end = S[global_block_id + 1];
  const int block_idx_begin = global_block_id * BLOCK_THREAD_NUM * ITEMS_PER_THREAD;

  using BlockScanT = cub::BlockScan<KeyValuePair<int, T>, BLOCK_THREAD_NUM>;

  __shared__ char temp_storage[sizeof(ReductionTempStorage<T, BlockScanT, BLOCK_THREAD_NUM, ITEMS_PER_THREAD>)];
  KeyValuePair<int, T> *temp_storage_for_pair = reinterpret_cast<KeyValuePair<int, T> *>(temp_storage);
  typename BlockScanT::TempStorage *temp_storage_for_scan =
      reinterpret_cast<typename BlockScanT::TempStorage *>(temp_storage);
  using ReduceOpT = ReduceByKeyOp;
  ReduceOpT op;
  KeyValuePair<int, T> pair[ITEMS_PER_THREAD];
  unsigned int is_last = 0;
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    KeyValuePair<int, T> tmp_pair = {block_row_end, static_cast<T>(0)};
    const int idx = block_idx_begin + BLOCK_THREAD_NUM * i + block_thread_id;
    // binary search
    tmp_pair.key = largest_less_equal_binary_search(block_row_begin, block_row_end + 1, rowptr, idx);
    if (idx < nnz) {
      tmp_pair.val = alpha * value[idx] * x[colindex[idx]];
    }
    temp_storage_for_pair[idx - block_idx_begin] = tmp_pair;
  }
  __syncthreads();
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int offset = ITEMS_PER_THREAD * block_thread_id + i;
    pair[i] = temp_storage_for_pair[offset];
    is_last |= (block_idx_begin + offset + 1 == rowptr[pair[i].key + 1]) << i;
  }
  __syncthreads();
  BlockScanT(*temp_storage_for_scan).InclusiveScan<ITEMS_PER_THREAD>(pair, pair, op);
  __syncthreads();
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (is_last & 0b1) {
      y[pair[i].key] = pair[i].val;
    }
    if (block_thread_id == BLOCK_THREAD_NUM - 1 && i == ITEMS_PER_THREAD - 1) {
      if (is_last & 0b1) {
        pair[i].val = 0;
      }
      r[global_block_id] = pair[i];
    }
    is_last >>= 1;
  }
}

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H