#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H

#include "merge_path_config.h"
#include "merge_path_utils.h"
#include <cub/block/block_scan.cuh>

template <typename T, typename I, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD = 1>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    reduction(const int alpha, int nnz, int *__restrict__ S, KeyValuePair<int, T> *__restrict__ r,
              const I *__restrict__ rowptr, const I *__restrict__ colindex, const T *__restrict__ value,
              const T *__restrict__ x, T *__restrict__ y, LinearSearchType) {
  static_assert(ITEMS_PER_THREAD <= 32, "ITEMS_PER_THREAD must less equal than 32.");
  const int global_block_id = blockIdx.x;
  const int block_thread_id = threadIdx.x;
  const int block_row_begin = S[global_block_id];
  const int block_row_end = S[global_block_id + 1];
  const int block_idx_begin = global_block_id * BLOCK_THREAD_NUM * ITEMS_PER_THREAD;

  using BlockScanT = cub::BlockScan<KeyValuePair<int, T>, BLOCK_THREAD_NUM>;
  __shared__ typename BlockScanT::TempStorage temp_storage_for_scan;
  using ReduceOpT = ReduceByKeyOp;
  ReduceOpT op;
  KeyValuePair<int, T> pair[ITEMS_PER_THREAD];
  unsigned int is_last = 0;
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    pair[i] = {block_row_end, static_cast<T>(0)};
    const int idx = block_idx_begin + ITEMS_PER_THREAD * block_thread_id + i;
    // linear search
    int next_row_idx;
    for (int row = block_row_begin + 1; row <= block_row_end; ++row) {
      next_row_idx = rowptr[row];
      if (idx < next_row_idx) {
        pair[i].key = row - 1;
        break;
      }
    }
    is_last |= (idx + 1 == next_row_idx) << i;
    if (idx < nnz) {
      pair[i].val = alpha * value[idx] * x[colindex[idx]];
    }
  }
  __syncthreads();
  BlockScanT(temp_storage_for_scan).InclusiveScan<ITEMS_PER_THREAD>(pair, pair, op);
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
    reduction(const int alpha, int nnz, int *__restrict__ S, KeyValuePair<int, T> *__restrict__ r,
              const I *__restrict__ rowptr, const I *__restrict__ colindex, const T *__restrict__ value,
              const T *__restrict__ x, T *__restrict__ y, BinarySearchType) {
  static_assert(ITEMS_PER_THREAD <= 32, "ITEMS_PER_THREAD must less equal than 32.");
  const int global_block_id = blockIdx.x;
  const int block_thread_id = threadIdx.x;
  const int block_row_begin = S[global_block_id];
  const int block_row_end = S[global_block_id + 1];
  const int block_idx_begin = global_block_id * BLOCK_THREAD_NUM * ITEMS_PER_THREAD;

  using BlockScanT = cub::BlockScan<KeyValuePair<int, T>, BLOCK_THREAD_NUM>;
  __shared__ typename BlockScanT::TempStorage temp_storage_for_scan;
  using ReduceOpT = ReduceByKeyOp;
  ReduceOpT op;
  KeyValuePair<int, T> pair[ITEMS_PER_THREAD];
  unsigned int is_last = 0;
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    pair[i] = {block_row_end, static_cast<T>(0)};
    const int idx = block_idx_begin + ITEMS_PER_THREAD * block_thread_id + i;
    // binary search
    pair[i].key = largest_less_equal_binary_search(block_row_begin, block_row_end + 1, rowptr, idx);
    is_last |= (idx + 1 == rowptr[pair[i].key + 1]) << i;
    if (idx < nnz) {
      pair[i].val = alpha * value[idx] * x[colindex[idx]];
    }
  }
  __syncthreads();
  BlockScanT(temp_storage_for_scan).InclusiveScan<ITEMS_PER_THREAD>(pair, pair, op);
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