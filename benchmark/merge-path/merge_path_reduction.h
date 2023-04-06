#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H

#include "merge_path_config.h"
#include "merge_path_utils.h"
#include <cub/block/block_scan.cuh>

template <typename T, typename I, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD = 1>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    reduction(const int alpha, int nnz, int *__restrict__ S, KeyValuePair<int, T> *__restrict__ r,
              const I *__restrict__ rowptr, const I *__restrict__ colindex, const T *__restrict__ value,
              const T *__restrict__ x, T *__restrict__ y) {
  const int global_block_id = blockIdx.x;
  const int block_thread_id = threadIdx.x;
  const int block_row_begin = S[global_block_id];
  const int block_row_end = S[global_block_id + 1];

  using BlockScanT = cub::BlockScan<KeyValuePair<int, T>, BLOCK_THREAD_NUM>;
  __shared__ typename BlockScanT::TempStorage temp_storage_for_scan;
  using ReduceOpT = ReduceByKeyOp;
  KeyValuePair<int, T> pair(block_row_end, static_cast<T>(0));
  ReduceOpT op;

  const int idx = global_block_id * BLOCK_THREAD_NUM * ITEMS_PER_THREAD + block_thread_id;
  bool is_last;
#ifdef REDUCE_LINEAR_SEARCH
  // linear search
  int next_row_idx;
  for (int row = block_row_begin + 1; row <= block_row_end; ++row) {
    next_row_idx = rowptr[row];
    if (idx < next_row_idx) {
      pair.key = row - 1;
      break;
    }
  }
  is_last = (idx + 1 == next_row_idx);
#endif
#ifdef REDUCE_BINARY_SEARCH
  // binary search
  pair.key = largest_less_equal_binary_search(block_row_begin, block_row_end + 1, rowptr, idx);
  is_last = (idx + 1 == rowptr[pair.key + 1]);
#endif
  if (idx < nnz) {
    pair.val = alpha * value[idx] * x[colindex[idx]];
  }
  __syncthreads();
  BlockScanT(temp_storage_for_scan).InclusiveScan(pair, pair, op);
  __syncthreads();
  if (is_last) {
    y[pair.key] = pair.val;
  }
  if (block_thread_id == BLOCK_THREAD_NUM - 1) {
    if (is_last) {
      pair.val = 0;
    }
    r[global_block_id] = pair;
  }
}

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_REDUCTION_H