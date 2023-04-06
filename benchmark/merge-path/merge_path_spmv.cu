#include <iostream>

#include "../utils/benchmark_time.h"
#include "api/types.h"
#include "merge_path_spmv.h"
#include <common/platforms/cuda/cuda_utils.hpp>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <utility>

#ifndef REDUCE_BINARY_SEARCH
#define REDUCE_BINARY_SEARCH
#endif

template <typename Key, typename Value> struct KeyValuePair {
  Key key;
  Value val;

  __host__ __device__ __forceinline__ KeyValuePair() {}

  __host__ __device__ __forceinline__ KeyValuePair(Key const &key, Value const &val) : key(key), val(val) {}
};

struct ReduceByKeyOp {
  __host__ __device__ __forceinline__ ReduceByKeyOp() {}

  template <typename KeyValuePairT>
  __host__ __device__ __forceinline__ KeyValuePairT operator()(const KeyValuePairT &first,
                                                               const KeyValuePairT &second) {
    KeyValuePairT retval = second;
    if (first.key == second.key) {
      retval.val = first.val + retval.val;
    }
    return retval;
  }
};

template <typename I>
__device__ __forceinline__ int largest_less_equal_binary_search(int left, int right, const I *__restrict__ row_ptr,
                                                                I target) {
  while (left < right - 1) {
    const int mid = (left + right) >> 1;
    if (row_ptr[mid] <= target) {
      left = mid;
    } else {
      right = mid;
    }
  }
  return left;
}

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

template <typename T, int BLOCK_THREAD_NUM>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    update(KeyValuePair<int, T> *__restrict__ r, int rows, int count, T *__restrict__ y) {
  using BlockScanT = cub::BlockScan<KeyValuePair<int, T>, BLOCK_THREAD_NUM>;
  __shared__ typename BlockScanT::TempStorage temp_storage_for_scan;
  using ReduceOpT = ReduceByKeyOp;
  const int tid = threadIdx.x;
  ReduceOpT op;
  const int rounds = count / BLOCK_THREAD_NUM;
  __shared__ int lds_rows[BLOCK_THREAD_NUM];
  __shared__ KeyValuePair<int, T> lds_carry_out;
  if (tid == 0) {
    lds_carry_out.key = 0;
    lds_carry_out.val = static_cast<T>(0);
  }
  for (int i = 0; i < rounds; ++i) {
    const int idx = i * BLOCK_THREAD_NUM + tid;
    auto pair = r[idx];
    lds_rows[tid] = pair.key;
    if (tid == 0) {
      if (pair.key == lds_carry_out.key) {
        pair.val += lds_carry_out.val;
      } else {
        y[lds_carry_out.key] += lds_carry_out.val;
      }
    }
    __syncthreads();
    BlockScanT(temp_storage_for_scan).InclusiveScan(pair, pair, op);
    __syncthreads();
    if (tid == BLOCK_THREAD_NUM - 1) {
      lds_carry_out = pair;
    } else if (lds_rows[tid] != lds_rows[tid + 1]) {
      y[pair.key] += pair.val;
    }
  }
  if (rounds * BLOCK_THREAD_NUM < count) {
    const int idx = rounds * BLOCK_THREAD_NUM + tid;
    KeyValuePair<int, T> pair(rows, static_cast<T>(0));
    if (idx < count) {
      pair = r[idx];
    }
    lds_rows[tid] = pair.key;
    if (tid == 0) {
      if (pair.key == lds_carry_out.key) {
        pair.val += lds_carry_out.val;
      } else {
        y[lds_carry_out.key] += lds_carry_out.val;
      }
    }
    __syncthreads();
    BlockScanT(temp_storage_for_scan).InclusiveScan(pair, pair, op);
    __syncthreads();
    if (tid != BLOCK_THREAD_NUM - 1 && lds_rows[tid] != lds_rows[tid + 1]) {
      y[pair.key] += pair.val;
    }
  }
}

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
  int *S;
  KeyValuePair<int, T> *r;
  cudaMalloc((void **)&S, (GlobalBlockNum + 1) * sizeof(int));
  cudaMalloc((void **)&r, GlobalBlockNum * sizeof(KeyValuePair<int, T>));
  // step 1: partition
  partition<int, 256, ITEMS_PER_BLOCK><<<512, 256>>>(d_csr_desc.row_ptr, h_csr_desc.rows, GlobalBlockNum + 1, S);
  cudaDeviceSynchronize();
  pre_timer.stop();
  // step 2: reduction
  calc_timer.start();
  reduction<T, I, BLOCK_THREAD_NUM, ITEMS_PER_THREAD><<<GlobalBlockNum, BLOCK_THREAD_NUM>>>(
      alpha, h_csr_desc.nnz, S, r, d_csr_desc.row_ptr, d_csr_desc.col_index, d_csr_desc.values, x, y);
  // step 3: update
  update<T, 256><<<1, 256>>>(r, h_csr_desc.rows, GlobalBlockNum, y);
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