#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_UPDATE_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_UPDATE_H

#include "merge_path_config.h"
#include "merge_path_utils.h"
#include <cub/block/block_scan.cuh>

template <typename T, int BLOCK_THREAD_NUM>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    single_block_update(KeyValuePair<int, T> *__restrict__ r, int rows, int count, T *__restrict__ y) {
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

using PartStateType = int4;

template <typename Key, typename Value> struct alignas(sizeof(PartStateType)) LookBackState {
  static constexpr int X = 0;
  static constexpr int I = 1;
  static constexpr int A = 2;
  static constexpr int P = 3;
  int flag;
  Key key;
  Value val;
};

template <typename Key, typename Value>
__device__ __forceinline__ void set_part_state(PartStateType *part_state, int flag, Key key, Value val) {
  LookBackState<Key, Value> tmp_state = {flag, key, val};
  PartStateType p;
  __builtin_memcpy(&p, &tmp_state, sizeof(PartStateType));
  *part_state = p;
}

template <typename Key, typename Value>
__device__ __forceinline__ KeyValuePair<Key, Value> get_pair(PartStateType *part_state) {
  LookBackState<Key, Value> tmp_state;
  __builtin_memcpy(&tmp_state, part_state, sizeof(PartStateType));
  return KeyValuePair<Key, Value>(tmp_state.key, tmp_state.val);
}

template <typename Key, typename Value> __device__ __forceinline__ int get_flag(PartStateType *part_state) {
  LookBackState<Key, Value> tmp_state;
  __builtin_memcpy(&tmp_state, part_state, sizeof(PartStateType));
  return tmp_state.flag;
}

template <typename T, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD = 1>
__global__ void __launch_bounds__(BLOCK_THREAD_NUM)
    look_back_update(KeyValuePair<int, T> *__restrict__ r, int rows, int count, T *__restrict__ y, int *dblock_id,
                     PartStateType *dpart_state_type) {
  static_assert(ITEMS_PER_THREAD == 1, "not support mutiple items per thread.");
  const int block_thread_id = threadIdx.x;
  using BlockScanT = cub::BlockScan<KeyValuePair<int, T>, BLOCK_THREAD_NUM>;
  __shared__ typename BlockScanT::TempStorage temp_storage_for_scan;
  using ReduceOpT = ReduceByKeyOp;
  using Key = int;
  using Value = T;
  static_assert(sizeof(LookBackState<Key, Value>) == sizeof(PartStateType), "LookBackState must alignas 128 bits.");
  ReduceOpT op;
  __shared__ int flat_block_id;
  __shared__ int lds_rows[BLOCK_THREAD_NUM];
  __shared__ KeyValuePair<int, T> lds_pair;
  __shared__ Key first_key;
  if (block_thread_id == 0) {
    flat_block_id = atomicAdd(dblock_id, 1);
  }
  __syncthreads();
  constexpr int ITEMS_PER_BLOCK = BLOCK_THREAD_NUM * ITEMS_PER_THREAD;
  const int block_item_begin = flat_block_id * ITEMS_PER_BLOCK;
  const int block_item_end = min(block_item_begin + ITEMS_PER_BLOCK, count);

  PartStateType *block_state = dpart_state_type + flat_block_id;
  KeyValuePair<Key, Value> pair(rows, static_cast<Value>(0));
  const int idx = block_item_begin + block_thread_id;
  if (idx < block_item_end) {
    pair = r[idx];
  }
  if (block_thread_id == 0) {
    first_key = pair.key;
  }
  if (block_thread_id == BLOCK_THREAD_NUM - 1) {
    set_part_state<Key, Value>(block_state, LookBackState<Key, Value>::I, pair.key, static_cast<Value>(0));
  }
  __syncthreads();
  BlockScanT(temp_storage_for_scan).InclusiveScan(pair, pair, op);
  __syncthreads();
  lds_rows[block_thread_id] = pair.key;
  if (flat_block_id == 0) {
    if (block_thread_id == BLOCK_THREAD_NUM - 1) {
      set_part_state<Key, Value>(block_state, LookBackState<Key, Value>::P, pair.key, pair.val);
    } else {
      if (lds_rows[block_thread_id] != lds_rows[block_thread_id + 1]) {
        y[pair.key] += pair.val;
      }
    }
  } else {
    if (block_thread_id == BLOCK_THREAD_NUM - 1) {
      if (pair.key == first_key) {
        set_part_state<Key, Value>(block_state, LookBackState<Key, Value>::A, pair.key, pair.val);
      } else {
        set_part_state<Key, Value>(block_state, LookBackState<Key, Value>::P, pair.key, pair.val);
      }
    }
    if (block_thread_id == 0) {
      const int track_flat_id = flat_block_id - 1;
      int f = get_flag<Key, Value>(&dpart_state_type[track_flat_id]);
      bool is_over = false;
      KeyValuePair<int, T> tmp_pair = get_pair<Key, Value>(&dpart_state_type[track_flat_id]);
      if ((f != LookBackState<Key, Value>::X) && tmp_pair.key != pair.key) {
        while (f == LookBackState<Key, Value>::X || f == LookBackState<Key, Value>::I) {
          __threadfence();
          f = get_flag<Key, Value>(&dpart_state_type[track_flat_id]);
        }
        tmp_pair = get_pair<Key, Value>(&dpart_state_type[track_flat_id]);
        y[tmp_pair.key] += tmp_pair.val;
        is_over = true;
      }
      while (!is_over && f != LookBackState<Key, Value>::P) {
        __threadfence();
        f = get_flag<Key, Value>(&dpart_state_type[track_flat_id]);
      }
      lds_pair = get_pair<Key, Value>(&dpart_state_type[track_flat_id]);
    }
    __syncthreads();
    if (block_thread_id == BLOCK_THREAD_NUM - 1) {
      if (pair.key == lds_pair.key) {
        pair.val += lds_pair.val;
        set_part_state<Key, Value>(block_state, LookBackState<Key, Value>::P, pair.key, pair.val);
      }
    } else {
      if (lds_rows[block_thread_id] != lds_rows[block_thread_id + 1]) {
        if (pair.key == lds_pair.key) {
          pair.val += lds_pair.val;
        }
        y[pair.key] += pair.val;
      }
    }
  }
}

template <typename T, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD = 1>
void update(KeyValuePair<int, T> *__restrict__ r, int rows, int count, T *__restrict__ y, void *temp_storage,
            SingleBlockType) {
  single_block_update<T, BLOCK_THREAD_NUM><<<1, BLOCK_THREAD_NUM>>>(r, rows, count, y);
}

template <typename T, int BLOCK_THREAD_NUM, int ITEMS_PER_THREAD = 1>
void update(KeyValuePair<int, T> *__restrict__ r, int rows, int count, T *__restrict__ y, void *temp_storage,
            LookBackType) {
  const int UpdateBlockNum = (count + BLOCK_THREAD_NUM) / BLOCK_THREAD_NUM;
  char *remain_storage = reinterpret_cast<char *>(temp_storage);
  cudaMemset(remain_storage, 0, ALIGN_256_BYTES(sizeof(int)) + ALIGN_256_BYTES(UpdateBlockNum * sizeof(PartStateType)));
  int *dblock_id = reinterpret_cast<int *>(remain_storage);
  remain_storage += ALIGN_256_BYTES(sizeof(int));
  PartStateType *dpart_state_type = reinterpret_cast<PartStateType *>(remain_storage);
  remain_storage += ALIGN_256_BYTES(UpdateBlockNum * sizeof(PartStateType));
  look_back_update<T, BLOCK_THREAD_NUM, ITEMS_PER_THREAD>
      <<<UpdateBlockNum, BLOCK_THREAD_NUM>>>(r, rows, count, y, dblock_id, dpart_state_type);
}

template <int BLOCK_THREAD_NUM> int update_temp_storage_bytes(int count, SingleBlockType) { return 0; }

template <int BLOCK_THREAD_NUM> int update_temp_storage_bytes(int count, LookBackType) {
  const int UpdateBlockNum = (count + BLOCK_THREAD_NUM) / BLOCK_THREAD_NUM;
  return ALIGN_256_BYTES(sizeof(int)) + ALIGN_256_BYTES(UpdateBlockNum * sizeof(PartStateType));
}

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_UPDATE_H