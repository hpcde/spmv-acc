#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_UTILS_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_UTILS_H

#define ALIGN_256_BYTES(n) ((n + 255) / 256 * 256)

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

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_UTILS_H