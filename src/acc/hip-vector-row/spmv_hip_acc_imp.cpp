//
// Created by chaohu on 2021/04/25.
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../common/utils.h"
#include "../common/global_mem_ops.hpp"

#define GLOBAL_LOAD_X2 // if defined, we load 2 double or 2 int in each loop.

constexpr int N_UNROLLING = 2;
constexpr int N_UNROLLING_SHIFT = 1;

/**
 * We solve SpMV with vector method.
 * In this method, wavefront can be divided into several groups (wavefront must be divided with no remainder).
 * (e.g. groups size can only be 1, 2,4,8,16,32,64 if \tparam WF_SIZE is 64).
 * Here, one group of threads are called a "vector".
 * Then, each vector can process one row of matrix A,
 * which also means one wavefront with multiple vectors can compute multiple rows.
 *
 * @tparam VECTOR_SIZE threads in one vector
 * @tparam WF_SIZE threads in one wavefront
 * @tparam WF_VECTORS vectors number in one wavefront
 * @tparam BLOCKS total blocks on one GPU (blocks in one grid).
 * @tparam T type of data in matrix A, vector x, vector y and alpha, beta.
 * @param m rows in matrix A
 * @param alpha alpha value
 * @param beta beta value
 * @param row_offset row offset array of csr matrix A
 * @param csr_col_ind col index of csr matrix A
 * @param csr_val matrix A in csr format
 * @param x vector x
 * @param y vector y
 * @return
 */
template <int VECTOR_SIZE, int WF_VECTORS, int WF_SIZE, int BLOCKS, typename T>
__global__ void spmv_vector_row_kernel(int m, const T alpha, const T beta, const int *row_offset,
                                       const int *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int vector_thread_id = global_thread_id % VECTOR_SIZE; // local thread id in current vector
  const int vector_id = global_thread_id / VECTOR_SIZE;        // global vector id
  const int vector_num = gridDim.x * blockDim.x / VECTOR_SIZE; // total vectors on device

  const int nwf_in_block = blockDim.x / WF_SIZE;          // wavefront number in a block
  const int global_wf_id = global_thread_id / WF_SIZE;    // global wavefront id
  const int thread_id_in_wf = global_thread_id % WF_SIZE; // thread id in current wavefront
  const int wf_id_in_block = threadIdx.x / WF_SIZE;       // wavefront id in current block

  constexpr unsigned int shared_len = 64 * 1024 / (BLOCKS / 64) / (sizeof(T) + sizeof(int));
  __shared__ T shared_csr[shared_len];
  __shared__ int shared_col_inx[shared_len];
  const int shared_len_wf = shared_len / nwf_in_block;            // data size in a wavefront.
  const int shared_wf_start_inx = wf_id_in_block * shared_len_wf; // start index of shared mem for current wavefront.
  T *_wf_shared_csr = shared_csr + shared_wf_start_inx;           // LDS memory for current wavefront.
  int *_wf_shared_col_inx = shared_col_inx + shared_wf_start_inx; // LDS memory for current wavefront.

  const int n_loops = m / vector_num + (m % vector_num == 0 ? 0 : 1);
  int row = vector_id;
  for (int k = 0; k < n_loops; k++) { // all threads in one wavefront will have the same `n_loops`.
    // load data into LDS.
    const int left_base_index = min((row / WF_VECTORS) * WF_VECTORS, m);
    const int right_base_index = min(left_base_index + WF_VECTORS, m);
    const int start_index = row_offset[left_base_index];
    const int end_index = row_offset[right_base_index];
    // todo: assert (end_index - start_index < shared_len/nwf_in_block)

#ifdef GLOBAL_LOAD_X2
    const int n_lds_load = end_index - start_index;
    if (n_lds_load <= WF_SIZE) { // load all data just in one round.
      if (thread_id_in_wf < n_lds_load) {
        _wf_shared_csr[thread_id_in_wf] = csr_val[start_index + thread_id_in_wf];
        _wf_shared_col_inx[thread_id_in_wf] = csr_col_ind[start_index + thread_id_in_wf];
      }
    } else {
      // unrolling
      const int unrolling_loop_end = start_index + ((n_lds_load >> N_UNROLLING_SHIFT) << N_UNROLLING_SHIFT);
      for (int j = start_index + N_UNROLLING * thread_id_in_wf; j < unrolling_loop_end; j += N_UNROLLING * WF_SIZE) {
        dbl_x2 dbl_v_x2;
        int_x2 int_v_x2;
        global_load(static_cast<const void *>(csr_val + j), dbl_v_x2);
        global_load_int(static_cast<const void *>(csr_col_ind + j), int_v_x2);
        _wf_shared_csr[j - start_index] = dbl_v_x2.a;
        _wf_shared_csr[j - start_index + 1] = dbl_v_x2.b;
        _wf_shared_col_inx[j - start_index] = int_v_x2.a;
        _wf_shared_col_inx[j - start_index + 1] = int_v_x2.b;
      }
      for (int j = unrolling_loop_end + thread_id_in_wf; j < end_index; j += WF_SIZE) {
        _wf_shared_csr[j - start_index] = csr_val[j];
        _wf_shared_col_inx[j - start_index] = csr_col_ind[j];
      }
    }
#endif
#ifndef GLOBAL_LOAD_X2
    for (int i = start_index + thread_id_in_wf; i < end_index; i += WF_SIZE) {
      _wf_shared_csr[i - start_index] = csr_val[i];
      _wf_shared_col_inx[i - start_index] = csr_col_ind[i];
    }
#endif

    // calculate
    if (row < m) {
      const int row_start = row_offset[row];
      const int row_end = row_offset[row + 1];
      T sum = static_cast<T>(0);

      for (int i = row_start + vector_thread_id; i < row_end; i += VECTOR_SIZE) {
        asm_v_fma_f64(_wf_shared_csr[i - start_index], device_ldg(x + _wf_shared_col_inx[i - start_index]), sum);
      }

      // reduce inside a vector
      // #pragma unroll
      for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
        sum += __shfl_down(sum, i, VECTOR_SIZE);
      }

      if (vector_thread_id == 0) {
        y[row] = device_fma(beta, y[row], alpha * sum);
      }
    }
    row += vector_num;
  }
}

#define VECTOR_KERNEL_WRAPPER(N)                                                                                       \
  (spmv_vector_row_kernel<N, (64 / N), 64, 512, double>)<<<512, 256>>>(m, alpha, beta, rowptr, colindex, value, x, y)

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y) {
  //  const int avg_eles_per_row = ceil(rowptr[m] + 0.0 / m);
  const int avg_eles_per_row = rowptr[m] / m;

  if (avg_eles_per_row <= 4) {
    VECTOR_KERNEL_WRAPPER(2);
  } else if (avg_eles_per_row <= 8) {
    VECTOR_KERNEL_WRAPPER(4);
  } else if (avg_eles_per_row <= 16) {
    VECTOR_KERNEL_WRAPPER(8);
  } else if (avg_eles_per_row <= 32) {
    VECTOR_KERNEL_WRAPPER(16);
  } else if (avg_eles_per_row <= 64) {
    VECTOR_KERNEL_WRAPPER(32);
  } else {
    VECTOR_KERNEL_WRAPPER(64);
  }
}
