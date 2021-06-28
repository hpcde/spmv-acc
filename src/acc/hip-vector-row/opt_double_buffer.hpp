//
// Created by genshen on 2021/6/28.
//

#ifndef SPMV_ACC_VECTOR_ROW_OPT_DOUBLE_BUFFER_HPP
#define SPMV_ACC_VECTOR_ROW_OPT_DOUBLE_BUFFER_HPP

#include "../common/global_mem_ops.hpp"
#include "../common/utils.h"

#include "vector_config.h"

/**
 * one element data in csr matrix.
 * @tparam I type of column index type
 * @tparam T type of value in matrix
 */
template <typename I, typename T> struct _type_matrix_data {
  T value;   // value of matrix
  I col_ind; // column index of matrix
};

typedef int_x2 _type_row_offsets;

/**
 * double buffer support of vector strategy..
 * @tparam VECTOR_SIZE
 * @tparam WF_VECTORS
 * @tparam WF_SIZE
 * @tparam BLOCKS
 * @tparam I
 * @tparam T
 * @param m
 * @param alpha
 * @param beta
 * @param row_offset
 * @param csr_col_ind
 * @param csr_val
 * @param x
 * @param y
 * @return
 */
template <int VECTOR_SIZE, int WF_VECTORS, int WF_SIZE, int BLOCKS, typename I, typename T>
__global__ void spmv_vector_row_kernel_double_buffer(int m, const T alpha, const T beta, const I *row_offset,
                                                     const I *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int vector_thread_id = global_thread_id % VECTOR_SIZE; // local thread id in current vector
  const int vector_id = global_thread_id / VECTOR_SIZE;        // global vector id
  const int vector_num = gridDim.x * blockDim.x / VECTOR_SIZE; // total vectors on device

  // todo: assert(vector_id < m);
  _type_row_offsets next_row_offsets{0, 0};
  next_row_offsets.a = row_offset[vector_id];
  next_row_offsets.b = row_offset[vector_id + 1];

  _type_matrix_data<I, T> next_ele1, next_ele2, next_ele3;
  const I outer_next_load_count = next_row_offsets.b - next_row_offsets.a;
  I _outer_next_start = next_row_offsets.a + vector_thread_id;
  if (vector_thread_id < outer_next_load_count) {
    // for vector size 2, each thread load 1 element.
    next_ele1.value = csr_val[_outer_next_start];
    next_ele1.col_ind = csr_col_ind[_outer_next_start];
  }
  _outer_next_start += VECTOR_SIZE;
  if (vector_thread_id + VECTOR_SIZE < outer_next_load_count) {
    // for vector size 2, each thread load 2 element.
    next_ele2.value = csr_val[_outer_next_start];
    next_ele2.col_ind = csr_col_ind[_outer_next_start];
  }
  _outer_next_start += VECTOR_SIZE;
  if (vector_thread_id + 2 * VECTOR_SIZE < outer_next_load_count) { // outer_next_load_count <= 6 &&
    // for vector size 2, each thread load 3 element.
    next_ele3.value = csr_val[_outer_next_start];
    next_ele3.col_ind = csr_col_ind[_outer_next_start];
  } else {
    // todo:
  }

  for (I row = vector_id; row < m; row += vector_num) {
    const I row_start = next_row_offsets.a;
    const I row_end = next_row_offsets.b;
    const I next_row_inx = row + vector_num;
    if (next_row_inx < m) {
#ifdef SYNC_LOAD
      global_load_intx2_sync(static_cast<const void *>(row_offset + next_row_inx), next_row_offsets);
#endif
#ifndef SYNC_LOAD
      next_row_offsets.a = row_offset[next_row_inx];
      next_row_offsets.b = row_offset[next_row_inx + 1];
#endif
    }

    T sum = static_cast<T>(0);

    const _type_matrix_data<I, T> cur_ele1 = next_ele1, cur_ele2 = next_ele2, cur_ele3 = next_ele3;

    const I next_load_count = next_row_offsets.b - next_row_offsets.a;
    I _local_next_start = next_row_offsets.a + vector_thread_id;
    if (vector_thread_id < next_load_count) {
      // for vector size 2, each thread load 1 element.
      next_ele1.value = csr_val[_local_next_start];
      next_ele1.col_ind = csr_col_ind[_local_next_start];
    }
    _local_next_start += VECTOR_SIZE;
    if (vector_thread_id + VECTOR_SIZE < next_load_count) {
      // for vector size 2, each thread load 2 element.
      next_ele2.value = csr_val[_local_next_start];
      next_ele2.col_ind = csr_col_ind[_local_next_start];
    }
    _local_next_start += VECTOR_SIZE;
    if (vector_thread_id + 2 * VECTOR_SIZE < next_load_count) { // next_load_count <= 6 &&
      // for vector size 2, each thread load 3 element.
      next_ele3.value = csr_val[_local_next_start];
      next_ele3.col_ind = csr_col_ind[_local_next_start];
    } else {
      // todo:
    }

    // calculation
    const I cur_data_count = row_end - row_start;
    if (vector_thread_id < cur_data_count) { // cur_data_count <= 2 &&
      asm_v_fma_f64(cur_ele1.value, device_ldg(x + cur_ele1.col_ind), sum);
    }
    if (vector_thread_id + VECTOR_SIZE < cur_data_count) {
      asm_v_fma_f64(cur_ele2.value, device_ldg(x + cur_ele2.col_ind), sum);
    }
    if (vector_thread_id + 2 * VECTOR_SIZE < cur_data_count) {
      asm_v_fma_f64(cur_ele3.value, device_ldg(x + cur_ele3.col_ind), sum);
    } else {
      // todo:
    }

    // reduce inside a vector
#pragma unroll
    for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, VECTOR_SIZE);
    }

    if (vector_thread_id == 0) {
      y[row] = device_fma(beta, y[row], alpha * sum);
    }
  }
}

/**
 * legacy double buffer support of vector strategy.
 * @tparam VECTOR_SIZE
 * @tparam WF_VECTORS
 * @tparam WF_SIZE
 * @tparam BLOCKS
 * @tparam I
 * @tparam T
 * @param m
 * @param alpha
 * @param beta
 * @param row_offset
 * @param csr_col_ind
 * @param csr_val
 * @param x
 * @param y
 * @return
 */
template <int VECTOR_SIZE, int WF_VECTORS, int WF_SIZE, int BLOCKS, typename I, typename T>
__global__ void spmv_vector_row_kernel_double_buffer_legacy(int m, const T alpha, const T beta, const I *row_offset,
                                                            const I *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int vector_thread_id = global_thread_id % VECTOR_SIZE; // local thread id in current vector
  const int vector_id = global_thread_id / VECTOR_SIZE;        // global vector id
  const int vector_num = gridDim.x * blockDim.x / VECTOR_SIZE; // total vectors on device

  // todo: assert(vector_id < m);
  _type_row_offsets next_row_offsets{0, 0};
  next_row_offsets.a = row_offset[vector_id];
  next_row_offsets.b = row_offset[vector_id + 1];
  _type_matrix_data<I, T> buffer1{
      csr_val[next_row_offsets.a + vector_thread_id],
      csr_col_ind[next_row_offsets.a + vector_thread_id],
  };

  for (I row = vector_id; row < m; row += vector_num) {
    const I row_start = next_row_offsets.a;
    const I row_end = next_row_offsets.b;
    const I next_row_inx = row + vector_num;
    if (next_row_inx < m) {
#ifdef SYNC_LOAD
      global_load_intx2_sync(static_cast<const void *>(row_offset + next_row_inx), next_row_offsets);
#endif
#ifndef SYNC_LOAD
      next_row_offsets.a = row_offset[next_row_inx];
      next_row_offsets.b = row_offset[next_row_inx + 1];
#endif
    }

    T sum = static_cast<T>(0);

    for (I i = row_start + vector_thread_id; i < row_end; i += VECTOR_SIZE) {
#ifdef SYNC_LOAD
      s_waitcnt(); // wait loading row offset and buffer1
#endif
      const T cur_csr_value = buffer1.value;
      const I cur_csr_col_inx = buffer1.col_ind;

      asm_v_fma_f64(cur_csr_value, device_ldg(x + cur_csr_col_inx), sum);
      const I next_col_inx = i + VECTOR_SIZE;
      // load next element
      if (next_col_inx < row_end) {
#ifdef SYNC_LOAD
        global_load_dbl_sync(static_cast<const void *>(csr_val + next_col_inx), buffer1.value);
        global_load_int_sync(static_cast<const void *>(csr_col_ind + next_col_inx), buffer1.col_ind);
#endif
#ifndef SYNC_LOAD
        buffer1.value = csr_val[next_col_inx];
        buffer1.col_ind = csr_col_ind[next_col_inx];
#endif
      }
    }

    // load first element of next row/line
    // note: what if the last row (or some row) have no element.
    const I next_ele_index = next_row_offsets.a + vector_thread_id;
    if (next_ele_index < next_row_offsets.b && next_row_inx < m) {
#ifdef SYNC_LOAD
      global_load_dbl_sync(static_cast<const void *>(csr_val + next_ele_index), buffer1.value);
      global_load_int_sync(static_cast<const void *>(csr_col_ind + next_ele_index), buffer1.col_ind);
#endif
#ifndef SYNC_LOAD
      buffer1.value = csr_val[next_ele_index];
      buffer1.col_ind = csr_col_ind[next_ele_index];
#endif
    }

    // reduce inside a vector
#pragma unroll
    for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, VECTOR_SIZE);
    }

    if (vector_thread_id == 0) {
      y[row] = device_fma(beta, y[row], alpha * sum);
    }
  }
}

#define VECTOR_KERNEL_WRAPPER_DB_BUFFER(N)                                                                             \
  (spmv_vector_row_kernel_double_buffer<N, (64 / N), 64, 512, int, double>)<<<512, 256>>>(m, alpha, beta, rowptr,      \
                                                                                          colindex, value, x, y)

#define VECTOR_KERNEL_WRAPPER_DB_BUFFER_LEGACY(N)                                                                      \
  (spmv_vector_row_kernel_double_buffer_legacy<N, (64 / N), 64, 512, int, double>)<<<512, 256>>>(                      \
      m, alpha, beta, rowptr, colindex, value, x, y)

#endif // SPMV_ACC_VECTOR_ROW_OPT_DOUBLE_BUFFER_HPP
