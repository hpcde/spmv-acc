//
// Created by genshen on 2021/7/22.
//

#ifndef SPMV_ACC_VECTOR_ROW_NATIVE_HPP
#define SPMV_ACC_VECTOR_ROW_NATIVE_HPP

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
template <int VECTOR_SIZE, int WF_SIZE, typename T>
__global__ void native_vector_row_kernel(int m, const T alpha, const T beta, const int *row_offset,
                                         const int *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int vector_thread_id = global_thread_id % VECTOR_SIZE; // local thread id in current vector
  const int vector_id = global_thread_id / VECTOR_SIZE;        // global vector id
  const int vector_num = gridDim.x * blockDim.x / VECTOR_SIZE; // total vectors on device

  for (int row = vector_id; row < m; row += vector_num) {
    const int row_start = row_offset[row];
    const int row_end = row_offset[row + 1];
    T sum = static_cast<T>(0);

    for (int i = row_start + vector_thread_id; i < row_end; i += VECTOR_SIZE) {
      asm_v_fma_f64(csr_val[i], device_ldg(x + csr_col_ind[i]), sum);
    }

    // reduce inside a vector
    for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, VECTOR_SIZE);
    }

    const int tid_in_wf = global_thread_id % WF_SIZE;
    const int wf_id = global_thread_id / WF_SIZE;
    const int WF_VECTORS = WF_SIZE / VECTOR_SIZE;

    typedef union dbl_b32 {
      double val;
      uint32_t b32[2];
    } dbl_b32_t;
    dbl_b32_t vec_sum;
    dbl_b32_t recv_sum;
    vec_sum.val = sum;

    int src_tid = tid_in_wf * VECTOR_SIZE + wf_id * WF_SIZE; // each vector's thread-0 in current wavefront
    if (tid_in_wf < WF_VECTORS) { // load each vector's sum to the first WF_VECTORS threads in a wavefront
      recv_sum.b32[0] = __hip_ds_bpermute(4 * src_tid, vec_sum.b32[0]);
      recv_sum.b32[1] = __hip_ds_bpermute(4 * src_tid, vec_sum.b32[1]);
    } else if (tid_in_wf % VECTOR_SIZE == 0) { // enable each thread in permute op
      recv_sum.b32[0] = __hip_ds_bpermute(4 * global_thread_id, vec_sum.b32[0]);
      recv_sum.b32[1] = __hip_ds_bpermute(4 * global_thread_id, vec_sum.b32[1]);
    }

    __syncthreads();

    int vec_row = row - tid_in_wf / VECTOR_SIZE + tid_in_wf;
    if (tid_in_wf < WF_VECTORS && vec_row < m) {
      y[vec_row] = device_fma(beta, y[vec_row], alpha * recv_sum.val);
    }
  }
}

#define NATIVE_VECTOR_KERNEL_WRAPPER(N)                                                                                \
  (native_vector_row_kernel<N, 64, double>)<<<512, 256>>>(m, alpha, beta, rowptr, colindex, value, x, y);

#endif // SPMV_ACC_VECTOR_ROW_NATIVE_HPP
