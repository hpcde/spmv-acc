//
// Created by dly on 2021/4/19.
//
// spmv_csr_scalar_kernel version: one thread process one row of matrix A.

#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.

__global__ void native_thread_row(int trans, const double alpha, const double beta, int m, int n, const int *rowptr,
                                  const int *colindex, const double *value, const double *x, double *y) {
  int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int next_row_step = blockDim.x * gridDim.x;
  double y0 = 0.0;
  for (int i = thread_id; i < m; i += next_row_step) {
    y0 = 0.0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
      y0 += value[j] * x[colindex[j]];
    }
    y[i] = alpha * y0 + beta * y[i];
  }
}

/**
 * another thread row strategy with different data loading method.
 *
 * @tparam ROWS_PER_WF each wavefront may process 64*N rows.
 * @tparam WF_SIZE threads number in a wavefront
 * @tparam BLOCKS total blocks in system.
 * @tparam THREADS threads number in block.
 *
 * @tparam I index type
 * @tparam T data type. (e.g matrix value, vector x, y)
 * @param alpha, beta alpha and beta value
 * @param m row number in matrix A.
 * @param row_ptr row offset pointer in CSR format.
 * @param csr_col_inx column index pointer in CSR format.
 * @param csr_val matrix value in CSR format.
 * @param x vector x for @param alpha * A*x
 * @param y result vector for y = alpha*A*x + beta*y.
 * @return
 */
template <int N, int WF_SIZE, int THREADS, typename I, typename T>
__global__ void kernel_thread_row(const T alpha, const T beta, const I m, const I *__restrict__ row_ptr,
                                  const I *__restrict__ csr_col_inx, const T *__restrict__ csr_val,
                                  const T *__restrict__ x, T *__restrict__ y) {
  int t_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int global_threads_num = blockDim.x * gridDim.x;

  const int g_wf_id = t_id / WF_SIZE;                     // global wavefront id
  const int tid_in_wf = t_id % WF_SIZE;                   // thread id in current wavefront
  const int global_wf_num = global_threads_num / WF_SIZE; // wavefront number in system
  const int wf_id_in_block = threadIdx.x / WF_SIZE;       // wavefront id in current block

  constexpr int MAX_ROW_NNZ = 5;
  constexpr int shared_len = N * THREADS * MAX_ROW_NNZ;
  __shared__ T shared_data[shared_len];
  const int shared_len_wf = N * WF_SIZE * MAX_ROW_NNZ;            // data size in a wavefront.
  const int wf_shared_start_inx = wf_id_in_block * shared_len_wf; // start index of shared mem for current
  T *_wf_shared_val = shared_data + wf_shared_start_inx;          // LDS memory for current wavefront.

  // In each loop, each thread process N rows.
  const I wf_rounds = m / WF_SIZE + (m % (WF_SIZE) == 0 ? 0 : 1);

  for (I i = N * g_wf_id; i < wf_rounds; i += N * global_wf_num) {
    // each wavefront process `N * WF_SIZE` rows.
    // In a wavefront, read data from row g_wf_id to g_wf_id + N*WF_SIZE.
    const I wf_row_start_id = min(i * WF_SIZE, m - 1);
    const I wf_row_end_id = min((i + 1) * WF_SIZE, m);
    // we have: wf_row_start_id < wf_row_end_id and wf_row_start_id < m.
    const I wf_start_index = row_ptr[wf_row_start_id];
    const I wf_end_index = row_ptr[wf_row_end_id];
    for (I j = wf_start_index + tid_in_wf; j < wf_end_index; j += WF_SIZE) {
      const T local_val = csr_val[j] * x[csr_col_inx[j]];
      // sum += local_val;
      _wf_shared_val[j - wf_start_index] = local_val;
    }

    // reduction
    // todo: multiples rows per thread support in reduction
    // The last row may be reduced and stored more than once by threads in the last wavefront,
    // but it does not matter.
    const I reduce_row_id = min(wf_row_start_id + tid_in_wf, m - 1);
    const I reduce_start_index = row_ptr[reduce_row_id] - wf_start_index;
    const I reduce_end_index = row_ptr[reduce_row_id + 1] - wf_start_index;

    T sum = static_cast<T>(0);
    for (I k = reduce_start_index; k < reduce_end_index; k++) {
      sum += _wf_shared_val[k];
    }

    y[reduce_row_id] = alpha * sum + beta * y[reduce_row_id];
  }
}

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *d_row_ptr,
                 const int *d_csr_col_index, const double *d_csr_value, const double *d_x, double *d_y) {
  //  native_thread_row<<<1, 1024>>>(trans, alpha, beta, m, n, d_row_ptr, d_csr_col_index, d_csr_value, d_x, d_y);
  (kernel_thread_row<1, 64, 512, int, double>)<<<512, 512>>>(alpha, beta, m, d_row_ptr, d_csr_col_index, d_csr_value,
                                                             d_x, d_y);
}
