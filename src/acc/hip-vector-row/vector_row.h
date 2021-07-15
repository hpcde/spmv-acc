//
// Created by chaohu on 2021/05/25.
//

#ifndef SPMV_ACC_VECTOR_ROW_H
#define SPMV_ACC_VECTOR_ROW_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

template <int VECTOR_SIZE, int WF_VECTORS, int WF_SIZE, int BLOCKS, typename T>
__global__ void spmv_vector_row_kernel(int m, const T alpha, const T beta, const int *row_offset,
                                       const int *csr_col_ind, const T *csr_val, const T *x, T *y);

template <int VECTOR_SIZE, int WF_VECTORS, int WF_SIZE, int BLOCKS, typename I, typename T>
__global__ void vector_row_kernel_pipeline(int m, const T alpha, const T beta, const I *row_offset,
                                           const I *csr_col_ind, const T *csr_val, const T *x, T *y);

template <int VECTOR_SIZE, int WF_VECTORS, int WF_SIZE, int BLOCKS, typename I, typename T>
__global__ void spmv_vector_row_kernel_double_buffer(int m, const T alpha, const T beta, const I *row_offset,
                                                     const I *csr_col_ind, const T *csr_val, const T *x, T *y);

template <int VECTOR_SIZE, int WF_VECTORS, int WF_SIZE, int BLOCKS, typename I, typename T>
__global__ void spmv_vector_row_kernel_double_buffer_legacy(int m, const T alpha, const T beta, const I *row_offset,
                                                            const I *csr_col_ind, const T *csr_val, const T *x, T *y);

void vec_row_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                         const int *colindex, const double *value, const double *x, double *y);

#include "opt_double_buffer.hpp"
#include "vector_row.inl"

#endif // SPMV_ACC_VECTOR_ROW_H
