//
// Created by genshen on 2021/06/30.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

template <int ROW_SIZE, int WF_SIZE, int BLOCKS, typename I, typename T>
__global__ void spmv_line_kernel(int m, const T alpha, const T beta, const I *row_offset, const I *csr_col_ind,
                                 const T *csr_val, const T *x, T *y);

template <int ROW_SIZE, int MAX_ROW_NNZ, typename I, typename T>
__global__ void spmv_line_one_pass_kernel(int m, const T alpha, const T beta, const I *row_offset, const I *csr_col_ind,
                                          const T *csr_val, const T *x, T *y);

void line_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                      const double *value, const double *x, double *y);

#include "line_imp_one_pass.inl"
#include "line_kernel_imp.inl"

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H
