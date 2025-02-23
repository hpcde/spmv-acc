//
// Created by genshen on 2025/2/20.
//

#ifndef SPMV_ACC_CSR_ADAPTIVE2_SPMV_IMP_H
#define SPMV_ACC_CSR_ADAPTIVE2_SPMV_IMP_H

template <int WF_SIZE, int ROWS_PER_BLOCK, int R, int THREADS, typename I, typename T>
__global__ void line_enhance_plus_kernel(const int m, const T alpha, const T beta, const I *__restrict__ row_offset,
                                         const I *__restrict__ csr_col_ind, const T *__restrict__ csr_val,
                                         const T *__restrict__ x, T *__restrict__ y);

#include "csr_adaptive_plus_spmv_imp.inl"

#endif // SPMV_ACC_CSR_ADAPTIVE2_SPMV_IMP_H
