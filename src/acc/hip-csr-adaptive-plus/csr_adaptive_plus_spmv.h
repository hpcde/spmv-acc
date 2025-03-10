//
// Created by genshen on 2024/12/31.
//

#ifndef SPMV_ACC_CSR_ADAPTIVE2_SPMV_H
#define SPMV_ACC_CSR_ADAPTIVE2_SPMV_H

#include "../api/types.h"

template <typename Index> struct csr_adaptive_plus_analyze_info {
  Index *break_points = nullptr;
  Index *first_process_block_of_row = nullptr;
  Index HIP_BLOCKS = 0;
};

template <typename I, typename T, int R, int THREADS_PER_BLOCK, int MIN_NNZ_PER_BLOCK, int REDUCE_OPTION, int VEC_SIZE>
csr_adaptive_plus_analyze_info<I>
csr_adaptive_plus_sparse_spmv_analyze(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                      const csr_desc<I, T> d_csr_desc, const T *x, T *y);

template <typename I, typename T, int R, int THREADS_PER_BLOCK, int MIN_NNZ_PER_BLOCK, int REDUCE_OPTION, int VEC_SIZE>
void csr_adaptive_plus_sparse_spmv_kernel(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                          const csr_desc<I, T> d_csr_desc, const T *x, T *y,
                                          csr_adaptive_plus_analyze_info<I> info);

template <typename I> void csr_adaptive_plus_sparse_spmv_destroy(csr_adaptive_plus_analyze_info<I> info);

template <typename I, typename T>
void csr_adaptive_plus_sparse_spmv(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                   const csr_desc<I, T> d_csr_desc, const T *x, T *y);

#endif // SPMV_ACC_CSR_ADAPTIVE2_SPMV_H
