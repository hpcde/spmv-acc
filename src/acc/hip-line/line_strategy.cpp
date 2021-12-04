//
// Created by genshen on 2021/7/15.
//

#include "line_strategy.h"
#include "../common/macros.h"

void line_sparse_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> d_csr_desc,
                      const double *x, double *y) {
  VAR_FROM_CSR_DESC(d_csr_desc)

  // const int avg_eles_per_row = ceil(d_csr_desc.nnz + 0.0 / m);
  const int avg_eles_per_row = d_csr_desc.nnz / m;
  // ROW_NUM * MAX_NNZ_NUM < HIP_THREAD
  constexpr int HIP_THREAD = 256;
  constexpr int R = 2;
  if (avg_eles_per_row <= 5) {
    constexpr int MAX_NNZ_NUM = 5;
    const int ROW_NUM = HIP_THREAD / MAX_NNZ_NUM * R;
    const int HIP_BLOCK = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
    LINE_ONE_PASS_KERNEL_WRAPPER(ROW_NUM, MAX_NNZ_NUM, HIP_BLOCK, HIP_THREAD);
  } else {
    constexpr int MAX_NNZ_NUM = 14;
    constexpr int ROW_NUM = HIP_THREAD / MAX_NNZ_NUM * R;
    const int HIP_BLOCK = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
    LINE_ONE_PASS_KERNEL_WRAPPER(ROW_NUM, MAX_NNZ_NUM, HIP_BLOCK, HIP_THREAD);
  }
  // LINE_KERNEL_WRAPPER(5);
}

template <int R, int BLOCK_LDS_SIZE, int VEC_SIZE, int HIP_THREADS, typename I, typename T>
void inline adaptive_line_wrapper(const I m, const T alpha, const T beta, const I *row_offset, const I *csr_col_ind,
                                  const T *csr_val, const T *x, T *y) {
  // note: we can make HIP_THREADS / VEC_SIZE * N = ROW_NUM (where N is an integer),
  // then each vector can only process N row if vector-row is selected.
  constexpr int ROW_NUM = HIP_THREADS / (2 * VEC_SIZE) * R;

  const int HIP_BLOCKS = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
  (spmv_adaptive_line_kernel<ROW_NUM, BLOCK_LDS_SIZE, HIP_THREADS, __WF_SIZE__, VEC_SIZE, int, double,
                             false>)<<<HIP_BLOCKS, HIP_THREADS>>>(m, alpha, beta, row_offset, csr_col_ind, csr_val, x,
                                                                  y);
}

void adaptive_line_sparse_spmv(int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc,
                               const double *x, double *y) {
  VAR_FROM_CSR_DESC(d_csr_desc)

  const int nnz_per_row = d_csr_desc.nnz / m;
  // make ROW_NUM * NNZ_PER_ROW <= HIP_THREAD
  constexpr int HIP_THREADS = 256;
  constexpr int R = 2;
  constexpr int BLOCK_LDS_SIZE = HIP_THREADS * R;

  if (nnz_per_row <= 4 || __WF_SIZE__ <= 2) {
    (adaptive_line_wrapper<R, BLOCK_LDS_SIZE, 2, HIP_THREADS, int, double>)(m, alpha, beta, rowptr, colindex, value, x,
                                                                            y);
  } else if (nnz_per_row <= 8 || __WF_SIZE__ <= 4) {
    (adaptive_line_wrapper<R, BLOCK_LDS_SIZE, 4, HIP_THREADS, int, double>)(m, alpha, beta, rowptr, colindex, value, x,
                                                                            y);
  } else if (nnz_per_row <= 16 || __WF_SIZE__ <= 8) {
    (adaptive_line_wrapper<R, BLOCK_LDS_SIZE, 8, HIP_THREADS, int, double>)(m, alpha, beta, rowptr, colindex, value, x,
                                                                            y);
  } else if (nnz_per_row <= 32 || __WF_SIZE__ <= 16) {
    (adaptive_line_wrapper<R, BLOCK_LDS_SIZE, 16, HIP_THREADS, int, double>)(m, alpha, beta, rowptr, colindex, value, x,
                                                                             y);
  } else if (nnz_per_row <= 64 || __WF_SIZE__ <= 32) {
    (adaptive_line_wrapper<R, BLOCK_LDS_SIZE, 32, HIP_THREADS, int, double>)(m, alpha, beta, rowptr, colindex, value, x,
                                                                             y);
  } else {
    (adaptive_line_wrapper<R, BLOCK_LDS_SIZE, 64, HIP_THREADS, int, double>)(m, alpha, beta, rowptr, colindex, value, x,
                                                                             y);
  }
}

void adaptive_line_sparse_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> d_csr_desc,
                               const double *x, double *y) {
  adaptive_line_sparse_spmv(trans, static_cast<double>(alpha), static_cast<double>(beta), d_csr_desc, x, y);
}
