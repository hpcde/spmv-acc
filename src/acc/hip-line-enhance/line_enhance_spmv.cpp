//
// Created by chu genshen on 2021/10/2.
//

#include "line_enhance_spmv.h"
#include "../common/macros.h"
#include "line_enhance_spmv_imp.h"

void line_enhance_sparse_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> d_csr_desc,
                              const double *x, double *y) {
  adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  return;

  constexpr int R = 2;
  constexpr int ROWS_PER_BLOCK = 32; // note: make sure ROWS_PER_BLOCK * VEC_SIZE <= THREADS_PER_BLOCK.

  constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
  constexpr int VEC_SIZE = 4; // note: if using direct reduce, VEC_SIZE must set to 1.

  VAR_FROM_CSR_DESC(d_csr_desc)

  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  constexpr int THREADS_PER_BLOCK = 512;
  LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
}

#define ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(R, ROWS_PER_BLOCK, REDUCE_OPTION, VEC_SIZE)                                 \
  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);                                                 \
  LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);

void adaptive_enhance_sparse_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> d_csr_desc,
                                  const double *x, double *y) {
  // common parameters:
  VAR_FROM_CSR_DESC(d_csr_desc)
  constexpr int THREADS_PER_BLOCK = 512;

  // for small matrix.
  const int mtx_nnz = d_csr_desc.nnz;
  const int nnz_per_row = mtx_nnz / m;
  if (mtx_nnz <= (1 << 24)) { // 2^24=16,777,216
    if (nnz_per_row >= 64) {
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 16, LE_REDUCE_OPTION_VEC, 32);
    } else if (nnz_per_row >= 32) { // matrix has long rows
      // R, ROWS_PER_BLOCK, REDUCE_OPTION, VEC_SIZE
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 32, LE_REDUCE_OPTION_VEC, 16);
    } else if (nnz_per_row >= 16) { // matrix has short rows, then, use less thread(e.g. direct reduction) for reduction
                                    //      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 32, LE_REDUCE_OPTION_VEC, 8);
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 32, LE_REDUCE_OPTION_VEC, 16);
    } else if (nnz_per_row >= 8) { // matrix has short rows, then, use less thread(e.g. direct reduction) for reduction
                                   //      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 32, LE_REDUCE_OPTION_VEC, 8);
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 32, LE_REDUCE_OPTION_VEC, 16);
    } else if (nnz_per_row >= 4) {
      //      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 64, LE_REDUCE_OPTION_VEC, 4);
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 48, LE_REDUCE_OPTION_VEC, 8);
    } else {
      //      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 128, LE_REDUCE_OPTION_DIRECT, 1);
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(4, 96, LE_REDUCE_OPTION_DIRECT, 2);
    }
    return;
  } else { // for large matrix
    constexpr int R = 2;
    if (nnz_per_row >= 24) { // long row matrix
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 64, LE_REDUCE_OPTION_VEC, 4);
    } else { // short row matrix, use direct reduce
      ADAPTIVE_ENHANCE_SPARSE_MV_WRAPPER(2, 128, LE_REDUCE_OPTION_DIRECT, 1);
    }
    return;
  }
}
