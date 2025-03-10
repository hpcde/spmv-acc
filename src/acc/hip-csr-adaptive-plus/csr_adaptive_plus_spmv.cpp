//
// Created by genshen on 2024/12/31.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#include "common/macros.h"
#include "common/mem_bandwidth.hpp"
#include "csr_adaptive_plus_analyze.h"
#include "csr_adaptive_plus_config.h"
#include "csr_adaptive_plus_spmv.h"
#include "csr_adaptive_plus_spmv_imp.inl"

template <typename I, typename T, int R, int THREADS_PER_BLOCK, int MIN_NNZ_PER_BLOCK, int REDUCE_OPTION, int VEC_SIZE>
csr_adaptive_plus_analyze_info<I>
csr_adaptive_plus_sparse_spmv_analyze(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                      const csr_desc<I, T> d_csr_desc, const T *x, T *y) {
  VAR_FROM_CSR_DESC(d_csr_desc);
  const I nnz = h_csr_desc.row_ptr[m];

  // start from 0. additional element for the starting 0.
  const I max_break_points_len = nnz / MIN_NNZ_PER_BLOCK + ((nnz % MIN_NNZ_PER_BLOCK) == 0 ? 0 : 1) + 1;
  std::vector<I> break_points_host;               // each block has a break point.
  std::vector<I> first_process_block_of_row_host; // first block for processing the row
  break_points_host.reserve(max_break_points_len);
  first_process_block_of_row_host.resize(m + 1);

  const I HIP_BLOCKS = csr_adaptive_plus_analyze_imp<int, THREADS_PER_BLOCK, VEC_SIZE>(
      m, nnz, MIN_NNZ_PER_BLOCK, break_points_host, first_process_block_of_row_host, h_csr_desc.row_ptr,
      d_csr_desc.row_ptr);

  csr_adaptive_plus_analyze_info<I> info;
  info.HIP_BLOCKS = HIP_BLOCKS;

  hipMalloc((void **)&(info.break_points), break_points_host.size() * sizeof(I));
  hipMalloc((void **)&(info.first_process_block_of_row), (m + 1) * sizeof(I));
  hipMemcpy(info.break_points, break_points_host.data(), break_points_host.size() * sizeof(I), hipMemcpyHostToDevice);
  hipMemcpy(info.first_process_block_of_row, first_process_block_of_row_host.data(), (m + 1) * sizeof(I),
            hipMemcpyHostToDevice);

  if (spmv::gpu::adaptive_plus::DEBUG) {
    for (I i = 1; i < (HIP_BLOCKS + 1); i++) {
      printf("block %d: [%d %d) rows: %d, nnz: %d\n", i, break_points_host[i - 1], break_points_host[i],
             break_points_host[i] - break_points_host[i - 1],
             h_csr_desc.row_ptr[break_points_host[i]] - h_csr_desc.row_ptr[break_points_host[i - 1]]);
    }
    printf("\\\\(total rows %d).\n", m);
  }

  return info;
}

template <typename I, typename T, int R, int THREADS_PER_BLOCK, int MIN_NNZ_PER_BLOCK, int REDUCE_OPTION, int VEC_SIZE>
void csr_adaptive_plus_sparse_spmv_kernel(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                          const csr_desc<I, T> d_csr_desc, const T *x, T *y,
                                          csr_adaptive_plus_analyze_info<I> info) {
  VAR_FROM_CSR_DESC(d_csr_desc);
  // clean array y:
  // hipMemset(y, 0, m * sizeof(double));

  constexpr I MAX_ROWS_PER_BLOCK = 0; // unused.

  (line_enhance_plus_kernel<REDUCE_OPTION, __WRAP_SIZE__, VEC_SIZE, MAX_ROWS_PER_BLOCK, MIN_NNZ_PER_BLOCK, R,
                            THREADS_PER_BLOCK, I, T>)<<<info.HIP_BLOCKS, THREADS_PER_BLOCK>>>(
      m, info.break_points, info.first_process_block_of_row, alpha, beta, rowptr, colindex, value, x, y);
}

template <typename I> void csr_adaptive_plus_sparse_spmv_destroy(csr_adaptive_plus_analyze_info<I> info) {
  hipFree(info.break_points);
  hipFree(info.first_process_block_of_row);
}

template <typename I, typename T>
void csr_adaptive_plus_sparse_spmv(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                   const csr_desc<I, T> d_csr_desc, const T *x, T *y) {
  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  // each block can process `MIN_NNZ_PER_BLOCK` non-zeros.
  constexpr int MIN_NNZ_PER_BLOCK = 2 * R * THREADS_PER_BLOCK; // fixme: add more nnz.

  constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
  constexpr int VEC_SIZE = 8; // note: if using direct reduce, VEC_SIZE must set to 1.

  auto info =
      csr_adaptive_plus_sparse_spmv_analyze<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
          trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);

  csr_adaptive_plus_sparse_spmv_kernel<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
      trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, info);

  csr_adaptive_plus_sparse_spmv_destroy<I>(info);
}

template void csr_adaptive_plus_sparse_spmv<int, double>(int trans, const double alpha, const double beta,
                                                         const csr_desc<int, double> h_csr_desc,
                                                         const csr_desc<int, double> d_csr_desc, const double *x,
                                                         double *y);

template void csr_adaptive_plus_sparse_spmv_kernel<int, double, 2, 512, 2048, 1, 8>(
    int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,
    const csr_desc<int, double> d_csr_desc, const double *x, double *y, csr_adaptive_plus_analyze_info<int> info);

template void csr_adaptive_plus_sparse_spmv_destroy<int>(csr_adaptive_plus_analyze_info<int> info);
