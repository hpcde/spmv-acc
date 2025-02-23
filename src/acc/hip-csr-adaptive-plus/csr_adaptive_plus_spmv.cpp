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
#include "csr_adaptive_plus_spmv_imp.inl"

void csr_adaptive_plus_sparse_spmv(int trans, const double alpha, const double beta,
                                   const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc,
                                   const double *x, double *y) {
  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  // each block can process `MIN_NNZ_PER_BLOCK` non-zeros.
  constexpr int MIN_NNZ_PER_BLOCK = R * THREADS_PER_BLOCK;

  constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
  constexpr int VEC_SIZE = 1; // note: if using direct reduce, VEC_SIZE must set to 1.

  VAR_FROM_CSR_DESC(d_csr_desc);
  const int nnz = h_csr_desc.row_ptr[m]; // todo:

  // start from 0. additional element for the starting 0.
  const int max_break_points_len = nnz / MIN_NNZ_PER_BLOCK + ((nnz % MIN_NNZ_PER_BLOCK) == 0 ? 0 : 1) + 1;
  std::vector<int> break_points_host;
  break_points_host.reserve(max_break_points_len);
  const int HIP_BLOCKS = csr_adaptive_plus_analyze<int, THREADS_PER_BLOCK, VEC_SIZE>(
      m, nnz, MIN_NNZ_PER_BLOCK, break_points_host, h_csr_desc.row_ptr, d_csr_desc.row_ptr);

  // each block has a break point.
  int *break_points;
  hipMalloc((void **)&break_points, (HIP_BLOCKS + 1) * sizeof(int));
  // hipMemset(break_points, 0, break_points_len * sizeof(int)); // todo: can be removed.
  hipMemcpy(break_points, break_points_host.data(), (HIP_BLOCKS + 1) * sizeof(int), hipMemcpyHostToDevice);

  if (spmv::gpu::adaptive_plus::DEBUG) {
    for (int i = 1; i < (HIP_BLOCKS + 1); i++) {
      printf("block %d: [%d %d) rows: %d, nnz: %d\n", i, break_points_host[i - 1], break_points_host[i],
             break_points_host[i] - break_points_host[i - 1],
             h_csr_desc.row_ptr[break_points_host[i]] - h_csr_desc.row_ptr[break_points_host[i - 1]]);
    }
    printf("\\\\(total rows %d).\n", m);
  }
  LE_PLUS_KERNEL_WRAPPER(REDUCE_OPTION, MIN_NNZ_PER_BLOCK, VEC_SIZE, R, HIP_BLOCKS, THREADS_PER_BLOCK);
}
