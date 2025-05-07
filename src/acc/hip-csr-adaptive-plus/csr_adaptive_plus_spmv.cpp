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

template <int VEC_SIZE, int R, int THREADS_PER_BLOCK, int MIN_NNZ_PER_BLOCK, typename I, typename T>
void csr_adaptive_plus_sparse_spmv_wrapper(SpMVAccHanele *handle, int trans, const T alpha, const T beta,
                                           const csr_desc<I, T> h_csr_desc, const csr_desc<I, T> d_csr_desc, const T *x,
                                           T *y) {
  constexpr int REDUCE_OPTION = VEC_SIZE == 1 ? LE_REDUCE_OPTION_DIRECT : LE_REDUCE_OPTION_VEC;

  auto info =
      csr_adaptive_plus_sparse_spmv_analyze<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
          trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);

  csr_adaptive_plus_sparse_spmv_kernel<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
      trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, info);

  csr_adaptive_plus_sparse_spmv_destroy<I>(info);
}

// run spmv with profile info in it
template <int VEC_SIZE, int R, int THREADS_PER_BLOCK, int MIN_NNZ_PER_BLOCK, typename I, typename T>
void csr_adaptive_plus_sparse_spmv_profile(SpMVAccHanele *handle, int trans, const T alpha, const T beta,
                                           const csr_desc<I, T> h_csr_desc, const csr_desc<I, T> d_csr_desc, const T *x,
                                           T *y) {
  constexpr int REDUCE_OPTION = VEC_SIZE == 1 ? LE_REDUCE_OPTION_DIRECT : LE_REDUCE_OPTION_VEC;

  const auto start1 = std::chrono::high_resolution_clock::now();
  auto info =
      csr_adaptive_plus_sparse_spmv_analyze<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
          trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
  const auto end1 = std::chrono::high_resolution_clock::now();
  const double pre_time = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

  const auto start2 = std::chrono::high_resolution_clock::now();
  csr_adaptive_plus_sparse_spmv_kernel<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
      trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, info);
  hipDeviceSynchronize();
  const auto end2 = std::chrono::high_resolution_clock::now();
  const double kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

  const auto start3 = std::chrono::high_resolution_clock::now();
  csr_adaptive_plus_sparse_spmv_destroy<I>(info);
  const auto end3 = std::chrono::high_resolution_clock::now();
  const double des_time = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count();

  handle->profile_analyze_time = pre_time;
  handle->profile_kernel_time = kernel_time;
  handle->profile_destroy_time = des_time;
}

template <bool PROFILE, typename I, typename T>
void csr_adaptive_plus_sparse_spmv(SpMVAccHanele *handle, int trans, const T alpha, const T beta,
                                   const csr_desc<I, T> h_csr_desc, const csr_desc<I, T> d_csr_desc, const T *x, T *y) {
  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  // each block can process `MIN_NNZ_PER_BLOCK` non-zeros.
  constexpr int MIN_NNZ_PER_BLOCK = 2 * R * THREADS_PER_BLOCK; // fixme: add more nnz.

  const int avg_eles_per_row = h_csr_desc.row_ptr[h_csr_desc.rows] / h_csr_desc.rows;

#define SPMV_CSR_ADPTIVE_PLUE_WRAPPER(V)                                                                               \
  constexpr int VEC_SIZE = V;                                                                                          \
  if (PROFILE) {                                                                                                       \
    csr_adaptive_plus_sparse_spmv_profile<VEC_SIZE, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, I, T>(                    \
        handle, trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);                                                     \
  } else {                                                                                                             \
    csr_adaptive_plus_sparse_spmv_wrapper<VEC_SIZE, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, I, T>(                    \
        handle, trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);                                                     \
  }

  if (avg_eles_per_row <= 2 || __WF_SIZE__ <= 2) {
    SPMV_CSR_ADPTIVE_PLUE_WRAPPER(1);
  } else if (avg_eles_per_row <= 4 || __WF_SIZE__ <= 2) {
    SPMV_CSR_ADPTIVE_PLUE_WRAPPER(2);
  } else if (avg_eles_per_row <= 8 || __WF_SIZE__ <= 4) {
    SPMV_CSR_ADPTIVE_PLUE_WRAPPER(4);
  } else if (avg_eles_per_row <= 16 || __WF_SIZE__ <= 8) {
    SPMV_CSR_ADPTIVE_PLUE_WRAPPER(8);
  } else if (avg_eles_per_row <= 32 || __WF_SIZE__ <= 16) {
    SPMV_CSR_ADPTIVE_PLUE_WRAPPER(16);
  } else if (avg_eles_per_row <= 64 || __WF_SIZE__ <= 32) {
    SPMV_CSR_ADPTIVE_PLUE_WRAPPER(32);
  } else {
    SPMV_CSR_ADPTIVE_PLUE_WRAPPER(64);
  }

#undef SPMV_CSR_ADPTIVE_PLUE_WRAPPER
}

template void csr_adaptive_plus_sparse_spmv<true, int, double>(SpMVAccHanele *handle, int trans, const double alpha,
                                                               const double beta,
                                                               const csr_desc<int, double> h_csr_desc,
                                                               const csr_desc<int, double> d_csr_desc, const double *x,
                                                               double *y);
template void csr_adaptive_plus_sparse_spmv<false, int, double>(SpMVAccHanele *handle, int trans, const double alpha,
                                                                const double beta,
                                                                const csr_desc<int, double> h_csr_desc,
                                                                const csr_desc<int, double> d_csr_desc, const double *x,
                                                                double *y);

template void csr_adaptive_plus_sparse_spmv_destroy<int>(csr_adaptive_plus_analyze_info<int> info);

// int VEC_SIZE, int R, int THREADS_PER_BLOCK, int MIN_NNZ_PER_BLOCK
#define CSR_ADAPTIVE_PLUS_WRAPPER_INS(V, R, THREADS, MIN_NNZ)                                                          \
  template void csr_adaptive_plus_sparse_spmv_wrapper<V, R, THREADS, MIN_NNZ, int, double>(                            \
      SpMVAccHanele * handle, int trans, const double alpha, const double beta,                                        \
      const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc, const double *x, double *y);     \
  template void csr_adaptive_plus_sparse_spmv_profile<V, R, THREADS, MIN_NNZ, int, double>(                            \
      SpMVAccHanele * handle, int trans, const double alpha, const double beta,                                        \
      const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc, const double *x, double *y);     \
  template void csr_adaptive_plus_sparse_spmv_kernel<int, double, R, THREADS, MIN_NNZ, LE_REDUCE_OPTION_VEC, V>(       \
      int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,                        \
      const csr_desc<int, double> d_csr_desc, const double *x, double *y, csr_adaptive_plus_analyze_info<int> info);

// R=1, THREADS=256
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 1, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 1, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 1, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 1, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 1, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 1, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 1, 256, 1024);

// R=2, THREADS=256
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 2, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 2, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 2, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 2, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 2, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 2, 256, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 2, 256, 1024);

// R=1, THREADS=512
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 1, 512, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 1, 512, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 1, 512, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 1, 512, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 1, 512, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 1, 512, 1024);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 1, 512, 1024);

// R=2, THREADS=512
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 2, 512, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 2, 512, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 2, 512, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 2, 512, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 2, 512, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 2, 512, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 2, 512, 2048);

// R=1, THREADS=1024
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 1, 1024, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 1, 1024, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 1, 1024, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 1, 1024, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 1, 1024, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 1, 1024, 2048);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 1, 1024, 2048);

// R=2, THREADS=1024
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 2, 1024, 4096);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 2, 1024, 4096);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 2, 1024, 4096);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 2, 1024, 4096);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 2, 1024, 4096);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 2, 1024, 4096);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 2, 1024, 4096);

// R=2, THREADS=1024, X=4
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 2, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 2, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 2, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 2, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 2, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 2, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 2, 1024, 8192);

// R=4, THREADS=1024, X=2
CSR_ADAPTIVE_PLUS_WRAPPER_INS(64, 4, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(32, 4, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(16, 4, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(8, 4, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(4, 4, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(2, 4, 1024, 8192);
CSR_ADAPTIVE_PLUS_WRAPPER_INS(1, 4, 1024, 8192);
