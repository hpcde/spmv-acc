//
// Created by reget on 2025/03/11.
//

#include "csr_adaptive_plus_benchmark.h"
#include "hip-csr-adaptive-plus/csr_adaptive_plus_spmv.h"

/**
 * the same as csr_adaptive_plus_sparse_spmv, but with timer in it.
 */
template <typename I, typename T>
void csr_adaptive_plus_sparse_spmv_with_profile(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                                const csr_desc<I, T> d_csr_desc, const T *x, T *y, my_timer &pre_timer,
                                                my_timer &calc_timer, my_timer &destroy_timer) {
  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  // each block can process `MIN_NNZ_PER_BLOCK` non-zeros.
  constexpr int MIN_NNZ_PER_BLOCK = 2 * R * THREADS_PER_BLOCK; // fixme: add more nnz.

  constexpr int REDUCE_OPTION = 1; // it is LE_REDUCE_OPTION_VEC;
  constexpr int VEC_SIZE = 8; // note: if using direct reduce, VEC_SIZE must set to 1.

  pre_timer.start();
  auto info =
      csr_adaptive_plus_sparse_spmv_analyze<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
          trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
  pre_timer.stop();

  calc_timer.start();
  // launch the kernerl;
  csr_adaptive_plus_sparse_spmv_kernel<I, T, R, THREADS_PER_BLOCK, MIN_NNZ_PER_BLOCK, REDUCE_OPTION, VEC_SIZE>(
      trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, info);
  hipDeviceSynchronize();
  calc_timer.stop();

  destroy_timer.start();
  csr_adaptive_plus_sparse_spmv_destroy<I>(info);
  destroy_timer.stop();
}

template void csr_adaptive_plus_sparse_spmv_with_profile<int, double>(int trans, const double alpha, const double beta,
                                                                      const csr_desc<int, double> h_csr_desc,
                                                                      const csr_desc<int, double> d_csr_desc,
                                                                      const double *x, double *y, my_timer &pre_timer,
                                                                      my_timer &calc_timer, my_timer &destroy_timer);
