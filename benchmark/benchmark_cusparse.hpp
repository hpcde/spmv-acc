//
// Created by reget on 2021/11/16.
//

#ifndef SPMV_ACC_BENCHMARK_CUSPARSE_HPP
#define SPMV_ACC_BENCHMARK_CUSPARSE_HPP

#include <cusparse.h>

#include <api/types.h>

#include "timer.h"
#include "utils/benchmark_time.h"
#include "utils/timer_utils.h"

struct CuSparseGeneral : CsrSpMV {

  inline cusparseSpMVAlg_t get_spmv_algo() {
// CUSPARSE_MV_ALG_DEFAULT is deprecated in 11.2.1 and removed in 11.2.2
#if CUSPARSE_VERSION < 11201
    return CUSPARSE_MV_ALG_DEFAULT;
#else
    return CUSPARSE_SPMV_ALG_DEFAULT;
#endif
  }

  void csr_spmv_impl(int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {

    const double cu_alpha = static_cast<double>(alpha);
    const double cu_beta = static_cast<double>(beta);
    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;
    pre_timer.start();
    // Create cuSPARSE handle
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    // Create matrix, vector x and vector y
    cusparseSpMatDescr_t cu_mat;
    cusparseDnVecDescr_t cu_x, cu_y;
    cusparseCreateCsr(&cu_mat, h_csr_desc.rows, h_csr_desc.cols, h_csr_desc.nnz, (void *)d_csr_desc.row_ptr,
                      (void *)d_csr_desc.col_index, (void *)d_csr_desc.values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&cu_x, h_csr_desc.cols, (void *)x, CUDA_R_64F);
    cusparseCreateDnVec(&cu_y, h_csr_desc.rows, (void *)y, CUDA_R_64F);
    // Allocate an external buffer
    void *d_buffer = NULL;
    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &cu_alpha, cu_mat, cu_x, &cu_beta, cu_y,
                            CUDA_R_64F, get_spmv_algo(), &buffer_size);
    cudaMalloc(&d_buffer, buffer_size);
    pre_timer.stop();
    calc_timer.start();
    // Execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &cu_alpha, cu_mat, cu_x, &cu_beta, cu_y, CUDA_R_64F,
                 get_spmv_algo(), d_buffer);
    calc_timer.stop(true);
    destroy_timer.start();
    // Clear up on device
    cusparseDestroySpMat(cu_mat);
    cusparseDestroyDnVec(cu_x);
    cusparseDestroyDnVec(cu_y);
    cusparseDestroy(handle);
    destroy_timer.stop();
    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // SPMV_ACC_BENCHMARK_CUSPARSE_HPP
