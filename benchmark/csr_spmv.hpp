//
// Created by reget on 2021/11/16.
//

#ifndef SPMV_ACC_CSR_SPMV
#define SPMV_ACC_CSR_SPMV

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "api/types.h"
#include "timer.h"
#include "utils.hpp"
#include "verification.h"

/**
 * @tparam T type of data
 * @param m rows
 * @param n cols
 * @param nnz number of non-zeros
 * @param time time in us
 */
template <typename T> void print_statistics(std::string mtx_name, int rows, int cols, int nnz, double time) {
  double mem_bytes = static_cast<double>(sizeof(T) * (2 * rows + nnz) + sizeof(int) * (rows + 1 + nnz));
  double bandwidth = (mem_bytes + 0.0) / (1024 * 1024 * 1024) / (time / 1e3 / 1e3);
  double gflops = static_cast<double>(2 * nnz) / time / 1e3;

  std::cout << "matrix name: " << mtx_name << ", rows: " << rows << ", cols: " << cols << ", nnz: " << nnz
            << ", nnz/row: " << (nnz + 0.0) / rows << ", GB/s: " << bandwidth << ", GFLOPS: " << gflops
            << ", usec: " << time << std::endl;
}

// see also: https://stackoverflow.com/questions/4173254/what-is-the-curiously-recurring-template-pattern-crtp
template <class T> struct CsrSpMV {
  bool csr_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    return static_cast<T *>(this)->csr_spmv_impl(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
  }

  void test(std::string mtx_path, enum sparse_operation operation, dtype alpha, dtype beta, type_csr h_csr,
            type_csr d_csr, host_vectors<dtype> h_vectors, dtype *&dev_x, dtype *&dev_y) {
    // warm up GPU
    for (int i = 0; i < 10; ++i) {
      // call sparse spmv
      HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
      csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y);
    }
    hipDeviceSynchronize();

    my_timer timer1;
    timer1.start();
    // execute device SpMV
    for (int i = 0; i < 1; i++) {
      csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y);
      hipDeviceSynchronize();
    }
    timer1.stop();

    // device result check
    HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, h_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
    bool flag = csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y);
    HIP_CHECK(hipMemcpy(h_vectors.hY, dev_y, d_csr.rows * sizeof(dtype), hipMemcpyDeviceToHost));

    // host side verification
    if (flag == true) {
      // y = alpha*A*x + beta*y
      host_spmv(alpha, beta, h_csr.values, h_csr.row_ptr, h_csr.col_index, h_csr.rows, h_csr.cols, h_csr.nnz,
                h_vectors.hX, h_vectors.hhY);
    } else {
      // y = A*x
      host_spmv(h_csr.values, h_csr.row_ptr, h_csr.col_index, h_csr.rows, h_csr.cols, h_csr.nnz, h_vectors.hX,
                h_vectors.hhY);
    }
    print_statistics<dtype>(mtx_path, h_csr.rows, h_csr.cols, h_csr.nnz, timer1.time_use);
    verify(h_vectors.hY, h_vectors.hhY, h_csr.rows);
    memcpy(h_vectors.hhY, h_vectors.temphY, d_csr.rows * sizeof(dtype));
    HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
  }
};

#include "spmv_acc_impl.inl"

// rocm
#ifdef __HIP_PLATFORM_HCC__
#endif

// cuda
#ifndef __HIP_PLATFORM_HCC__
#include "cub_impl.inl"
#include "cusparse_impl.inl"
#include "hola_impl.inl"
#endif

#endif