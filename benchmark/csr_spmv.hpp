//
// Created by reget on 2021/11/16.
//

#ifndef SPMV_ACC_CSR_SPMV
#define SPMV_ACC_CSR_SPMV

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "api/types.h"
#include "utils.hpp"
#include "utils/benchmark_time.h"
#include "verification.h"

/**
 * @tparam T type of data
 * @param m rows
 * @param n cols
 * @param nnz number of non-zeros
 * @param time time in us
 */
template <typename T>
void print_statistics(std::string mtx_name, std::string strategy_name, int rows, int cols, int nnz, BenchmarkTime bmt) {
  double mem_bytes = static_cast<double>(sizeof(T) * (2 * rows + nnz) + sizeof(int) * (rows + 1 + nnz));

  double calc_time_bandwidth = (mem_bytes + 0.0) / (1024 * 1024 * 1024) / (bmt.calc_time_use / 1e3 / 1e3);
  double calc_time_gflops = static_cast<double>(2 * nnz) / bmt.calc_time_use / 1e3;

  double total_time_bandwidth = (mem_bytes + 0.0) / (1024 * 1024 * 1024) / (bmt.total_time_use / 1e3 / 1e3);
  double total_time_gflops = static_cast<double>(2 * nnz) / bmt.total_time_use / 1e3;

  std::cout << "matrix name: " << mtx_name << ", strategy name: " << strategy_name << ", rows: " << rows
            << ", cols: " << cols << ", nnz: " << nnz << ", nnz/row: " << (nnz + 0.0) / rows
            << ", GB/s(calc_time): " << calc_time_bandwidth << ", GFLOPS(calc_time): " << calc_time_gflops
            << ", GB/s(total_time): " << total_time_bandwidth << ", GFLOPS(total_time): " << total_time_gflops
            << ", mid pre cost: " << bmt.pre_time_use << ", mid calc cost: " << bmt.calc_time_use
            << ", mid destroy cost: " << bmt.destroy_time_use << ", mid total cost: " << bmt.total_time_use
            << std::endl;
}

// see also: https://stackoverflow.com/questions/4173254/what-is-the-curiously-recurring-template-pattern-crtp
template <class T> struct CsrSpMV {
  /**
   *
   * @param trans Matrix transpose. current only support operation_none.
   * @param alpha alpha in y = alpha*A*x + beta*y
   * @param beta beta in y = alpha*A*x + beta*y
   * @param h_rowptr row offset on host size.
   * @param d_csr_desc csr description.
   * @param x device vector x.
   * @param y device vector y.
   * @param bmt spmv elapsed time
   * @return bool is support alpha and beta
   * @note return true: y = alpha*A*x + beta*y else: y = A*x
   */
  bool csr_spmv(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    return static_cast<T *>(this)->csr_spmv_impl(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }

  void test(std::string mtx_path, const std::string &strategy_name, enum sparse_operation operation, dtype alpha,
            dtype beta, type_csr h_csr, type_csr d_csr, host_vectors<dtype> h_vectors, dtype *&dev_x, dtype *&dev_y) {
    BenchmarkTimeArray bmt_array;
    // warm up GPU
    for (int i = 0; i < 10; ++i) {
      // call sparse spmv
      HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
      try {
        csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y, nullptr);
      } catch (const std::runtime_error &error) {
        std::cout << "matrix name: " << mtx_path << ", strategy name: " << get_strategy_name() << std::endl;
        std::cout << "error occur, return" << std::endl;
        std::cerr << error.what() << std::endl;
        hipDeviceSynchronize();
        memcpy(h_vectors.hhY, h_vectors.temphY, d_csr.rows * sizeof(dtype));
        HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
        return;
      }
    }
    hipDeviceSynchronize();

    // execute device SpMV
    for (int i = 0; i < BENCHMARK_ARRAY_SIZE; i++) {
      BenchmarkTime bmt;
      HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
      csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y, &bmt);
      hipDeviceSynchronize();
      bmt_array.append(bmt);
    }

    // device result check
    HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, h_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
    bool flag = csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y, nullptr);
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
    print_statistics<dtype>(mtx_path, strategy_name, h_csr.rows, h_csr.cols, h_csr.nnz, bmt_array.get_mid_time());
    verify(h_vectors.hY, h_vectors.hhY, h_csr.rows);
    memcpy(h_vectors.hhY, h_vectors.temphY, d_csr.rows * sizeof(dtype));
    HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
  }
};

#include "spmv_acc_impl.inl"

// rocm
#ifdef __HIP_PLATFORM_HCC__
#include "rocsparse_impl.inl"
#endif

// cuda
#ifndef __HIP_PLATFORM_HCC__
#include "cub_impl.inl"
#include "cusparse_impl.inl"
#include "hola_impl.inl"
#endif

#endif
