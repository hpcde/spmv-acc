//
// Created by chu genshen on 2021/9/14.
//
// This is the cli entrypoint with main function.
//
// Note, there is also another cli `main` entrypoint in src directory,
// which is used for [PRA](https://cas-pra.sugon.com) purpose.
//

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "api/spmv.h"
#include "api/types.h"
#include "building_config.h"
#include "clipp.h"

#include "csr_mtx_reader.hpp"
#include "matrix_market_reader.hpp"
#include "sparse_format.h"
#include "timer.h"
#include "utils.hpp"
#include "verification.h"

void test_spmv(std::string mtx_path, type_csr h_csr, host_vectors<dtype> h_vectors);

int main(int argc, char **argv) {
  std::string mtx_path = "", fmt = "csr";

  auto cli =
      (clipp::value("input file", mtx_path),
       clipp::option("-f", "--format").doc("input matrix format, can be `csr` (default) or `mm` (matrix market)") &
           clipp::value("format", fmt));

  if (!parse(argc, argv, cli)) {
    std::cout << clipp::make_man_page(cli, argv[0]);
    return 0;
  }

  type_csr h_csr;
  host_vectors<dtype> h_vectors{};
  if (fmt == "csr") {
    csr_mtx_reader<int, dtype> csr_reader(mtx_path);
    csr_reader.fill_mtx();
    csr_reader.close_stream();

    h_csr.rows = csr_reader.rows();
    h_csr.cols = csr_reader.cols();
    h_csr.nnz = csr_reader.nnz();

    // don't allocate new memory, just reuse memory in file parsing.
    // array data in `h_csr` is keep in instance `csr_reader`.
    csr_reader.as_raw_ptr(h_csr.values, h_csr.col_index, h_csr.row_ptr, h_vectors.hX);
    create_host_data(h_csr, h_vectors);
    test_spmv(mtx_path, h_csr, h_vectors);
  } else {
    matrix_market_reader<int, dtype> mm_reader;
    coo_mtx<int, dtype> coo_sparse = mm_reader.load_mat(mtx_path);
    h_csr = coo_sparse.to_csr();
    create_host_data(h_csr, h_vectors, true);
    test_spmv(mtx_path, h_csr, h_vectors);
  }
}

void test_spmv(std::string mtx_path, type_csr h_csr, host_vectors<dtype> h_vectors) {
  hipSetDevice(0);
  dtype *dev_x, *dev_y;
  type_csr d_csr = create_device_data(h_csr, h_vectors.hX, h_vectors.temphY, dev_x, dev_y);

  // set parameters
  enum sparse_operation operation = operation_none;
  dtype alpha = 1.0;
  dtype beta = 1.0;

  // warm up GPU
  for (int i = 0; i < 10; ++i) {
    // call sparse spmv
    HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
    sparse_csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y);
  }
  hipDeviceSynchronize();

  my_timer timer1;
  timer1.start();
  // execute device SpMV
  for (int i = 0; i < 1; i++) {
    sparse_csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y);
    hipDeviceSynchronize();
  }
  timer1.stop();

  // device result check
  HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, h_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
  sparse_csr_spmv(operation, alpha, beta, h_csr.as_const(), d_csr.as_const(), dev_x, dev_y);
  HIP_CHECK(hipMemcpy(h_vectors.hY, dev_y, d_csr.rows * sizeof(dtype), hipMemcpyDeviceToHost));

#ifdef gpu
  // device side verification
  HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(dtype), hipMemcpyHostToDevice))
  my_timer timer2;
  timer2.start();
  rocsparse(d_csr, dev_x, dev_y, alpha, beta);
  hipDeviceSynchronize();
  timer2.stop();
  std::cout << "rocsparse elapsed time:" << timer2.time_use << "(us)" << std::endl;
  HIP_CHECK(hipMemcpy(h_vectors.hhY, dev_y, d_csr.rows * sizeof(dtype), hipMemcpyDeviceToHost));
#else
  // host side verification
  host_spmv(alpha, beta, h_csr.values, h_csr.row_ptr, h_csr.col_index, h_csr.rows, h_csr.cols, h_csr.nnz, h_vectors.hX,
            h_vectors.hhY);
#endif

  verify(h_vectors.hY, h_vectors.hhY, h_csr.rows);
  std::cout << mtx_path << " elapsed time:" << timer1.time_use << "(us)" << std::endl;

  return;
}
