//
// Created by genshen on 2021/11/15.
//

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "api/types.h"
#include "clipp.h"

#include "csr_mtx_reader.hpp"
#include "matrix_market_reader.hpp"
#include "sparse_format.h"
#include "utils.hpp"

#include "csr_spmv.hpp"

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
    destroy_host_data(h_vectors);
  } else {
    matrix_market_reader<int, dtype> mm_reader;
    coo_mtx<int, dtype> coo_sparse = mm_reader.load_mat(mtx_path);
    h_csr = coo_sparse.to_csr();
    create_host_data(h_csr, h_vectors, true);
    test_spmv(mtx_path, h_csr, h_vectors);
    destroy_host_data(h_vectors);
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
  // spmv-acc
  CsrSpMV<SpMVAccDefault> spmv_acc_default;
  CsrSpMV<SpMVAccAdaptive> spmv_acc_adaptive;
  CsrSpMV<SpMVAccBlockRow> spmv_acc_block_row;
  CsrSpMV<SpMVAccFlat> spmv_acc_flat;
  CsrSpMV<SpMVAccLight> spmv_acc_light;
  CsrSpMV<SpMVAccLine> spmv_acc_line;
  CsrSpMV<SpMVAccThreadRow> spmv_acc_thread_row;
  CsrSpMV<SpMVAccVecRow> spmv_acc_vec_row;
  CsrSpMV<SpMVAccWfRow> spmv_acc_wf_row;
  CsrSpMV<SpMVAccLineEnhance> spmv_acc_line_enhance;

  spmv_acc_default.test(mtx_path, "spmv-acc-default", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);
  spmv_acc_adaptive.test(mtx_path, "spmv-acc-adaptive", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);
  spmv_acc_block_row.test(mtx_path, "spmv-acc-block-row", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x,
                          dev_y);
  spmv_acc_flat.test(mtx_path, "spmv-acc-flat", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);
  spmv_acc_light.test(mtx_path, "spmv-acc-light", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);
  // todo: catch error of line strategy
  // spmv_acc_line.test(mtx_path, operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);
  spmv_acc_thread_row.test(mtx_path, "spmv-acc-thread-row", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x,
                           dev_y);
  spmv_acc_vec_row.test(mtx_path, "spmv-acc-vector-row", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);
  spmv_acc_wf_row.test(mtx_path, "spmv-acc-wavefront-row", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x,
                       dev_y);
  spmv_acc_line_enhance.test(mtx_path, "spmv-acc-line-enhance", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x,
                             dev_y);

#ifdef __HIP_PLATFORM_HCC__
  // rocsparse
  CsrSpMV<RocSparseVecRow> rocsparse_vec_row;
  rocsparse_vec_row.test(mtx_path, "rocSparse-vector-row", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x,
                         dev_y);

  CsrSpMV<RocSparseAdaptive> rocsparse_adaptive;
  rocsparse_adaptive.test(mtx_path, "rocSparse-adaptive", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x,
                          dev_y);
  // hola
  CsrSpMV<HolaHipSpMV> hola_hip_spmv;
  hola_hip_spmv.test(mtx_path, "hip-hola", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);

#endif

#ifndef __HIP_PLATFORM_HCC__
  // cusparse
  CsrSpMV<CuSparseGeneral> cusparse_general;
  cusparse_general.test(mtx_path, "cuSparse", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);

  // cub
  CsrSpMV<CubDeviceSpMV> cub_device_spmv;
  cub_device_spmv.test(mtx_path, "cub", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);

  // hola
  CsrSpMV<HolaSpMV> hola_spmv;
  hola_spmv.test(mtx_path, "hola", operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);
#endif

  destroy_device_data(d_csr, dev_x, dev_y);
}
