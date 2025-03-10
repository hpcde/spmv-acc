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
#include "benchmark_merge_path.hpp"
#include "clipp.h"

#include "csr_binary_reader.hpp"
#include "csr_mtx_reader.hpp"
#include "matrix_market_reader.hpp"
#include "sparse_format.h"
#include "utils.hpp"
#include "utils/statistics_logger.h"

#include "benchmark_config.h"
#include "csr_spmv.hpp"

void test_spmv(std::string mtx_path, type_csr h_csr, host_vectors<dtype> h_vectors);

int main(int argc, char **argv) {
  std::string mtx_path = "", fmt = "csr";

  auto cli = (clipp::value("input file", mtx_path),
              clipp::option("-f", "--format")
                      .doc("input matrix format, can be `csr` (default), `mm` (matrix market) or "
                           "`bin` (csr binary)") &
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
  } else if (fmt == "bin") {
    csr_binary_reader<int32_t, dtype> csr_bin_reader;
    csr_bin_reader.load_mat(mtx_path);
    csr_bin_reader.close_stream();

    h_csr.rows = csr_bin_reader.rows();
    h_csr.cols = csr_bin_reader.cols();
    h_csr.nnz = csr_bin_reader.nnz();
    // just reuse memory in file reading step.
    csr_bin_reader.as_raw_ptr(h_csr.values, h_csr.col_index, h_csr.row_ptr);

    create_host_data(h_csr, h_vectors, true);
    test_spmv(mtx_path, h_csr, h_vectors);
    destroy_host_data(h_vectors);
  } else {
    matrix_market_reader<int, dtype> mm_reader;
    matrix_market<int, dtype> mm = mm_reader.load_mat(mtx_path);
    h_csr = mm.to_csr();

    create_host_data(h_csr, h_vectors, true);
    test_spmv(mtx_path, h_csr, h_vectors);
    destroy_host_data(h_vectors);
  }
}

#define SPMV_BENCHMARK(instance, name, flag)                                                                           \
  {                                                                                                                    \
    instance ins;                                                                                                      \
    if (flag) {                                                                                                        \
      ins.test(mtx_path, name, operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);                         \
    }                                                                                                                  \
  }

void test_spmv(std::string mtx_path, type_csr h_csr, host_vectors<dtype> h_vectors) {
  HIP_CHECK(hipSetDevice(0));
  dtype *dev_x, *dev_y;
  type_csr d_csr = create_device_data(h_csr, h_vectors.hX, h_vectors.temphY, dev_x, dev_y);
  // set parameters
  enum sparse_operation operation = operation_none;
  dtype alpha = 1.0;
  dtype beta = 1.0;

  statistics::print_statistics_header();

  // spmv-acc
  // SPMV_BENCHMARK(SpMVAccDefault, "spmv-acc-default", ENABLE_SPMV_ACC_DEFAULT);
  SPMV_BENCHMARK(SpMVAccAdaptive, "spmv-acc-adaptive", ENABLE_SPMV_ACC_ADAPTIVE);
  SPMV_BENCHMARK(SpMVAccBlockRow, "spmv-acc-block-row", ENABLE_SPMV_ACC_BLOCK_ROW);
  SPMV_BENCHMARK(SpMVAccFlat, "spmv-acc-flat", ENABLE_SPMV_ACC_FLAT);
  SPMV_BENCHMARK(SpMVAccFlatV2, "spmv-acc-flat-v2", ENABLE_SPMV_ACC_FLAT);
  SPMV_BENCHMARK(SpMVAccLight, "spmv-acc-light", ENABLE_SPMV_ACC_LIGHT);
  SPMV_BENCHMARK(SpMVAccLine, "spmv-acc-line", ENABLE_SPMV_ACC_LINE);
  SPMV_BENCHMARK(SpMVAccThreadRow, "spmv-acc-thread-row", ENABLE_SPMV_ACC_THREAD_ROW);
  SPMV_BENCHMARK(SpMVAccVecRow, "spmv-acc-vector-row", ENABLE_SPMV_ACC_VECTOR_ROW);
  SPMV_BENCHMARK(SpMVAccWfRow, "spmv-acc-wavefront-row", ENABLE_SPMV_ACC_WF_ROW);
  SPMV_BENCHMARK(SpMVAccLineEnhance, "spmv-acc-line-enhance", ENABLE_SPMV_ACC_LE_ROW);
  SPMV_BENCHMARK(SpMVAccAdaptivePlus, "spmv-acc-adaptive-plus", ENABLE_SPMV_ACC_ADAPTIVE_PLUS);
  SPMV_BENCHMARK(SpMVAccFlatSegSum, "spmv-acc-flat-seg-sum", ENABLE_SPMV_ACC_FLAT_SEG_SUM);

#ifdef __HIP_PLATFORM_HCC__
  // rocsparse
  SPMV_BENCHMARK(RocSparseVecRow, "rocSparse-vector-row", ENABLE_ROC_VECTOR_ROW);
  SPMV_BENCHMARK(RocSparseAdaptive, "rocSparse-adaptive", ENABLE_ROC_ADAPTIVE);
  // hola
  SPMV_BENCHMARK(HolaHipSpMV, "hip-hola", ENABLE_HIP_HOLA);
#endif

#ifndef __HIP_PLATFORM_HCC__
  // cusparse
  SPMV_BENCHMARK(CuSparseGeneral, "cuSparse", ENABLE_CU_SPARSE);
  // cub
  SPMV_BENCHMARK(CubDeviceSpMV, "cub", ENABLE_CUB);
  // hola
  SPMV_BENCHMARK(HolaSpMV, "hola", ENABLE_HOLA);
  // merge path
  SPMV_BENCHMARK(MergePathSingleBlockUpdateSpMV, "merge-path-single-block-update", ENABLE_MERGE_PATH)
  SPMV_BENCHMARK(MergePathLookBackUpdateSpMV, "merge-path-look-back-update", ENABLE_MERGE_PATH)
  // ACSR
  SPMV_BENCHMARK(ACSRSpMV, "acsr", ENABLE_ACSR);
#endif

  destroy_device_data(d_csr, dev_x, dev_y);
}
