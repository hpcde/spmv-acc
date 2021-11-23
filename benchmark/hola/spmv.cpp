#include <iostream>
#include <stdexcept>

#include "include/dCSR.h"
#include "include/dVector.h"
#include "include/holaspmv.h"

#include "../utils/benchmark_time.h"
#include "api/types.h"

void spmv(int trans, int rows, int cols, int nnz, const csr_desc<int, double> d_csr_desc, const double *x, double *y,
          BenchmarkTime *bmt) {
  dCSR<double> d_csr;
  d_csr.rows = rows;
  d_csr.cols = cols;
  d_csr.nnz = nnz;
  d_csr.data = const_cast<double *>(d_csr_desc.values);
  d_csr.row_offsets = static_cast<unsigned int *>(static_cast<void *>((const_cast<int *>(d_csr_desc.row_ptr))));
  d_csr.col_ids = static_cast<unsigned int *>(static_cast<void *>((const_cast<int *>(d_csr_desc.col_index))));

  dDenseVector<double> d_y;
  d_y.size = rows;
  d_y.data = const_cast<double *>(y);

  dDenseVector<double> d_x;
  d_x.size = rows;
  d_x.data = const_cast<double *>(x);
  size_t temp_size = 0;
  my_timer pre_timer, calc_timer, destroy_timer;
  pre_timer.start();
  // get tempmem size first
  try {
    hola_spmv(nullptr, temp_size, d_y, d_csr, d_x, HolaMode::Default, false, false);
  } catch (const std::runtime_error &error) {
    throw error;
  }
  // allocate the temp memory.
  void *d_hola_temp;
  cudaMalloc(&d_hola_temp, temp_size);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cudaFree(d_hola_temp);
    throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
  }
  // use the temp memory for calculation.
  constexpr int padding = 0;
  pre_timer.stop();
  calc_timer.start();
  try {
    hola_spmv(d_hola_temp, temp_size, d_y, d_csr, d_x, HolaMode::Default, false, padding >= 512 ? true : false);
  } catch (const std::runtime_error &error) {
    cudaFree(d_hola_temp);
    throw error;
  }
  cudaDeviceSynchronize();
  calc_timer.stop();
  destroy_timer.start();
  cudaFree(d_hola_temp);
  destroy_timer.stop();
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, destroy_timer.time_use);
  }
}
