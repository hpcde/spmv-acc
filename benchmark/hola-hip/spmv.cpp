#include <iostream>
#include <stdexcept>

#include <api/types.h>

#include "d_csr.h"
#include "hola_spmv.h"
#include "hola_vector.h"

#include "../utils/benchmark_time.h"
#include "../utils/timer_utils.h"

void spmv(int trans, int rows, int cols, int nnz, const csr_desc<int, double> d_csr_desc, const double *x, double *y,
          BenchmarkTime *bmt) {
  dCSR<double> d_csr(rows, cols, nnz, const_cast<double *>(d_csr_desc.values),
                     static_cast<unsigned int *>(static_cast<void *>((const_cast<int *>(d_csr_desc.row_ptr)))),
                     static_cast<unsigned int *>(static_cast<void *>((const_cast<int *>(d_csr_desc.col_index)))));
  dDenseVector<double> d_y(rows, const_cast<double *>(y));
  dDenseVector<double> d_x(rows, const_cast<double *>(x));

  size_t temp_size = 0;
  hip::timer::event_timer pre_timer, calc_timer, destroy_timer;
  pre_timer.start();
  // get tempmem size first
  try {
    hola_spmv(nullptr, temp_size, d_y, d_csr, d_x, HolaMode::Default, false, false);
  } catch (const std::runtime_error &error) {
    throw error;
  }
  // allocate the temp memory.
  void *d_hola_temp;
  hipMalloc(&d_hola_temp, temp_size);
  if (hipDeviceSynchronize() != hipSuccess) {
    hipFree(d_hola_temp);
    throw std::runtime_error(hipGetErrorString(hipGetLastError()));
  }
  // use the temp memory for calculation.
  constexpr int padding = 0;
  pre_timer.stop();
  calc_timer.start();
  try {
    hola_spmv(d_hola_temp, temp_size, d_y, d_csr, d_x, HolaMode::Default, false, padding >= 512 ? true : false);
  } catch (const std::runtime_error &error) {
    hipFree(d_hola_temp);
    throw error;
  }
  hipDeviceSynchronize();
  calc_timer.stop();
  destroy_timer.start();
  hipFree(d_hola_temp);
  destroy_timer.stop();
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
  }
}
