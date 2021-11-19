#include "include/dCSR.h"
#include "include/dVector.h"
#include "include/holaspmv.h"

#include "api/types.h"

void spmv(int trans, int rows, int cols, int nnz, const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
  // dCSR<double> d_csr(rows, cols, nnz, const_cast<double *>(d_csr_desc.values), static_cast<unsigned int
  // *>(static_cast<void *>((const_cast<int *>(d_csr_desc.row_ptr)))),
  //                   static_cast<unsigned int *>(static_cast<void *>((const_cast<int *>(d_csr_desc.col_index)))));
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
  // get tempmem size first
  hola_spmv(nullptr, temp_size, d_y, d_csr, d_x, HolaMode::Default, false, false);
  // allocate the temp memory.
  void *d_hola_temp;
  cudaMalloc(&d_hola_temp, temp_size);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
  }
  // use the temp memory for calculation.
  constexpr int padding = 0;
  hola_spmv(d_hola_temp, temp_size, d_y, d_csr, d_x, HolaMode::Default, false, padding >= 512 ? true : false);
}