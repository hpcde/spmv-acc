#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_debug.cuh>

#include "api/types.h"

void spmv(int trans, const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc, const double *x,
          double *y) {
  double *d_values = const_cast<double *>(d_csr_desc.values);
  int *d_row_ptr = const_cast<int *>(d_csr_desc.row_ptr);
  int *d_col_index = const_cast<int *>(d_csr_desc.col_index);
  double *d_x = const_cast<double *>(x);
  double *d_y = const_cast<double *>(y);
  int rows = h_csr_desc.rows;
  int cols = h_csr_desc.cols;
  int nnz = h_csr_desc.nnz;
  // Caching allocator for device memory
  cub::CachingDeviceAllocator g_allocator(true);
  void *d_buffer = NULL;
  size_t d_buffer_size = 0;
  // Get buffer size
  CubDebugExit(cub::DeviceSpmv::CsrMV<double>(d_buffer, d_buffer_size, d_values, d_row_ptr, d_col_index, d_x, d_y, rows,
                                              cols, nnz, (cudaStream_t)0, false));
  // Allocate an external buffer
  CubDebugExit(g_allocator.DeviceAllocate(&d_buffer, d_buffer_size));
  // Execute SpMV
  CubDebugExit(cub::DeviceSpmv::CsrMV<double>(d_buffer, d_buffer_size, d_values, d_row_ptr, d_col_index, d_x, d_y, rows,
                                              cols, nnz, (cudaStream_t)0, false));
}