#define CUB_STDERR

#include <iostream>
#include <type_traits>

#include <cub/spmv.h>

struct CubDeviceSpMV : CsrSpMV<CubDeviceSpMV> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    spmv(trans, h_csr_desc, d_csr_desc, x, y);
  }
};