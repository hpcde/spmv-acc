#include <iostream>
#include <type_traits>

#include "hola/spmv.h"

struct HolaSpMV : CsrSpMV<HolaSpMV> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    spmv(trans, h_csr_desc.rows, h_csr_desc.cols, h_csr_desc.nnz, d_csr_desc, x, y);
  }
};