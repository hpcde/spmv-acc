//
// Created by genshen on 2021/11/16.
//

#include <cmath>
#include <iostream>

#ifdef gpu
#include <hipsparse.h>
#include <rocsparse.h>
#endif

#include "verification.h"

void verify(double *dy, double *hy, int n) {
  int total_validation = 0;
  for (int i = 0; i < n; i++) {
    if (std::fabs(dy[i] - hy[i]) / std::fabs(hy[i]) >= 1e-7) {
      std::cout << std::fabs(dy[i] - hy[i]) << " i:" << i << " dy[i]:" << dy[i] << " hy[i]:" << hy[i] << std::endl;
      std::cout << "Failed verification,please check your code\n" << std::endl;
      return;
    }
    total_validation = i;
  }
  std::cout << "Congratulation, pass " << total_validation + 1 << " validation!\n" << std::endl;
}

void host_spmv(dtype alpha, dtype beta, const dtype *value, const int *rowptr, const int *colindex, int m, int n, int a,
               const dtype *x, dtype *y) {
  // calculate the matrix-vector multiply where matrix is stored in the form of CSR
  for (int i = 0; i < m; i++) {
    dtype y0 = 0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
      y0 += value[j] * x[colindex[j]];
    }
    y[i] = alpha * y0 + beta * y[i];
  }
}

void host_spmv(const dtype *value, const int *rowptr, const int *colindex, int m, int n, int a, const dtype *x,
               dtype *y) {
  // calculate the matrix-vector multiply where matrix is stored in the form of CSR
  for (int i = 0; i < m; i++) {
    dtype y0 = 0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
      y0 += value[j] * x[colindex[j]];
    }
    y[i] = y0;
  }
}

#ifdef gpu
void rocsparse(type_csr d_csr, dtype *dev_x, dtype *dev_y, dtype &alpha, dtype &beta) {
  const int A_num_rows = d_csr.rows;
  const int A_num_cols = d_csr.cols;
  const int A_nnz = d_csr.nnz;

  int *dA_csrOffsets = d_csr.row_ptr;
  int *dA_columns = d_csr.col_index;
  dtype *dA_values = d_csr.values;
  dtype *dX = dev_x;
  dtype *dY = dev_y;

  rocsparse_handle handle = nullptr;
  rocsparse_spmat_descr matA;
  rocsparse_dnvec_descr vecX, vecY;
  void *dBuffer = nullptr;
  size_t bufferSize = 0;

  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
  // Create sparse matrix A in CSR format
  ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA, A_num_rows, A_num_cols, A_nnz, dA_csrOffsets, dA_columns, dA_values,
                                             rocsparse_indextype_i32, rocsparse_indextype_i32,
                                             rocsparse_index_base_zero, rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecX, A_num_cols, dX, rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecY, A_num_rows, dY, rocsparse_datatype_f64_r));
  // allocate an external buffer if needed

  ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matA, vecX, &beta, vecY,
                                 rocsparse_datatype_f64_r, rocsparse_spmv_alg_default, &bufferSize, nullptr));
  HIP_CHECK(hipMalloc(&dBuffer, bufferSize));
  ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none, &alpha, matA, vecX, &beta, vecY,
                                 rocsparse_datatype_f64_r, rocsparse_spmv_alg_default, &bufferSize, dBuffer));
}
#endif
