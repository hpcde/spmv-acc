//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_CSR_H
#define SPMV_ACC_CSR_H

typedef double dtype;

template <typename I, typename T> class csr_mtx {
public:
  I rows = 0;
  I cols = 0;
  I nnz = 0;

  I *row_ptr = nullptr;
  I *col_index = nullptr;
  T *values = nullptr;
};

typedef csr_mtx<int, dtype> type_csr;

#endif // SPMV_ACC_CSR_H
