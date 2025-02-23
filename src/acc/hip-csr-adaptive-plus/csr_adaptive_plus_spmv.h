//
// Created by genshen on 2024/12/31.
//

#ifndef SPMV_ACC_CSR_ADAPTIVE2_SPMV_H
#define SPMV_ACC_CSR_ADAPTIVE2_SPMV_H

void csr_adaptive_plus_sparse_spmv(int trans, const double alpha, const double beta,
                                   const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc,
                                   const double *x, double *y);

#endif // SPMV_ACC_CSR_ADAPTIVE2_SPMV_H
