//
// Created by genshen on 2021/06/30.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr, const int *colindex,
                 const double *value, const double *x, double *y);

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H