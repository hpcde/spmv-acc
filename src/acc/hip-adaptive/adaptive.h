//
// Created by genshen on 2021/7/28.
//

#ifndef SPMV_ACC_ADAPTIVE_H
#define SPMV_ACC_ADAPTIVE_H

void adaptive_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                          const int *colindex, const double *value, const double *x, double *y);

#endif // SPMV_ACC_ADAPTIVE_H
