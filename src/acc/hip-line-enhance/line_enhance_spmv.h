//
// Created by chu genshen on 2021/10/2.
//

#ifndef SPMV_ACC_LINE_ENHANCE_SPMV_H
#define SPMV_ACC_LINE_ENHANCE_SPMV_H

void line_enhance_sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *rowptr,
                              const int *colindex, const double *value, const double *x, double *y);

#endif // SPMV_ACC_LINE_ENHANCE_SPMV_H
