//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_VERIFICATION_H
#define SPMV_ACC_VERIFICATION_H

#include "building_config.h"
#include "sparse_format.h"

void verify(double *dy, double *hy, int n);

void host_spmv(dtype alpha, dtype beta, const dtype *value, const int *rowptr, const int *colindex, int m, int n, int a,
               const dtype *x, dtype *y);

void host_spmv(const dtype *value, const int *rowptr, const int *colindex, int m, int n, int a, const dtype *x,
               dtype *y);

#ifdef gpu
void rocsparse(type_csr d_csr, dtype *dev_x, dtype *dev_y, dtype &alpha, dtype &beta);
#endif

#endif // SPMV_ACC_VERIFICATION_H
