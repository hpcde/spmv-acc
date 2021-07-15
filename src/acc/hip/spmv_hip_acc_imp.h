//
// Created by genshen on 2021/4/15.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_DEFAULT_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_DEFAULT_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

__global__ void default_device_sparse_spmv_acc(int trans, const int alpha, const int beta, int m, int n,
                                               const int *rowptr, const int *colindex, const double *value,
                                               const double *x, double *y);

void default_sparse_spmv(int htrans, const int halpha, const int hbeta, int hm, int hn, const int *hrowptr,
                         const int *hcolindex, const double *hvalue, const double *hx, double *hy);

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_DEFAULT_H
