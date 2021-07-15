//
// Created by dly on 2021/4/26.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_BLOCK_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_BLOCK_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

__global__ void block_row_device_sparse_spmv_acc(int trans, const int alpha, const int beta, int m, int n,
                                                 const int *rowptr, const int *colindex, const double *value,
                                                 const double *x, double *y);

void block_row_sparse_spmv(int htrans, const int halpha, const int hbeta, int hm, int hn, const int *hrowptr,
                           const int *hcolindex, const double *hvalue, const double *hx, double *hy);

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_BLOCK_H
