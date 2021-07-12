//
// Created by dly on 2021/4/19.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_H

void sparse_spmv(int trans, const int alpha, const int beta, int m, int n, const int *d_row_ptr,
                 const int *d_csr_col_index, const double *d_csr_value, const double *d_x, double *d_y);

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_H
