//
// Created by genshen on 2021/6/29.
//

#ifndef SPMV_ACC_GLOBAL_DATA_H
#define SPMV_ACC_GLOBAL_DATA_H

//新demo程序从外部读取csr数据，使用vector装载程序
std::vector<double> csr_data;
std::vector<int> csr_indices;
std::vector<int> csr_indptr;
std::vector<double> dense_vector;

#endif // SPMV_ACC_GLOBAL_DATA_H
