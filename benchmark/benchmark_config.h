//
// Created by genshen on 2022/4/12.
//

#ifndef SPMV_ACC_BENCHMARK_CONFIG_H
#define SPMV_ACC_BENCHMARK_CONFIG_H

// spmv-acc
constexpr bool ENABLE_SPMV_ACC_DEFAULT = false;
constexpr bool ENABLE_SPMV_ACC_ADAPTIVE = true;
constexpr bool ENABLE_SPMV_ACC_BLOCK_ROW = true;
constexpr bool ENABLE_SPMV_ACC_FLAT = true;
constexpr bool ENABLE_SPMV_ACC_LIGHT = true;
constexpr bool ENABLE_SPMV_ACC_LINE = true;
constexpr bool ENABLE_SPMV_ACC_THREAD_ROW = false;
constexpr bool ENABLE_SPMV_ACC_VECTOR_ROW = true;
constexpr bool ENABLE_SPMV_ACC_WF_ROW = true;
constexpr bool ENABLE_SPMV_ACC_LE_ROW = true;
constexpr bool ENABLE_SPMV_ACC_FLAT_SEG_SUM = true;

// rocm
constexpr bool ENABLE_ROC_VECTOR_ROW = true;
constexpr bool ENABLE_ROC_ADAPTIVE = true;
constexpr bool ENABLE_HIP_HOLA = true;

// cuda
constexpr bool ENABLE_CU_SPARSE = true;
constexpr bool ENABLE_CUB = true;
constexpr bool ENABLE_HOLA = true;

#endif // SPMV_ACC_BENCHMARK_CONFIG_H
