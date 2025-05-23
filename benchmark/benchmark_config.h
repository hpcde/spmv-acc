//
// Created by genshen on 2022/4/12.
//

#ifndef SPMV_ACC_BENCHMARK_CONFIG_H
#define SPMV_ACC_BENCHMARK_CONFIG_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

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
constexpr bool ENABLE_SPMV_ACC_ADAPTIVE_PLUS = true;
constexpr bool ENABLE_SPMV_ACC_FLAT_SEG_SUM = true;

// rocm
constexpr bool ENABLE_ROC_VECTOR_ROW = true;
constexpr bool ENABLE_ROC_ADAPTIVE = true;
constexpr bool ENABLE_HIP_HOLA = true;

// cuda
constexpr bool ENABLE_CU_SPARSE = true;
constexpr bool ENABLE_CUB = true;
constexpr bool ENABLE_HOLA = true;
constexpr bool ENABLE_MERGE_PATH = true;
constexpr bool ENABLE_ACSR = true;

// FLAT method config
constexpr int FLAT_PRE_CALC_BP_KERNEL_VERSION_V1 = 1;
constexpr int FLAT_PRE_CALC_BP_KERNEL_VERSION_V2 = 2;
constexpr int FLAT_PRE_CALC_BP_KERNEL_VERSION = FLAT_PRE_CALC_BP_KERNEL_VERSION_V1;

#ifdef MACRO_BENCHMARK_FORCE_KERNEL_SYNC
constexpr bool BENCHMARK_FORCE_KERNEL_SYNC = true;
#endif
#ifndef MACRO_BENCHMARK_FORCE_KERNEL_SYNC
constexpr bool BENCHMARK_FORCE_KERNEL_SYNC = false;
#endif

#endif // SPMV_ACC_BENCHMARK_CONFIG_H
