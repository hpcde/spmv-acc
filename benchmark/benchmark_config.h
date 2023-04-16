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
constexpr bool ENABLE_SPMV_ACC_BLOCK_ROW = false;
constexpr bool ENABLE_SPMV_ACC_FLAT = true;
constexpr bool ENABLE_SPMV_ACC_LIGHT = true;
constexpr bool ENABLE_SPMV_ACC_LINE = true;
constexpr bool ENABLE_SPMV_ACC_THREAD_ROW = false;
constexpr bool ENABLE_SPMV_ACC_VECTOR_ROW = true;
constexpr bool ENABLE_SPMV_ACC_WF_ROW = true;
constexpr bool ENABLE_SPMV_ACC_LE_ROW = true;

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

#ifdef MACRO_BENCHMARK_FORCE_KERNEL_SYNC
constexpr bool BENCHMARK_FORCE_KERNEL_SYNC = true;
#endif
#ifndef MACRO_BENCHMARK_FORCE_KERNEL_SYNC
constexpr bool BENCHMARK_FORCE_KERNEL_SYNC = false;
#endif

/// @brief lazy device sync will try to not sync device unless user the sync flag (from global config or local config) has been specificed.
/// @param sync_flag the local sync flag.
inline void lazy_device_sync(bool sync_flag = false) {
  if (sync_flag || BENCHMARK_FORCE_KERNEL_SYNC) {
    hipDeviceSynchronize();
  }
}

#endif // SPMV_ACC_BENCHMARK_CONFIG_H
