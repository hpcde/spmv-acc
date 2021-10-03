//
// Created by chu genshen on 2021/10/2.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "building_config.h"

template <int WF_SIZE, int ROWS_PER_BLOCK, int R, int THREADS, typename I, typename T>
__global__ void line_enhance_kernel(int m, const T alpha, const T beta, const I *__restrict__ row_offset,
                                    const I *__restrict__ csr_col_ind, const T *__restrict__ csr_val,
                                    const T *__restrict__ x, T *__restrict__ y) {
}

#define LINE_ENHANCE_KERNEL_WRAPPER(ROWS_PER_BLOCK, R, BLOCKS, THREADS)                                                \
  (line_enhance_kernel<__WF_SIZE__, ROWS_PER_BLOCK, R, THREADS, int, double>)<<<BLOCKS, THREADS>>>(                    \
      m, alpha, beta, rowptr, colindex, value, x, y)
