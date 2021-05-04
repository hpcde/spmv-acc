//
// Created by genshen on 2021/5/4.
//

#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.

#include "building_config.h"
#include "wavefront_row_default.hpp"
#include "wavefront_row_lds.hpp"
#include "wavefront_row_reg.hpp"

typedef int type_index;
typedef double type_values;

void sparse_spmv(int htrans, const int halpha, const int hbeta, int hm, int hn, const int *hrowptr,
                 const int *hcolindex, const double *hvalue, const double *hx, double *hy) {
#if defined WF_REDUCE_DEFAULT
  (device_spmv_wf_row_default<1024, 64, int, int, double>)<<<1, 1024>>>(hm, halpha, hbeta, hrowptr, hcolindex, hvalue,
      hx, hy);
  // or:
  //  hipLaunchKernelGGL((device_spmv_wf_row_default<256, 64, int, int, double>), 64, 256, 0, 0, hm, halpha, hbeta,
  //     hrowptr, hcolindex, hvalue, hx, hy);
#elif defined WF_REDUCE_LDS
  (device_spmv_wf_row_lds<256, 64>)<<<64, 256>>>(htrans, halpha, hbeta, hm, hn, hrowptr, hcolindex, hvalue, hx, hy);
#elif defined WF_REDUCE_REG
  (device_spmv_wf_row_reg<256, 64>)<<<64, 256>>>(htrans, halpha, hbeta, hm, hn, hrowptr, hcolindex, hvalue, hx, hy);
#endif
}
