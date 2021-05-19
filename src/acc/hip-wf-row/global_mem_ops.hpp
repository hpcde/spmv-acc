//
// Created by genshen on 2021/5/7.
//

#ifndef SPMV_ACC_GLOBAL_MEM_OPS_H
#define SPMV_ACC_GLOBAL_MEM_OPS_H

typedef struct {
  double a;
  double b;
} dbl_x2;

/**
 * load data from memory with the address specified by @param ptr to variable @param val
 * @tparam offset
 * @param ptr target address
 * @param val the receive variable
 * @return
 */
__device__ __forceinline__ void global_load(const void *ptr, dbl_x2 &val) {
  asm volatile("global_load_dwordx4 %0, %1, off " : "=v"(val) : "v"(ptr));
}

#endif // SPMV_ACC_GLOBAL_MEM_OPS_H
