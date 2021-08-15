//
// Created by genshen on 2021/7/13.
//

#ifndef SPMV_ACC_THREAD_ROW_CONFIG_H
#define SPMV_ACC_THREAD_ROW_CONFIG_H

#define THREAD_ROW_GLOBAL_LOAD_X2 // if defined, we load 2 double or 2 int in each loop.

// remapping memory accessing mode of vector x.
// In most cases, the matrix is a diagonal matrix.
// Thus, it is better to access vector x with column first mode.
// If the macro below is enabled, the memory accessing mode of vector x is column-first,
// otherwise, the mode is row-first.
// #define OPT_THREAD_ROW_REMAP_VEC_X

#define OPT_THREAD_ROW_BLOCK_LOAD_X2

#endif // SPMV_ACC_THREAD_ROW_CONFIG_H
