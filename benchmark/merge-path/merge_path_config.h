#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_CONFIG_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_CONFIG_H

// reduction algorithm:
// 1. linear search -> REDUCE_LINEAR_SEARCH
// 2. binary serach -> REDUCE_BINARY_SEARCH

// #define REDUCE_LINEAR_SEARCH
#define REDUCE_BINARY_SEARCH

// update algorithm:
// 1. single block -> UPDATE_SINGLE_BLOCK
// 2. look back    -> UPDATE_LOOK_BACK

// #define UPDATE_SINGLE_BLOCK
#define UPDATE_LOOK_BACK

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_CONFIG_H