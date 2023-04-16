#!/bin/bash
module purge
module load apps/cuda/10.2
module load compiler/gnu/7.4.0
export CC=gcc
export CXX=${PWD}/scripts/hipcc-nv-wrapper.sh

source /public/share/genshen/local/rocm/4.3.1/load-hip.sh

cmake -B./cmake-build-release/ -S./ \
   -DSPMV_OMP_ENABLED_FLAG=ON \
   -DBENCHMARK_CUDA_ENABLE_FLAG=ON \
   -DSPMV_BUILD_BENCHMARK=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DKERNEL_STRATEGY=Adaptive \
   -DWAVEFRONT_SIZE=32 \
   -DHIP_HIPCC_FLAGS="-std=c++14" \
   -DHIP_NVCC_FLAGS="-arch=sm_70 -rdc=true" \
   -DCMAKE_CXX_FLAGS="-std=c++14"