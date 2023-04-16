# enable to use hip to accelerate on GPU side, currently, it must be `ON`.
option(HIP_ENABLE_FLAG "Enable HIP" ON)
option(SPMV_BUILD_TOOLS "build tools" OFF)
option(SPMV_BUILD_BENCHMARK "build benchmark" OFF)
option(SPMV_OMP_ENABLED_FLAG "Enable OpenMP to parse input file" OFF)

# default: perform result verification on CPU side.
# However, if it is set to `ON`, it will use device side verification.
option(DEVICE_SIDE_VERIFY_FLAG "Verify result of device side" OFF)
option(BENCHMARK_CUDA_ENABLE_FLAG "Enable performance benchmark test on CUDA device" OFF)
option(BENCHMARK_FORCE_SYNC_KERNELS "Add `hipDeviceSynchronize` between kernelss while benchmarking for getting real kernel time" OFF)
set(AVAILABLE_CU "60" CACHE STRING "available Compute Units per GPU")
# threads number in a wavefront (which is usually 64 for AMD GPU and 32 for NVIDIA GPU)
set(WAVEFRONT_SIZE "64" CACHE STRING "threads number in a wavefront")
set(KERNEL_STRATEGY "DEFAULT" CACHE STRING "SpMV strategy")
# options are:
# - DEFAULT: default strategy. The source files are saved in 'acc/hip'.
# - ADAPTIVE: mixed adaptive method. The source files are saved in 'acc/hip-adaptive'.
# - THREAD_ROW: each thread process one row. The source files are saved in 'acc/hip-thread-row'
# - BLOCK_ROW_ORDINARY: each block process one row with ordinary method. The source files are saved in 'acc/hip-block-row-ordinary'
# - WF_ROW: each wavefront process one row. The source files are saved in 'acc/hip-wf-row'
# - LIGHT: each vector process one row using method described in https://doi.org/10.1007/s11265-016-1216-4.
#         The source files are saved in 'acc/hip-light'
# - VECTOR_ROW: each vector process one row.  The source files are saved in 'acc/hip-vector-row'.
# - LINE: Calculate and save results to LDS, then reduce (the CSR-Stream method in https://doi.org/10.1109/SC.2014.68. )
#         The source files are saved in 'acc/hip-line'.
# - FLAT: flat algorithm, each Block calculate the same number of elements, and then perform reduction.
#         The source files are saved in 'acc/hip-flat'.

set(WF_REDUCE "DEFAULT" CACHE STRING "reduce strategy of wavefront row kernel strategy")
# options are:
#  - DEFAULT: the same method used in rocSparse
#  - LDS: USE shared memory of LDS for reducing
#  - REG: use register and `__shfl_down` for reducing

# check CU number
if (NOT AVAILABLE_CU MATCHES "^[0-9]+$")
    MESSAGE(FATAL_ERROR "`AVAILABLE_CU` must be a number.")
endif ()

# check WF_REDUCE
string(TOLOWER ${WF_REDUCE} WF_REDUCE_LOWER)
if ((WF_REDUCE_LOWER MATCHES "default") OR (WF_REDUCE_LOWER MATCHES "lds")
        OR (WF_REDUCE_LOWER MATCHES "reg"))
    MESSAGE(STATUS "current wavefront reduction strategy is: ${WF_REDUCE}")
else ()
    MESSAGE(FATAL_ERROR "unsupported wavefront reduction strategy ${WF_REDUCE}")
endif ()

# flat reduce strategy
option(FLAT_SEGMENT_SUM_REDUCE "flat reducting strategy is segement sum" OFF)

if (HIP_ENABLE_FLAG)
    set(SPMV_BIN_NAME spmv-hip)
    set(SPMV_CLI_LIB sparse_cli)
    set(SPMV_KERNEL_LIB_NAME spmv-acc-kernels)
    # add linked libs
    set(ACC_LIBS ${ACC_LIBS} ${SPMV_KERNEL_LIB_NAME})
else ()
    set(SPMV_BIN_NAME spmv-cpu)
endif ()

# add lib rocsparse if using device verification.
if (DEVICE_SIDE_VERIFY_FLAG)
    set(ACC_LIBS ${ACC_LIBS} rocsparse)
endif ()

if (DEVICE_SIDE_VERIFY_FLAG AND (NOT HIP_ENABLE_FLAG))
    message(FATAL_ERROR "To perform device side verification, we must enable `HIP_ENABLE_FLAG`")
endif ()
