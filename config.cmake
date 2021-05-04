# enable to use hip to accelerate on GPU side, currently, it must be `ON`.
option(HIP_ENABLE_FLAG "Enable HIP" ON)

# default: perform result verification on CPU side.
# However, if it is set to `ON`, it will use device side verification.
option(DEVICE_SIDE_VERIFY_FLAG "Verify result of device side" OFF)
set(KERNEL_STRATEGY "DEFAULT" CACHE STRING "SpMV strategy")
# options are:
# - DEFAULT: default strategy. The source files are saved in 'acc/hip'.
# - THREAD_ROW: each thread process one row. The source files are saved in 'acc/hip-thread-row'
# - BLOCK_ROW_ORDINARY: each block process one row with ordinary method. The source files are saved in 'acc/hip-block-row-ordinary'
# - WF_ROW: each wavefront process one row. The source files are saved in 'acc/hip-wf-row'

set(WF_REDUCE "DEFAULT" CACHE STRING "reduce strategy of wavefront row kernel strategy")
# options are:
#  - DEFAULT: the same method used in rocSparse
#  - LDS: USE shared memory of LDS for reducing
#  - REG: use register and `__shfl_down` for reducing

# check strategies
string(TOLOWER ${KERNEL_STRATEGY} KERNEL_STRATEGY_LOWER)
if ((KERNEL_STRATEGY_LOWER MATCHES "default") OR (KERNEL_STRATEGY_LOWER MATCHES "thread_row")
        OR (KERNEL_STRATEGY_LOWER MATCHES "wf_row"))
    MESSAGE(STATUS "current kernel strategy is: ${KERNEL_STRATEGY}")
elseif (KERNEL_STRATEGY_LOWER MATCHES "block_row_ordinary")
    MESSAGE(STATUS "current kernel strategy is: ${KERNEL_STRATEGY}")
else ()
    MESSAGE(FATAL_ERROR "unsupported kernel strategy ${KERNEL_STRATEGY}")
endif ()

# set directory postfix for strategy source files.
if (KERNEL_STRATEGY_LOWER MATCHES "default")
    set(KERNEL_STRATEGY_SRC_DIR_POSTFIX "")
else ()
    string(REPLACE "_" "-" KERNEL_STRATEGY_SRC_DIR_POSTFIX "-${KERNEL_STRATEGY_LOWER}")
endif ()

# check WF_REDUCE
string(TOLOWER ${WF_REDUCE} WF_REDUCE_LOWER)
if ((WF_REDUCE_LOWER MATCHES "default") OR (WF_REDUCE_LOWER MATCHES "lds")
        OR (WF_REDUCE_LOWER MATCHES "reg"))
    MESSAGE(STATUS "current wavefront reduction strategy is: ${WF_REDUCE}")
else()
    MESSAGE(FATAL_ERROR "unsupported wavefront reduction strategy ${WF_REDUCE}")
endif ()

if (HIP_ENABLE_FLAG)
    set(SPMV_BIN_NAME spmv-hip)
else ()
    set(SPMV_BIN_NAME spmv-cpu)
endif ()

if (DEVICE_SIDE_VERIFY_FLAG AND (NOT HIP_ENABLE_FLAG))
    message(FATAL_ERROR "To perform device side verification, we must enable `HIP_ENABLE_FLAG`")
endif ()
