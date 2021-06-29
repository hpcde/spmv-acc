# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_FILES
        ${CURRENT_ACC_SOURCE_DIR}/common/global_mem_ops.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spmv_hip.cpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spmv_hip.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/wavefront_row_default.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/wavefront_row_lds.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/wavefront_row_reg.hpp
        )
