# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_FILES
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spmv_hip_acc_imp.cpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spmv_hip_acc_imp.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/opt_double_buffer.hpp
        )
