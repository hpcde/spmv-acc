# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_adaptive_plus_spmv.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_adaptive_plus_spmv_imp.h
        )

set(ACC_SOURCE_FILES ${ACC_SOURCE_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_adaptive_plus_spmv.cpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_adaptive_plus_analyze.cpp
        )
