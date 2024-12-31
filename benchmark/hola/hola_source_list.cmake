set(SP_LIB_HEADER ${SP_LIB_HEADER}
        ${SP_LIB_SRC_BASE}/hola/spmv.h
        ${HOLASPMV_SRC_DIR}/include/dCSR.h
        ${HOLASPMV_SRC_DIR}/include/dVector.h
        ${HOLASPMV_SRC_DIR}/include/holaspmv.h
        ${HOLASPMV_SRC_DIR}/include/common.cuh
        )

set(SP_LIB_SOURCE ${SP_LIB_SOURCE}
        ${SP_LIB_SRC_BASE}/hola/spmv.cpp
        ${HOLASPMV_SRC_DIR}/source/holaspmv.cu
        ${HOLASPMV_SRC_DIR}/source/dCSR.cpp
        )
