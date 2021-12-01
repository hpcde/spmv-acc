set(SP_LIB_HEADER ${SP_LIB_HEADER}
        ${SP_LIB_SRC_BASE}/hola/spmv.h
        ${CMAKE_SOURCE_DIR}/third-party/holaspmv/include/dCSR.h
        ${CMAKE_SOURCE_DIR}/third-party/holaspmv/include/dVector.h
        ${CMAKE_SOURCE_DIR}/third-party/holaspmv/include/holaspmv.h
        ${CMAKE_SOURCE_DIR}/third-party/holaspmv/include/common.cuh
        )

set(SP_LIB_SOURCE ${SP_LIB_SOURCE}
        ${SP_LIB_SRC_BASE}/hola/spmv.cpp
        ${CMAKE_SOURCE_DIR}/third-party/holaspmv/source/holaspmv.cu
        ${CMAKE_SOURCE_DIR}/third-party/holaspmv/source/dCSR.cpp
        )
