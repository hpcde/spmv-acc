set(SP_LIB_HEADER ${SP_LIB_HEADER}
        ${SP_LIB_SRC_BASE}/spmv.h
        ${CMAKE_SOURCE_DIR}/third-party/holahip/hip-hola/d_csr.h
        ${CMAKE_SOURCE_DIR}/third-party/holahip/hip-hola/hola_vector.h
        ${CMAKE_SOURCE_DIR}/third-party/holahip/hip-hola/hola_spmv.h
        )

set(SP_LIB_SOURCE ${SP_LIB_SOURCE}
        ${SP_LIB_SRC_BASE}/hola-hip/spmv.cpp
        ${CMAKE_SOURCE_DIR}/third-party/holahip/hip-hola/hola_spmv.cpp
        ${CMAKE_SOURCE_DIR}/third-party/holahip/hip-hola/d_csr.cpp
        )
