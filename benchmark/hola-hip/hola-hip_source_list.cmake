set(SP_LIB_HEADER ${SP_LIB_HEADER}
        ${SP_LIB_SRC_BASE}/hola-hip/spmv.h
        ${HOLA_HIP_SRC_DIR}/hip-hola/d_csr.h
        ${HOLA_HIP_SRC_DIR}/hip-hola/hola_vector.h
        ${HOLA_HIP_SRC_DIR}/hip-hola/hola_spmv.h
        )

set(SP_LIB_SOURCE ${SP_LIB_SOURCE}
        ${SP_LIB_SRC_BASE}/hola-hip/spmv.cpp
        ${HOLA_HIP_SRC_DIR}/hip-hola/hola_spmv.cpp
        ${HOLA_HIP_SRC_DIR}/hip-hola/d_csr.cpp
        )
