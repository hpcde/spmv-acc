
set(SP_LIB_SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR})

include(flat/flat_source_list.cmake)

if (BENCHMARK_CUDA_ENABLE_FLAG)
    # add hola and cub
    include(cub/cub_source_list.cmake)
    include(hola/hola_source_list.cmake)
    include(merge-path/merge_path_source_list.cmake)
    include(acsr/acsr_source_list.cmake)
else ()
    # add hola-hip
    include(hola-hip/hola-hip_source_list.cmake)
endif()

set_source_files_properties(${SP_LIB_SOURCE} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_library(${SPMV_BENCHMARK_SPARSE_LIB} STATIC ${SP_LIB_HEADER} ${SP_LIB_SOURCE} HIPCC_OPTIONS ${CUB_HIPCC_OPTIONS})
target_link_libraries(
        ${SPMV_BENCHMARK_SPARSE_LIB}
        PUBLIC
        ${SPMV_CLI_LIB}
        ${SPMV_BENCHMARK_UTILS_LIB}
)

target_include_directories(
    ${SPMV_BENCHMARK_SPARSE_LIB}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${CMAKE_SOURCE_DIR}/third-party/holahip/hip-hola$<SEMICOLON>${CMAKE_SOURCE_DIR}/third-party/holaspmv$<SEMICOLON>${CMAKE_SOURCE_DIR}/third-party/include$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)
