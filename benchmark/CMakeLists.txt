set(SPMV_BENCHMARK_UTILS_LIB benchmarkutil)
add_subdirectory(utils)

set(BENCHMARK_DEP_LIBS ${SPMV_CLI_LIB} ${SPMV_BENCHMARK_UTILS_LIB})

if (BENCHMARK_CUDA_ENABLE_FLAG)
    set(SPMV_BENCHMARK_CUB_LIB cubspmv)
    set(SPMV_BENCHMARK_HOLA_LIB holaspmv)
    add_subdirectory(cub)
    add_subdirectory(hola)

    set(BENCHMARK_DEP_LIBS
            ${BENCHMARK_DEP_LIBS}
            ${SPMV_BENCHMARK_CUB_LIB}
            ${SPMV_BENCHMARK_HOLA_LIB}
            cudart
            cusparse
            )
else ()
    set(SPMV_BENCHMARK_HOLA_HIP_LIB holahip)
    add_subdirectory(hola-hip)

    set(BENCHMARK_DEP_LIBS
            ${BENCHMARK_DEP_LIBS}
            ${SPMV_BENCHMARK_HOLA_HIP_LIB}
            rocsparse
            )
endif ()

set(HEADER_FILES
        )

set(SOURCE_FILES
        main.cpp
        )

set(SPMV_BENCHMARK_NAME spmv-gpu-benchmark)

# set binary path and lib storage path for compiling process.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_executable(${SPMV_BENCHMARK_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(
        ${SPMV_BENCHMARK_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${CMAKE_SOURCE_DIR}/third-party/include$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
        ${SPMV_BENCHMARK_NAME}
        PUBLIC
        ${BENCHMARK_DEP_LIBS}
)

##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${SPMV_BENCHMARK_NAME} ${SPMV_BENCHMARK_CUB_LIB} ${SPMV_BENCHMARK_HOLA_LIB} ${SPMV_BENCHMARK_UTILS_LIB} ${SPMV_BENCHMARK_HOLA_HIP_LIB}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
