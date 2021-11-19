set(SPMV_BENCHMARK_CUB_LIB cubspmv)
set(SPMV_BENCHMARK_HOLA_LIB holaspmv)

add_subdirectory(cub)
add_subdirectory(hola)

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
        ${SPMV_CLI_LIB}
        ${SPMV_BENCHMARK_CUB_LIB}
        ${SPMV_BENCHMARK_HOLA_LIB}
        cudart
        cusparse
)


##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${SPMV_BENCHMARK_NAME} ${SPMV_BENCHMARK_CUB_LIB} ${SPMV_BENCHMARK_HOLA_LIB} ${SPMV_BENCHMARK_SCALAR_MUL_LIB}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
