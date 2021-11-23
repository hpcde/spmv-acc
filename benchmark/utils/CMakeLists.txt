set(BENCHMARK_UTILS_HEADER_FILES
        benchmark_time.h
        )

set(BENCHMARK_UTILS_SOURCE_FILES
        benchmark_time.cpp
        )

# set binary path and lib storage path for compiling process.
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# build static lib
add_library(${SPMV_BENCHMARK_UTILS_LIB} STATIC ${BENCHMARK_UTILS_SOURCE_FILES} ${BENCHMARK_UTILS_HEADER_FILES})

target_link_libraries(
        ${SPMV_BENCHMARK_UTILS_LIB}
        PUBLIC
        ${SPMV_CLI_LIB}
)