set(SPMV_BENCHMARK_UTILS_LIB benchmarkutil)
add_subdirectory(utils)

set(SPMV_BENCHMARK_SPARSE_LIB benchmark_sparse)
set(BENCHMARK_DEP_LIBS ${SPMV_CLI_LIB} ${SPMV_BENCHMARK_UTILS_LIB} ${SPMV_BENCHMARK_SPARSE_LIB})

include(sparse.cmake)

if (BENCHMARK_CUDA_ENABLE_FLAG)
    set(BENCHMARK_DEP_LIBS
            ${BENCHMARK_DEP_LIBS}
            cudart
            cusparse
            )
else ()
    set(BENCHMARK_DEP_LIBS
            ${BENCHMARK_DEP_LIBS}
            rocsparse
            )
endif ()

set(HEADER_FILES
        utils/statistics_logger.h
        )

set(SOURCE_FILES
        main.cpp
        utils/statistics_logger.cpp
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

if(BENCHMARK_FORCE_SYNC_KERNELS)
        target_compile_definitions(${SPMV_BENCHMARK_NAME} PUBLIC MACRO_BENCHMARK_FORCE_KERNEL_SYNC=1)
        message(STATUS "force kernel sync for target ${SPMV_BENCHMARK_NAME} is enabled")
else()
        message(STATUS "force kernel sync for target ${SPMV_BENCHMARK_NAME} is disabled")
endif()

target_link_libraries(
        ${SPMV_BENCHMARK_NAME}
        PUBLIC
        ${BENCHMARK_DEP_LIBS}
)

# See https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
if (SPMV_OMP_ENABLED_FLAG)
    find_package(OpenMP)
    if (${OPENMP_FOUND} AND ${OpenMP_CXX_FOUND})
        set_target_properties(${SPMV_BENCHMARK_NAME} PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
        set_target_properties(${SPMV_BENCHMARK_NAME} PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
    elseif (${OPENMP_FOUND} OR ${OpenMP_C_FOUND})
        set_target_properties(${SPMV_BENCHMARK_NAME} PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
        set_target_properties(${SPMV_BENCHMARK_NAME} PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
    else ()
        message(FATAL_ERROR "OpenMP not found, but required OpenMP support.")
    endif ()
    message(STATUS "Build ${SPMV_CLI_NAME} with OpenMP support.")
endif ()

##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${SPMV_BENCHMARK_NAME} ${SPMV_BENCHMARK_CUB_LIB} ${SPMV_BENCHMARK_HOLA_LIB} ${SPMV_BENCHMARK_UTILS_LIB} ${SPMV_BENCHMARK_HOLA_HIP_LIB}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
