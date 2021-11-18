set(CUB_HEADER_FILES
        spmv.h
        )

set(CUB_SOURCE_FILES
        spmv.cpp
        )

# set binary path and lib storage path for compiling process.
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# find HIP
if (HIP_ENABLE_FLAG)
    # see https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/2_Cookbook/12_cmake_hip_add_executable
    if (NOT DEFINED HIP_PATH)
        if (NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else ()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif ()
    endif ()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

    find_package(HIP REQUIRED)
    if (HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else ()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif ()
    set(CUB_HIPCC_OPTIONS)
endif ()

# build static lib
if (HIP_ENABLE_FLAG)
    hip_add_library(${SPMV_BENCHMARK_CUB_LIB} STATIC ${CUB_SOURCE_FILES} ${CUB_HEADER_FILES}  HIPCC_OPTIONS ${CUB_HIPCC_OPTIONS})
    target_link_libraries(
            ${SPMV_BENCHMARK_CUB_LIB}
            PUBLIC
            ${SPMV_CLI_LIB}
    )
    target_include_directories(
        ${SPMV_BENCHMARK_CUB_LIB}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
    )
endif ()