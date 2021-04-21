cmake_minimum_required(VERSION 3.8)

project(spmv-hip CXX)

set(CMAKE_CXX_STANDARD 11)

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


set(SPMV_HIP_ACC_FILES
        spmv_hip_acc_imp.cpp
        )
set_source_files_properties(${SPMV_HIP_ACC_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

set(SPMV_HIP_ACC_LIB_NAME spmv_acc_hip)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

hip_add_library(${SPMV_HIP_ACC_LIB_NAME} STATIC ${SPMV_HIP_ACC_FILES})
# target_compile_options(${SPMV_HIP_ACC_LIB_NAME} PRIVATE "-Xcompiler" "-fPIC")

target_include_directories(
        ${SPMV_HIP_ACC_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
        ${SPMV_HIP_ACC_LIB_NAME}
        PUBLIC
        hip_hcc  # hip lib for dcu.
        rocsparse
)
