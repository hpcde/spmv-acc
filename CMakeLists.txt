cmake_minimum_required(VERSION 3.6)
project(SPMV-Acc)

set(CMAKE_CXX_STANDARD 11)

include(config.cmake)

add_subdirectory(src)

if(SPMV_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
