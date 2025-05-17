cmake_minimum_required(VERSION 3.21)
project(SPMV-Acc)

set(CMAKE_CXX_STANDARD 14)

include(config.cmake)
include(pkg.dep.cmake)

add_subdirectory(src)
add_subdirectory(cli)
if(SPMV_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

if(SPMV_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
