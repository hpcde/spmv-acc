
set(HEADER_FILES
        mtx_reader.hpp
        )

set(SOURCE_FILES
        main.cpp
        )

set(SPMV_CLI_NAME spmv-cli)

# set binary path and lib storage path for compiling process.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_executable(${SPMV_CLI_NAME} ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(
        ${SPMV_CLI_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
        ${SPMV_CLI_NAME}
        PUBLIC
        ${ACC_LIBS}
)

# See https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
if (SPMV_OMP_ENABLED_FLAG)
    find_package(OpenMP REQUIRED)
    if (NOT TARGET OpenMP::OpenMP_CXX)
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif ()
    target_link_libraries(${SPMV_CLI_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif ()


##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${SPMV_CLI_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
