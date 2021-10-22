
set(HEADER_FILES
        csr_mtx_reader.hpp
        sparse_format.h
        timer.h
        utils.hpp
        verification.h
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
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${SPMV_CLI_NAME} PUBLIC OpenMP::OpenMP_CXX)
    else ()
        if (${OPENMP_FOUND} OR ${OpenMP_C_FOUND})
            set_target_properties(${SPMV_CLI_NAME} PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
            set_target_properties(${SPMV_CLI_NAME} PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
        else ()
            message(FATAL_ERROR "OpenMP not found, but required OpenMP support.")
        endif ()
    endif ()
    message(STATUS "Build ${SPMV_CLI_NAME} with OpenMP support.")
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
