
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

##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${SPMV_CLI_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

# install header.
install(DIRECTORY "./"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        )
