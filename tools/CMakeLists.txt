
add_executable(csr-reader main.cpp)

target_include_directories(csr-reader
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        )
