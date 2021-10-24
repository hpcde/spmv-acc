
add_executable(csr-reader main.cpp)

target_include_directories(csr-reader
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${CMAKE_SOURCE_DIR}/third-party/include>
        $<INSTALL_INTERFACE:include>
        )
