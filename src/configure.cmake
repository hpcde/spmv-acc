# configure a header file to pass some of the CMake settings to the source code
if (HIP_ENABLE_FLAG)
    set(gpu ON)
    set(ARCH_NAME hip)
    set(ACCELERATE_ENABLED ON)
    set(ARCH_HIP ON)
    MESSAGE(STATUS "HIP acceleration is enabled")
else()

endif ()

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/building_config.h.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/building_config.h"
)
