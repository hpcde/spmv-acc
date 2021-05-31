# configure a header file to pass some of the CMake settings to the source code
if (HIP_ENABLE_FLAG)
    set(ARCH_NAME hip)
    set(ACCELERATE_ENABLED ON)
    set(ARCH_HIP ON)
    MESSAGE(STATUS "HIP acceleration is enabled")
else ()

endif ()

if (DEVICE_SIDE_VERIFY_FLAG)
    set(DEVICE_SIDE_VERIFY ON)
    set(gpu ON)
endif ()


if (WF_REDUCE_LOWER MATCHES "default")
    set(WF_REDUCE_DEFAULT ON)
elseif (WF_REDUCE_LOWER MATCHES "lds")
    set(WF_REDUCE_LDS ON)
elseif (WF_REDUCE_LOWER MATCHES "reg")
    set(WF_REDUCE_REG ON)
endif ()

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/building_config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/building_config.h"
)

# install the generated file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/building_config.h"
        DESTINATION "include"
        )
