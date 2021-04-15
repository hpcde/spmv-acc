# enable to use hip to accelerate on GPU side, currently, it must be `ON`.
option(HIP_ENABLE_FLAG "Enable HIP" ON)

# default: perform result verification on CPU side.
# However, if it is set to `ON`, it will use device side verification.
option(DEVICE_SIDE_VERIFY_FLAG "Verify result of device side" OFF)

if (HIP_ENABLE_FLAG)
    set(SPMV_BIN_NAME spmv-hip)
else ()
    set(SPMV_BIN_NAME spmv-cpu)
endif ()

if (DEVICE_SIDE_VERIFY_FLAG AND (NOT HIP_ENABLE_FLAG))
    message(FATAL_ERROR "To perform device side verification, we must enable `HIP_ENABLE_FLAG`")
endif ()
