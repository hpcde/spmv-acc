option(HIP_ENABLE_FLAG "Enable HIP" OFF) # enable to use hip to accelerate on GPU side

if (HIP_ENABLE_FLAG)
    set(SPMV_BIN_NAME spmv-hip)
else ()
    set(SPMV_BIN_NAME spmv-cpu)
endif ()
