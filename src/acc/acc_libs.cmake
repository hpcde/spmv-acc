set(CURRENT_ACC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/acc)

if (ARCH_NAME)
    # set(ACC_LIBS spmv_acc_${ARCH_NAME}) # Deprecated

    # set default arch directory and include it.
    if (NOT ACC_SRC_PATH)
        set(ACC_SRC_PATH "${CURRENT_ACC_SOURCE_DIR}/${ARCH_NAME}")
    endif ()
    # check ACC_SRC_PATH and add_subdirectory
    if (NOT EXISTS "${ACC_SRC_PATH}" OR NOT IS_DIRECTORY "${ACC_SRC_PATH}")
        message(FATAL_ERROR "Architecture source files directory not found: ${ACC_SRC_PATH}")
    else ()
        MESSAGE(STATUS "Arch source files is ${ACC_SRC_PATH}")
        include(${ACC_SRC_PATH}/source_list.cmake) # set ${ACC_FILES} here.
        # set(MD_SOURCE_INCLUDES "${PROJECT_SOURCE_DIR}/src" CACHE PATH "PATH of includes in arch code.")
        # add_subdirectory(${ACC_SRC_PATH} arch_${ARCH_NAME})
    endif ()
endif ()
