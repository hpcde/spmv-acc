set(SP_LIB_HEADER ${SP_LIB_HEADER}
        ${SP_LIB_SRC_BASE}/merge-path/merge_path_spmv.h
        ${SP_LIB_SRC_BASE}/merge-path/merge_path_config.h
        ${SP_LIB_SRC_BASE}/merge-path/merge_path_partition.h
        ${SP_LIB_SRC_BASE}/merge-path/merge_path_reduction.h
        ${SP_LIB_SRC_BASE}/merge-path/merge_path_update.h
        ${SP_LIB_SRC_BASE}/merge-path/merge_path_utils.h
        )

set(SP_LIB_SOURCE ${SP_LIB_SOURCE}
        ${SP_LIB_SRC_BASE}/merge-path/merge_path_spmv.cu
        )
