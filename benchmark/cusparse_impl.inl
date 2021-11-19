#include <cusparse.h>

#include <api/types.h>

struct CuSparseGeneral : CsrSpMV<CuSparseGeneral> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    const double cu_alpha = static_cast<double>(alpha);
    const double cu_beta = static_cast<double>(beta);
    // Create cuSPARSE handle
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    // Create matrix, vector x and vector y
    cusparseSpMatDescr_t cu_mat;
    cusparseDnVecDescr_t cu_x, cu_y;
    cusparseCreateCsr(&cu_mat, h_csr_desc.rows, h_csr_desc.cols, h_csr_desc.nnz, (void *)d_csr_desc.row_ptr,
                      (void *)d_csr_desc.col_index, (void *)d_csr_desc.values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&cu_x, h_csr_desc.cols, (void *)x, CUDA_R_64F);
    cusparseCreateDnVec(&cu_y, h_csr_desc.rows, (void *)y, CUDA_R_64F);
    // Allocate an external buffer
    void *d_buffer = NULL;
    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &cu_alpha, cu_mat, cu_x, &cu_beta, cu_y,
                            CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &buffer_size);
    cudaMalloc(&d_buffer, buffer_size);
    // Execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &cu_alpha, cu_mat, cu_x, &cu_beta, cu_y, CUDA_R_64F,
                 CUSPARSE_MV_ALG_DEFAULT, d_buffer);
    // Clear up on device
    cusparseDestroySpMat(cu_mat);
    cusparseDestroyDnVec(cu_x);
    cusparseDestroyDnVec(cu_y);
    cusparseDestroy(handle);
    return true;
  }
};