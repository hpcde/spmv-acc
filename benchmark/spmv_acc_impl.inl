#include "api/types.h"
#include "hip-adaptive/adaptive.h"
#include "hip-block-row-ordinary/spmv_hip_acc_imp.h"
#include "hip-flat/spmv_hip_acc_imp.h"
#include "hip-light/spmv_hip_acc_imp.h"
#include "hip-line-enhance/line_enhance_spmv.h"
#include "hip-line/line_strategy.h"
#include "hip-thread-row/thread_row.h"
#include "hip-vector-row/vector_row.h"
#include "hip-wf-row/spmv_hip.h"
#include "hip/spmv_hip_acc_imp.h"

struct SpMVAccDefault : CsrSpMV<SpMVAccDefault> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    default_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};

struct SpMVAccAdaptive : CsrSpMV<SpMVAccAdaptive> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
  }
};

struct SpMVAccBlockRow : CsrSpMV<SpMVAccBlockRow> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    block_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};

struct SpMVAccFlat : CsrSpMV<SpMVAccFlat> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    flat_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
  }
};

struct SpMVAccLight : CsrSpMV<SpMVAccLight> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    light_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};

struct SpMVAccLine : CsrSpMV<SpMVAccLine> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    line_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};

struct SpMVAccThreadRow : CsrSpMV<SpMVAccThreadRow> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    thread_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};

struct SpMVAccVecRow : CsrSpMV<SpMVAccVecRow> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};

struct SpMVAccWfRow : CsrSpMV<SpMVAccWfRow> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    wf_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};

struct SpMVAccLineEnhance : CsrSpMV<SpMVAccLineEnhance> {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
  }
};