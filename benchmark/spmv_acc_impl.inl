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
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    default_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccAdaptive : CsrSpMV<SpMVAccAdaptive> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccBlockRow : CsrSpMV<SpMVAccBlockRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    block_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccFlat : CsrSpMV<SpMVAccFlat> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    flat_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccLight : CsrSpMV<SpMVAccLight> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    light_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccLine : CsrSpMV<SpMVAccLine> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    line_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccThreadRow : CsrSpMV<SpMVAccThreadRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    thread_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccVecRow : CsrSpMV<SpMVAccVecRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccWfRow : CsrSpMV<SpMVAccWfRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    wf_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};

struct SpMVAccLineEnhance : CsrSpMV<SpMVAccLineEnhance> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
    adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return true;
  }
};