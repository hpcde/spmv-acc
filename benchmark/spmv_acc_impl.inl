#include "api/types.h"
#include "timer.h"
#include "utils/benchmark_time.h"

#include "hip-adaptive/adaptive.h"
#include "hip-block-row-ordinary/spmv_hip_acc_imp.h"
#include "hip-light/spmv_hip_acc_imp.h"
#include "hip-line-enhance/line_enhance_spmv.h"
#include "hip-line/line_strategy.h"
#include "hip-thread-row/thread_row.h"
#include "hip-vector-row/vector_row.h"
#include "hip-wf-row/spmv_hip.h"
#include "hip/spmv_hip_acc_imp.h"
#include "spmv_acc_flat.hpp"

struct SpMVAccDefault : CsrSpMV<SpMVAccDefault> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    default_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccAdaptive : CsrSpMV<SpMVAccAdaptive> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccBlockRow : CsrSpMV<SpMVAccBlockRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    block_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccFlat : CsrSpMV<SpMVAccFlat> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    flat_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
    return true;
  }
};

struct SpMVAccLight : CsrSpMV<SpMVAccLight> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    light_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccLine : CsrSpMV<SpMVAccLine> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    line_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccThreadRow : CsrSpMV<SpMVAccThreadRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    thread_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccVecRow : CsrSpMV<SpMVAccVecRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccWfRow : CsrSpMV<SpMVAccWfRow> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    wf_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};

struct SpMVAccLineEnhance : CsrSpMV<SpMVAccLineEnhance> {
  bool csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    my_timer calc_timer;
    calc_timer.start();
    adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    hipDeviceSynchronize();
    calc_timer.stop();
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.);
    }
    return true;
  }
};