//
// Created by reget on 2021/11/16.
//

#ifndef SPMV_ACC_BENCHMARK_SPMV_ACC_HPP
#define SPMV_ACC_BENCHMARK_SPMV_ACC_HPP

#include "api/types.h"
#include "utils/timer_utils.h"
#include "utils/benchmark_time.h"

#include "csr_spmv.hpp"

#include "api/handle.h"
#include "flat/spmv_acc_flat.h"
#include "hip-adaptive/adaptive.h"
#include "hip-block-row-ordinary/spmv_hip_acc_imp.h"
#include "hip-csr-adaptive-plus/csr_adaptive_plus_spmv.h"
#include "hip-light/spmv_hip_acc_imp.h"
#include "hip-line-enhance/line_enhance_spmv.h"
#include "hip-line/line_strategy.h"
#include "hip-thread-row/thread_row.h"
#include "hip-vector-row/vector_row.h"
#include "hip-wf-row/spmv_hip.h"
#include "hip/spmv_hip_acc_imp.h"

struct SpMVAccDefault : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    default_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccAdaptive : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccBlockRow : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    block_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccFlat : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    flat_sparse_spmv<FLAT_PRE_CALC_BP_KERNEL_VERSION_V1>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccFlatV2 : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    flat_sparse_spmv<FLAT_PRE_CALC_BP_KERNEL_VERSION_V2>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccFlatSegSum : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    segment_sum_flat_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccLight : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    light_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccLine : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    adaptive_line_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccThreadRow : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    thread_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccVecRow : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccWfRow : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    wf_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccLineEnhance : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

struct SpMVAccAdaptivePlus : CsrSpMV {
  void csr_spmv_impl(int trans, const int alpha, const int beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y, BenchmarkTime *bmt) {
    SpMVAccHanele handle;
    csr_adaptive_plus_sparse_spmv<true, int, double>(&handle, trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);

    double pre_time_cost = handle.profile_analyze_time;
    double calc_time_cost = handle.profile_kernel_time;
    double destroy_time_cost = handle.profile_destroy_time;
    if (bmt != nullptr) {
      bmt->set_time(pre_time_cost, calc_time_cost, 0.0, destroy_time_cost);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // SPMV_ACC_BENCHMARK_SPMV_ACC_HPP
