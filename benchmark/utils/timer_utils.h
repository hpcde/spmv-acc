//
// Created by chu genshen on 2025/5/7.
//

#ifndef SPMV_ACC_HIP_TIMER_UTILS_H
#define SPMV_ACC_HIP_TIMER_UTILS_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "benchmark_config.h"

// another timer implementation using hip Event.
namespace hip {
    namespace timer {
      struct event_timer {
        hipEvent_t _start, _stop;
        double time_use; // us
        static constexpr bool use_event_sync = true;

        event_timer() {
          hipEventCreate(&_start);
          hipEventCreate(&_stop);
        }

        ~event_timer() {
          hipEventDestroy(_start);
          hipEventDestroy(_stop);
        }

        inline void start() {
          hipEventRecord(_start, NULL);
        }

        /// @brief lazy device sync will try to not sync device unless user the sync flag (from global config or local config) has been specificed.
        /// @param sync_flag the local sync flag.
        inline void stop(bool sync_flag = false) {
          hipEventRecord(_stop, NULL);
          if(sync_flag || BENCHMARK_FORCE_KERNEL_SYNC) {
            if (use_event_sync) {
              hipEventSynchronize(_stop);
            } else {
              hipDeviceSynchronize();
            }
          }

          float delta_t = 0.0f;
          hipEventElapsedTime(&delta_t, _start, _stop);
          time_use = delta_t * 1000.0;
        }
      };
    }
}

#endif // SPMV_ACC_HIP_TIMER_UTILS_H
