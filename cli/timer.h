//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_TIMER_H
#define SPMV_ACC_TIMER_H

#include <sys/time.h>
#include <unistd.h>

struct my_timer {
  struct timeval start_time, end_time;
  double time_use; // us
  void start() { gettimeofday(&start_time, NULL); }
  void stop() {
    gettimeofday(&end_time, NULL);
    time_use = (end_time.tv_sec - start_time.tv_sec) * 1.0e6 + end_time.tv_usec - start_time.tv_usec;
  }
};

#endif // SPMV_ACC_TIMER_H
