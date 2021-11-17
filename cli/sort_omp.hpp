//
// Created by genshen on 2021/11/17.
//

#ifndef SPMV_ACC_SORT_OMP_H
#define SPMV_ACC_SORT_OMP_H

#ifdef _OPENMP
#include <omp.h>

// the code in this file is copyied and modified from: https://github.com/eduardlopez/quicksort-parallel.
namespace sort {
  template <typename I, typename T> void quickSort_parallel_internal(T *array, I left, I right, I cutoff);

  template <typename I, typename T> void quickSort_parallel(T *array, I lenArray, I numThreads, I cutoff = 1024) {
#pragma omp parallel num_threads(numThreads)
    {
#pragma omp single nowait
      { quickSort_parallel_internal(array, static_cast<I>(0), static_cast<I>(lenArray - 1), static_cast<I>(cutoff)); }
    }
  }

  template <typename I, typename T> void quickSort_parallel_internal(T *array, I left, I right, I cutoff) {
    I i = left, j = right;
    T tmp;
    T pivot = array[(left + right) / 2];

    {
      /* PARTITION PART */
      while (i <= j) {
        while (array[i] < pivot) {
          i++;
        }
        while (array[j] > pivot) {
          j--;
        }
        if (i <= j) {
          tmp = array[i];
          array[i] = array[j];
          array[j] = tmp;
          i++;
          j--;
        }
      }
    }

    if (((right - left) < cutoff)) {
      if (left < j) {
        quickSort_parallel_internal(array, left, j, cutoff);
      }
      if (i < right) {
        quickSort_parallel_internal(array, i, right, cutoff);
      }
    } else {
#pragma omp task
      { quickSort_parallel_internal(array, left, j, cutoff); }
#pragma omp task
      { quickSort_parallel_internal(array, i, right, cutoff); }
    }
  }
} // namespace sort

#endif // _OPENMP
#endif // SPMV_ACC_SORT_OMP_H
