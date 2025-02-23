//
// Created by genshen on 2024/12/31.
//

#include <cassert>
#include <cstdio>
#include <vector>

#include "csr_adaptive_plus_analyze.h"
#include "csr_adaptive_plus_config.h"

template <typename I, int THREADS_PER_BLOCK, int VEC_SIZE>
I csr_adaptive_plus_analyze(const I m, const I nnz, const I MIN_NNZ_PER_BLOCK, std::vector<I> &break_points,
                            const I *host_row_ptr, const I *dev_row_ptr) {
  // each block process en entire row and at least `MIN_NNZ_PER_BLOCK` nnz.
  I bp_index = 0;
  I nnz_count = 0;
  break_points.emplace_back(0);
  I row_count_i = 0;

  constexpr I max_rows_per_block = THREADS_PER_BLOCK / VEC_SIZE;

  for (I i = 1; i <= m; i++) {
    nnz_count += host_row_ptr[i] - host_row_ptr[i - 1];
    row_count_i++;
    if (nnz_count >= MIN_NNZ_PER_BLOCK || (row_count_i >= max_rows_per_block) ||
        (i == m)) { // nnz count is large enough ro it is the last row.
      if (spmv::gpu::adaptive_plus::DEBUG) {
        if (bp_index > (nnz / MIN_NNZ_PER_BLOCK + 1)) {
          printf("%d %d %d %d^ \n", bp_index, nnz / MIN_NNZ_PER_BLOCK + 1, nnz_count, i);
        }
        assert(bp_index <= (nnz / MIN_NNZ_PER_BLOCK + 1));
      }
      break_points.emplace_back(i); // record the starting row for the next block.
      nnz_count = 0;                // reset nnz count
      row_count_i = 0;
    }
  }
  return break_points.size() - 1; // return the new block number or bp array length.
}

// instance of csr_adaptive_plus_analyze.
template int csr_adaptive_plus_analyze<int, 512, 8>(const int m, const int nnz, const int MIN_NNZ_PER_BLOCK,
                                                    std::vector<int> &break_points, const int *host_row_ptr,
                                                    const int *dev_row_ptr);

template int csr_adaptive_plus_analyze<int, 512, 16>(const int m, const int nnz, const int MIN_NNZ_PER_BLOCK,
                                                     std::vector<int> &break_points, const int *host_row_ptr,
                                                     const int *dev_row_ptr);
