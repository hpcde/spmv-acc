//
// Created by genshen on 2024/12/31.
//

#include <cassert>
#include <cstdio>
#include <vector>

#include "csr_adaptive_plus_analyze.h"
#include "csr_adaptive_plus_config.h"

template <typename I, int THREADS_PER_BLOCK, int VEC_SIZE>
I csr_adaptive_plus_analyze_imp(const I m, const I nnz, const I MIN_NNZ_PER_BLOCK, std::vector<I> &break_points,
                                std::vector<I> &first_block_of_row, const I *host_row_ptr, const I *dev_row_ptr) {
  // each block process en entire row and at least `MIN_NNZ_PER_BLOCK` nnz.
  I bp_index = 0;
  I nnz_count = 0;
  I row_count_i = 0;

  // index -1. Used in block of processing very long row for checking its previous block.
  // break_points.emplace_back(-1);
  break_points.emplace_back(0); // index: 0
  first_block_of_row[0] = 0;
  // for(int i =0; i< first_block_of_row.size(); i++) {
  //   first_block_of_row[i] = 0;
  // }

  constexpr I max_rows_per_block = THREADS_PER_BLOCK / VEC_SIZE;

  for (I i = 1; i <= m; i++) {
    I nnz_current_row = host_row_ptr[i] - host_row_ptr[i - 1];
    nnz_count += nnz_current_row;
    row_count_i++;
    if (nnz_count >= MIN_NNZ_PER_BLOCK) { // nnz count is large enough or it is the last row.
      if (spmv::gpu::adaptive_plus::DEBUG) {
        if (bp_index > (nnz / MIN_NNZ_PER_BLOCK + 1)) {
          printf("%d %d %d %d^ \n", bp_index, nnz / MIN_NNZ_PER_BLOCK + 1, nnz_count, i);
        }
        assert(bp_index <= (nnz / MIN_NNZ_PER_BLOCK + 1));
      }
      // a row can be very long, we assign the row to multiple blocks.
      // todo: another way: set array of length m. Each element in the array
      //       denotes the first block of processing this row.
      //       In spmv kernel, if block_id != the_firt_id.
      const I multi_blocks = nnz_current_row / MIN_NNZ_PER_BLOCK;
      const I is_multi_block_row = multi_blocks > 1 ? 1 : 0;
      if (is_multi_block_row) {
        for (int k = 0; k < multi_blocks; k++) {
          if (k == 0 && nnz_count == nnz_current_row) { // pre block is clean.
            // clean block. we ignore it. because row (i-1) is already asssigned to this block.
          } else {
             // assign row (i-1) to the next block.
             break_points.emplace_back(i - 1); // record the starting row for the next block.
          }
          if (k == 0) {
            // use a bit flag to notate if the row spaning multiple blocks.
            first_block_of_row[i - 1] = (break_points.size() - 1) * 2 + is_multi_block_row;
          }
        }
        // assign row i to the next block.
        break_points.emplace_back(i); // record the starting row for the next block.
      } else {
        break_points.emplace_back(i); // record the starting row for the next block.
        // row's first processing block is omitted.
      }
      nnz_count = 0; // reset nnz count
      row_count_i = 0;
    } else if ((row_count_i >= max_rows_per_block) || (i == m)) {
      // nnz count < MIN_NNZ_PER_BLOCK, but exceeds row limit, do: accept this row, put next row to next block.
      break_points.emplace_back(i); // set the starting row for the next block.
      first_block_of_row[i] = (break_points.size() - 1) * 2;
      nnz_count = 0; // reset nnz count
      row_count_i = 0;
    }
  }
  if (spmv::gpu::adaptive_plus::DEBUG) {
    printf("start row of each block (len = N blocks):\n");
    int j = 0;
    for (auto bp : break_points) {
      printf("%d:%d,", j, bp);
      j++;
    }
    printf("\n");
    // ----
    printf("first block of each row (len = m + 1):\n");
    int i = 0;
    for (auto bid : first_block_of_row) {
      if (bid != 0) {
        printf("%d:%d,", i, bid / 2);
      }
      i++;
    }
    printf("\n");
  }
  return break_points.size() - 1; // return the new block number or bp array length.
}

#define INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(PARAM_THREADS_PER_BLOCK, PARAM_VEC_SIZE)                                \
  template int csr_adaptive_plus_analyze_imp<int, PARAM_THREADS_PER_BLOCK, PARAM_VEC_SIZE>(                            \
      const int m, const int nnz, const int MIN_NNZ_PER_BLOCK, std::vector<int> &break_points,                         \
      std::vector<int> &row_block_id, const int *host_row_ptr, const int *dev_row_ptr);

// instance of csr_adaptive_plus_analyze_imp for THREADS_PER_BLOCK = 256
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(256, 1);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(256, 2);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(256, 4);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(256, 8);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(256, 16);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(256, 32);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(256, 64);

// instance of csr_adaptive_plus_analyze_imp for THREADS_PER_BLOCK = 512
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(512, 1);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(512, 2);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(512, 4);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(512, 8);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(512, 16);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(512, 32);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(512, 64);

// instance of csr_adaptive_plus_analyze_imp for THREADS_PER_BLOCK = 1024
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(1024, 1);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(1024, 2);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(1024, 4);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(1024, 8);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(1024, 16);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(1024, 32);
INSTANCE_CSR_ADAPTIVE_PLUS_ANALYZE_IMP(1024, 64);
