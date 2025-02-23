//
// Created by genshen on 2024/12/31.
//

#ifndef CSR_ADAPTIVE2_ANALYZE_H
#define CSR_ADAPTIVE2_ANALYZE_H

#include <vector>

template <typename I, int THREADS_PER_BLOCK, int VEC_SIZE>
I csr_adaptive_plus_analyze(const I m, const I nnz, const I MIN_NNZ_PER_BLOCK, std::vector<I> &break_points,
                            const I *host_row_ptr, const I *dev_row_ptr);

#endif // CSR_ADAPTIVE2_ANALYZE_H
