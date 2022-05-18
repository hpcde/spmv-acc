//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_SPARSE_FORMAT_H
#define SPMV_ACC_SPARSE_FORMAT_H

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "api/types.h"
#ifdef _OPENMP
#include "sort_omp.hpp"
#endif

typedef double dtype;

// CSR sparse matrix type
template <typename I, typename T> class csr_mtx : public var_csr_desc<I, T> {
public:
  void alloc(I _rows, I _cols, I _nnz) {
    this->nnz = _nnz;
    this->rows = _rows;
    this->cols = _cols;

    this->row_ptr = new I[_rows + 1];
    this->col_index = new I[_nnz];
    this->values = new T[_nnz];
  }
};

// an element in COO matrix
template <typename I, typename T> struct Entry {
  I r, c;
  T v;
  bool operator<(const Entry &other) const {
    if (r != other.r) {
      return r < other.r;
    }
    return c < other.c;
  }
  bool operator>(const Entry &other) const {
    if (r != other.r) {
      return r > other.r;
    }
    return c > other.c;
  }
};

// COO sparse matrix type
template <typename I, typename T> class coo_mtx {

public:
  I rows = 0;
  I cols = 0;
  I nnz = 0;

  std::unique_ptr<I[]> row_index = nullptr;
  std::unique_ptr<I[]> col_index = nullptr;
  std::unique_ptr<T[]> values = nullptr;
};

/**
 * head information of matrix market format.
 */
struct mm_header {
  std::size_t num_rows;
  std::size_t num_columns;
  std::size_t num_non_zeroes; // nnz in file body.
  bool pattern;
  bool hermitian;
  bool complex;
  bool symmetric;
};

struct body_line {
  std::size_t line_num;
  std::string line;
};

template <typename I, typename T> class matrix_market {
public:
  typedef Entry<I, T> type_entry;

  mm_header header;
  /*
   * nnz in real matrix.
   * it can be different from nnz in header (which is non-zeros in matrix-market file)
   * if the matrix is symmetric or hermitian.
   */
  std::size_t nnz;
  std::unique_ptr<type_entry[]> data;

  void set_header(const mm_header _header) { this->header = _header; }

  // convert to csr matrix format
  csr_mtx<I, T> to_csr() {
    csr_mtx<I, T> csr;
    type_entry *entries = data.get();
    std::size_t N = this->nnz;

    // sort by row id, then column id
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    sort::quickSort_parallel<long, Entry<I, T>>(entries, static_cast<long>(N), static_cast<long>(max_threads));
#endif
#ifndef _OPENMP
    std::sort(entries, entries + N);
#endif

    csr.alloc(this->header.num_rows, this->header.num_columns, this->nnz);
    memset(csr.row_ptr, 0, (this->header.num_rows + 1) * sizeof(I));

    for (size_t i = 0; i < this->nnz; i++) {
      csr.values[i] = entries[i].v;
      csr.col_index[i] = entries[i].c;
      ++csr.row_ptr[entries[i].r + 1]; // this line of code only set nnz of current row
    }

    for (size_t i = 0; i < this->header.num_rows; i++) {
      const unsigned int nnz_this_row = csr.row_ptr[i + 1];
      csr.row_ptr[i + 1] = csr.row_ptr[i] + nnz_this_row;
    }
    return csr;
  }

  coo_mtx<I, T> to_coo() {
    // todo:
  }

  /**
   * set metadata and allocate memory for COO matrix.
   */
  void alloc(I _rows, I _cols, I _nnz) {
    this->nnz = _nnz;
    this->rows = _rows;
    this->cols = _cols;

    this->row_index = std::make_unique<I[]>(_nnz);
    this->col_index = std::make_unique<I[]>(_nnz);
    this->values = std::make_unique<T[]>(_nnz);
  }
};

typedef csr_mtx<int, dtype> type_csr;

typedef coo_mtx<int, dtype> type_coo;

#endif // SPMV_ACC_SPARSE_FORMAT_H
