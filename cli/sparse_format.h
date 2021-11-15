//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_SPARSE_FORMAT_H
#define SPMV_ACC_SPARSE_FORMAT_H

#include <memory>

#include "api/types.h"

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
  bool operator<(const Entry &other) {
    if (r != other.r) {
      return r < other.r;
    }
    return c < other.c;
  }
};

// COO sparse matrix type
template <typename I, typename T> class coo_mtx {
  typedef Entry<I, T> coo_entry;

public:
  I rows = 0;
  I cols = 0;
  I nnz = 0;

  std::unique_ptr<I[]> row_index = nullptr;
  std::unique_ptr<I[]> col_index = nullptr;
  std::unique_ptr<T[]> values = nullptr;

  // convert to csr matrix format
  csr_mtx<I, T> to_csr() {
    csr_mtx<I, T> csr;
    std::vector<coo_entry> entries;
    entries.reserve(this->nnz);
    for (size_t i = 0; i < this->nnz; i++) {
      entries.push_back(coo_entry{this->row_index[i], this->col_index[i], this->values[i]});
    }
    // sort by row id, then column id
    std::sort(std::begin(entries), std::end(entries));

    csr.alloc(this->rows, this->cols, this->nnz);
    memset(csr.row_ptr, 0, (this->rows + 1) * sizeof(I));
    for (size_t i = 0; i < this->nnz; i++) {
      csr.values[i] = entries[i].v;
      csr.col_index[i] = entries[i].c;
      ++csr.row_ptr[entries[i].r + 1]; // this line of code only set nnz of current row
    }

    for (size_t i = 0; i < this->rows; i++) {
      const unsigned int nnz_this_row = csr.row_ptr[i + 1];
      csr.row_ptr[i + 1] = csr.row_ptr[i] + nnz_this_row;
    }
    return csr;
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
