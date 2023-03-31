//
// Created by genshen on 2021/11/28.
//

#ifndef SPMV_ACC_CSR_BINARY_READER_HPP
#define SPMV_ACC_CSR_BINARY_READER_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <api/types.h>

/**
 * Class `csr_binary_reader` read csr binary format sparse matrix from file to memory.
 * @tparam I type of integer
 * @tparam T type of float number data
 */
template <typename I, typename T> class csr_binary_reader : public var_csr_desc<I, T> {
  typedef var_csr_desc<I, T> P;
  typedef char *BytePtr;

public:
  I rows() { return P::rows; }
  I cols() { return P::cols; }
  I nnz() { return P::nnz; }

  void as_raw_ptr(T *&_value, I *&_col_index, I *&_row_ptr) {
    _value = P::values;
    _col_index = P::col_index;
    _row_ptr = P::row_ptr;
  }

  void load_mat(const std::string &mtx_path) {
    std::ifstream fin(mtx_path, std::ios::in | std::ios::binary);
    if (!fin.good()) {
      std::cerr << "file open failed, file: " << mtx_path << std::endl;
      return;
    }
    I &_r = P::rows;
    I &_c = P::cols;
    I &_nnz = P::nnz;
    fin.read((BytePtr)(&(_r)), sizeof(I)); // todo: make sure storage type is 4 bytes.
    fin.read((BytePtr)(&(_c)), sizeof(I));
    fin.read((BytePtr)(&(_nnz)), sizeof(I));

    P::row_ptr = new I[P::rows + 1];
    P::col_index = new I[P::nnz];
    P::values = new T[P::nnz];

    fin.read((BytePtr)(P::row_ptr), sizeof(I) * (P::rows + 1));
    fin.read((BytePtr)(P::col_index), sizeof(I) * P::nnz);
    fin.read((BytePtr)(P::values), sizeof(T) * P::nnz);
    fin.close();
  }

  void close_stream() {
    // nothing to do, just keep the same api as csr text reader
  }
};

#endif // SPMV_ACC_CSR_BINARY_READER_HPP
