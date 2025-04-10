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
    // read header: magic number
    int32_t magic_num;
    fin.read((BytePtr)(&(magic_num)), sizeof(magic_num));
    if (magic_num != 0x20211015) {
      std::cerr << "read file failed with mismatch magic number, file:" << mtx_path << std::endl;
      return;
    }
    // read header: binary format
    int32_t bin_format;
    fin.read((BytePtr)(&(bin_format)), sizeof(bin_format));
    if (bin_format != 0x2) {
      std::cerr << "we only support bin file version 2, bin file:" << mtx_path << std::endl;
      return;
    }
    // read header: value type
    typedef int32_t tp_val_tpye;
    tp_val_tpye val_type;
    fin.read((BytePtr)(&(val_type)), sizeof(tp_val_tpye));
    constexpr tp_val_tpye TP_BOOL = 1;
    constexpr tp_val_tpye TP_INT = 2;
    constexpr tp_val_tpye TP_FLOAT = 3;
    constexpr tp_val_tpye TP_COMPLEX = 4;
    if (val_type != TP_BOOL && val_type != TP_INT && val_type != TP_FLOAT && val_type != TP_COMPLEX) {
      std::cerr << "matrix value type not supported, bin file:" << mtx_path << std::endl;
      return;
    }

    // read csr matrix
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

    // read csr matrix: value part
    if (val_type == TP_BOOL) {
      for (int k = 0; k < P::nnz; k++) {
        P::values[k] = 1.0;
      }
    } else if (val_type == TP_INT) {
      int32_t *temp_int = new int32_t[P::nnz];
      fin.read((BytePtr)(temp_int), sizeof(T) * P::nnz);
      for (int k = 0; k < P::nnz; k++) {
        P::values[k] = temp_int[k];
      }
      delete[] temp_int;
    } else { // float or complex, read directly.
      fin.read((BytePtr)(P::values), sizeof(T) * P::nnz);
    }
    fin.close();
  }

  void close_stream() {
    // nothing to do, just keep the same api as csr text reader
  }
};

#endif // SPMV_ACC_CSR_BINARY_READER_HPP
