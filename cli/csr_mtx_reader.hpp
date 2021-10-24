//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_CSR_MTX_READER_HPP
#define SPMV_ACC_CSR_MTX_READER_HPP

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sparse_format.h"

/**
 * This class read csr format sparse matrix from file to memory.
 * @tparam I type of integer
 * @tparam T type of float number data
 */
template <typename I, typename T> class csr_mtx_reader {
public:
  std::vector<T> csr_data;     // matrix data
  std::vector<I> csr_indices;  // column index
  std::vector<I> csr_indptr;   // row ptr
  std::vector<T> dense_vector; // vector x

  static constexpr unsigned int MAX_VEC_LEN = 1 << 23; // 8,388,608
  static constexpr unsigned int MAX_NNZ_LEN = 1 << 26; // 67,108,864

  /**
   * init matrix reader with file path.
   */
  explicit csr_mtx_reader(const std::string &file_path) {
    csr_mtx_file.open(file_path, std::ios::in);
    if (!(csr_mtx_file.good())) {
      throw std::runtime_error("Open file failed");
    }
  }

  /**
   * fill the matrix using data in matrix file
   */
  void fill_mtx() {
#ifndef _OPENMP
    // If OpenMP is not enabled, set max vector capacity. otherwise, vector resize will be performed.
    csr_data.reserve(MAX_NNZ_LEN);
    csr_indices.reserve(MAX_NNZ_LEN);
    csr_indptr.reserve(MAX_VEC_LEN);
    dense_vector.reserve(MAX_VEC_LEN);
#endif

    // set buffer for faster reading.
    char stream_buf[128 * 1024];
    csr_mtx_file.rdbuf()->pubsetbuf(stream_buf, sizeof(stream_buf));

    // read all bytes into memory.
    std::streamsize file_size = csr_mtx_file.seekg(0, std::ios::end).tellg();
    char *file_buffer = new char[file_size + 1];
    file_buffer[file_size] = '\0'; // set a null character at the end
    csr_mtx_file.seekg(0, std::ios::beg).read(&file_buffer[0], file_size);

    // convert memory to stream again
    std::stringstream ss(file_buffer);

    char *line_buffer = new char[file_size];
    int line_num = 0;
    while (ss.getline(line_buffer, file_size)) {
      if (line_num == 1) {
        fast_parse_vector<T>(line_buffer, csr_data);
      }
      if (line_num == 2) {
        fast_parse_vector<I>(line_buffer, csr_indices);
      }
      if (line_num == 3) {
        fast_parse_vector<I>(line_buffer, csr_indptr);
      }
      if (line_num == 4) {
        fast_parse_vector<T>(line_buffer, dense_vector);
      }
      line_num++;
    }

    delete[] file_buffer;
    delete[] line_buffer;
  }

  void close_stream() {
    if (csr_mtx_file.good()) {
      csr_mtx_file.close();
    }
  }

  // convert vector data to raw pointer.
  void as_raw_ptr(T *&_value, I *&_col_index, I *&_row_ptr, T *&_dense_vector) {
    _value = this->csr_data.data();
    _col_index = this->csr_indices.data();
    _row_ptr = this->csr_indptr.data();
    _dense_vector = this->dense_vector.data();
  }

  inline int nnz() { return csr_data.size(); }

  inline int rows() { return csr_indptr.size() - 1; }

  inline int cols() { return dense_vector.size(); }

private:
  std::ifstream csr_mtx_file;

#ifdef _OPENMP
  // OpenMP version
  template <typename M> void fast_parse_vector(char *buffer, std::vector<M> &data) {
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    std::cout << "Parsing input using " << max_threads << " OpenMP thread(s)." << std::endl;

    std::vector<char *> number_vec;
    char *p;
    p = strtok(buffer, " ");
    while (p != NULL) {
      number_vec.emplace_back(p);
      p = strtok(NULL, " ");
    }

    const unsigned int N = number_vec.size();
    data.resize(N);
    char **number_ptr = number_vec.data();
    M *data_ptr = data.data();

#pragma omp parallel for shared(data_ptr, number_ptr, N) schedule(static)
    for (int i = 0; i < N; ++i) {
      bool isInt = std::is_same<T, int>::value;
      if (isInt) {
        data_ptr[i] = atoi(number_ptr[i]);
      } else {
        data_ptr[i] = atof(number_ptr[i]);
      }
    }
  }
#endif

#ifndef _OPENMP
  template <typename M> void fast_parse_vector(char *buffer, std::vector<M> &data) {
    char *p;
    p = strtok(buffer, " ");
    while (p != NULL) {
      bool isInt = std::is_same<T, int>::value;
      if (isInt) {
        data.emplace_back(atoi(p));
      } else {
        data.emplace_back(atof(p));
      }
      p = strtok(NULL, " ");
    }
  }
#endif

  template <typename M> void parse_vector(const char *buffer, std::vector<M> &data) {
    std::istringstream iss(buffer);
    M tmp;
    while (iss) {
      iss >> tmp;
      data.emplace_back(tmp);
    }
  }
};

#endif // SPMV_ACC_CSR_MTX_READER_HPP
