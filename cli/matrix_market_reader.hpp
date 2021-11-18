//
// Created by chu genshen on 2021/10/22.
//

#ifndef SPMV_ACC_MATRIX_MARKET_LOADER_H
#define SPMV_ACC_MATRIX_MARKET_LOADER_H

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sparse_format.h"

namespace {
  template <typename VALUE_TYPE> struct DataTypeValidator {
    static const bool validate(std::string type) { return false; }
  };

  template <> struct DataTypeValidator<float> {
    static const bool validate(std::string type) { return type.compare("real") == 0 || type.compare("integer") == 0; }
  };
  template <> struct DataTypeValidator<double> {
    static const bool validate(std::string type) { return type.compare("real") == 0 || type.compare("integer") == 0; }
  };

  template <> struct DataTypeValidator<int> {
    static const bool validate(std::string type) { return type.compare("integer") == 0; }
  };
} // namespace

/**
 * head information of matrix market format.
 */
struct mm_header {
  std::size_t num_rows;
  std::size_t num_columns;
  std::size_t num_non_zeroes;
  bool pattern;
  bool hermitian;
  bool complex;
  bool symmetric;
};

struct body_line {
  std::size_t line_num;
  std::string line;
};

/**
 * This class read COO format sparse matrix from file.
 * todo: currently, we only support float point number (does not support integer).
 * @tparam I type of integer
 * @tparam T type of float point number data
 * @note: the implementation of this function is copied and modified from
 * https://bitbucket.org/gpusmack/holaspmv/src/master/source/COO.cpp (MIT license).
 */
template <typename I, typename T> class matrix_market_reader {
public:
  coo_mtx<I, T> load_mat(std::string file) {
    std::ifstream fstream(file);
    if (!fstream.is_open()) {
      throw std::runtime_error(std::string("could not open \"") + file + "\"");
    }

    std::size_t line_counter = 0;
    // read and parse header of matrix market format.
    mm_header header = this->parse_header(fstream, file, line_counter);

    std::size_t reserve = header.num_non_zeroes;
    if (header.symmetric || header.hermitian) {
      reserve *= 2;
    }

    coo_mtx<I, T> res_matrix;
    // the nnz passed to alloc can be larger than the real fact when the matrix is symmetric.
    res_matrix.alloc(header.num_rows, header.num_columns, reserve);

    // get length of file
    const std::size_t body_length = this->body_length(fstream);
    char *body_buffer = new char[body_length + 1];
    body_buffer[body_length] = '\0'; // set a null character at the end
    // read data of body as a block
    fstream.read(body_buffer, body_length);
    fstream.close();

    size_t read = 0;
    size_t nnz_dia = 0;

#ifndef _OPENMP
    char *p = strtok(body_buffer, "\n");
    while (p != NULL) {
      ++line_counter;
      parse_line(file, res_matrix, header, coo_lines_vec[i], line_counter, nnz_dia, read);
      p = strtok(NULL, "\n");
    }
#endif // _OPENMP
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    std::cout << "Parsing input using " << max_threads << " OpenMP thread(s)." << std::endl;

    std::vector<char *> coo_lines_vec;
    char *p;
    p = strtok(body_buffer, "\n");
    while (p != NULL) {
      coo_lines_vec.emplace_back(p);
      p = strtok(NULL, "\n");
    }

    std::size_t N = coo_lines_vec.size();
    char **body_lines_data = coo_lines_vec.data();
#pragma omp parallel shared(body_lines_data, read, N) reduction(+ : nnz_dia)
    {
#pragma omp for schedule(static)
      for (std::size_t i = 0; i < N; i++) {
        const std::size_t line_num = line_counter + i + 1;
        char *local_line = body_lines_data[i];
        parse_line(file, res_matrix, header, local_line, line_num, nnz_dia, read);
      }
    }
#endif // _OPENMP

    delete[] body_buffer;
    res_matrix.nnz = read;

    // assert read count.
    if (read + nnz_dia != reserve) {
      throw std::runtime_error("mismatch non-zeros number, expect " + std::to_string(reserve) + ", but got " +
                               std::to_string(read));
    }

    return res_matrix;
  }

private:
  mm_header parse_header(std::ifstream &fstream, const std::string &file, std::size_t &line_counter) {
    std::string line;

    // parse the first line
    std::getline(fstream, line);
    if (line.compare(0, 32, "%%MatrixMarket matrix coordinate") != 0) {
      throw std::runtime_error("Can only read MatrixMarket format that is in coordinate form");
    }

    std::istringstream iss(line);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

    mm_header header{
        .num_rows = 0,
        .num_columns = 0,
        .num_non_zeroes = 0,
        .pattern = false,
        .hermitian = false,
        .complex = false,
        .symmetric = false,
    };

    if (tokens[3] == "pattern") {
      header.pattern = true;
    } else if (tokens[3] == "complex") {
      header.complex = true;
    } else if (DataTypeValidator<T>::validate(tokens[3]) == false) {
      throw std::runtime_error("MatrixMarket data type does not match matrix format");
    }

    if (tokens[4].compare("general") == 0) {
      header.symmetric = false;
    } else if (tokens[4].compare("symmetric") == 0) {
      header.symmetric = true;
    } else if (tokens[4].compare("Hermitian") == 0) {
      header.hermitian = true;
    } else {
      throw std::runtime_error("Can only read MatrixMarket format that is either symmetric, general or hermitian");
    }

    // skip comments and read metadata.
    while (std::getline(fstream, line)) {
      ++line_counter;
      // skip header
      if (line[0] == '%') {
        continue;
      }
      std::istringstream liness(line);
      liness >> header.num_rows >> header.num_columns >> header.num_non_zeroes;
      if (liness.fail()) {
        throw std::runtime_error(std::string("Failed to read matrix market header from \"") + file + "\"");
      }
      // std::cout << "Read matrix header" << std::endl;
      // std::cout << "rows: " << rows << " columns: " << columns << " nnz: " << nnz << std::endl;
      break;
    }
    return header;
  }

  std::size_t body_length(std::ifstream &fstream) {
    const long body_pos = fstream.tellg();
    fstream.seekg(0, fstream.end); // seek to end of file
    const long body_length = std::static_cast<long>(fstream.tellg()) - body_pos;
    fstream.seekg(body_pos, fstream.beg); // reset position
    return std::static_cast<std::size_t>(body_length);
  }

  /**
   * parse a line of matrix body
   */
  static void parse_line(const std::string &file, coo_mtx<I, T> &res_matrix, const mm_header header, char *line,
                         const std::size_t &line_counter, std::size_t &nnz_dia, std::size_t &read) {
    if (line[0] == '%') {
      return;
    }

    std::istringstream liness(line);

    // trim prefix space
    do {
      char ch;
      liness.get(ch);
      if (!isspace(ch)) {
        liness.putback(ch);
        break;
      }
    } while (!liness.eof());
    if (liness.eof() || line.length() == 0) {
      return;
    }

    // parse a line for COO
    uint32_t r, c;
    T value;
    liness >> r >> c;
    if (header.pattern) {
      value = 1;
    } else {
      liness >> value;
    }
    if (liness.fail()) {
      throw std::runtime_error(std::string("Failed to read data at line ") + std::to_string(line_counter) +
                               " from matrix market file \"" + file + "\"");
    }
    if (r > header.num_rows) {
      throw std::runtime_error(std::string("Row index out of bounds at line  ") + std::to_string(line_counter) +
                               " in matrix market file \"" + file + "\"");
    }
    if (c > header.num_columns) {
      throw std::runtime_error(std::string("Column index out of bounds at line  ") + std::to_string(line_counter) +
                               " in matrix market file \"" + file + "\"");
    }

    std::size_t index;
#ifdef _OPENMP
#pragma omp atomic capture
#endif
    {
      index = read;
      read++;
    }
    res_matrix.row_index[index] = r - 1;
    res_matrix.col_index[index] = c - 1;
    res_matrix.values[index] = value;
    if ((header.symmetric || header.hermitian) && r == c) {
      nnz_dia++;
    }
    if ((header.symmetric || header.hermitian) && r != c) {
      std::size_t index2;
#ifdef _OPENMP
#pragma omp atomic capture
#endif
      {
        index2 = read;
        read++;
      }
      res_matrix.row_index[index2] = c - 1;
      res_matrix.col_index[index2] = r - 1;
      res_matrix.values[index2] = value;
    }
  }
};

#endif // SPMV_ACC_MATRIX_MARKET_LOADER_H
