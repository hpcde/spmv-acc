#include <iomanip>
#include <iostream>
#include <map>
#include <cstdio>  // printf
#include <cstdlib> // EXIT_FAILURE
#include <cstring>
#include <type_traits>
#include <vector>

#include "global_data.h"

template <typename T> void init_csr_dense_vector(char *buf, std::vector<T> &vec) {
  char *p;
  p = strtok(buf, " ");
  while (p != nullptr) {
    bool isInt = std::is_same<T, int>::value;
    if (isInt)
      vec.push_back(atoi(p));
    else
      vec.push_back(atof(p));
    p = strtok(nullptr, " ");
  }
}

bool read_file(char *path) {
  FILE *in = fopen(path, "r");
  if (in == nullptr) {
    return false;
  }
  constexpr long buffer_size = 1024L * 1024L * 1024L * 5L;
  char *buf = new char[buffer_size]; // 5g buff
  int line_num = 0;
  while (fgets(buf, buffer_size, in) != nullptr) {
    if (line_num == 1) {
      init_csr_dense_vector<double>(buf, csr_data);
    }
    if (line_num == 2)
      init_csr_dense_vector<int>(buf, csr_indices);
    if (line_num == 3)
      init_csr_dense_vector<int>(buf, csr_indptr);
    if (line_num == 4)
      init_csr_dense_vector<double>(buf, dense_vector);
    line_num++;
  }
  delete[] buf;
  fclose(in);
  return true;
}

enum SubCmd { nnz = 0, dist = 1 };

int main(int argc, char **argv) {
  const std::string usage = R"""(Usage:
  csv-reader nnz [file-path]
  csv-reader dist [file-path]
  csv-reader --help
)""";
  if (argc < 2) {
    std::cout << usage;
    return 0;
  }
  const std::string sub_cmd_str = std::string(argv[1]);
  if (sub_cmd_str == "--help") {
    std::cout << usage;
    return 0;
  }
  SubCmd sub_cmd;
  if (sub_cmd_str == "nnz") {
    sub_cmd = nnz;
  } else if (sub_cmd_str == "dist") {
    sub_cmd = dist;
  } else {
    std::cerr << "unknown sub-command " << sub_cmd_str << std::endl;
    std::cout << usage;
    return 1;
  }
  if (argc < 3) {
    std::cerr << "missing file path." << std::endl;
    return 1;
  }

  char path[1024];
  strcpy(path, argv[2]);
  if (!read_file(path)) {
    std::cerr << "reading file error." << std::endl;
    return 1;
  }

  int m = csr_indptr.size() - 1;
  int n = dense_vector.size();

  if (sub_cmd == nnz) {
    bool is_first = true;
    int last_row = 0;
    for (int row_offset : csr_indptr) {
      if (!is_first) {
        std::cout << row_offset - last_row << std::endl;
      }
      last_row = row_offset;
      is_first = false;
    }
  } else {
    bool is_first = true;
    int last_row = 0;
    std::map<int, int> K;
    for (int row_offset : csr_indptr) {
      if (!is_first) {
        const int diff = row_offset - last_row;
        //      std::cout << row_offset - last_row << std::endl;
        if (K.find(diff) != K.end()) {
          K[diff] += 1;
        } else {
          K[diff] = 1;
        }
      }
      last_row = row_offset;
      is_first = false;
    }

    for (const auto &[key, value] : K) {
      std::cout << key << " = " << value << "\n";
    }
  }
  return 0;
}
