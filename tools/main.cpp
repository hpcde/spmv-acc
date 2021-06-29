#include <iomanip>
#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE
#include <string.h>
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

void read_file(char *path) {
  FILE *in = fopen(path, "r");
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
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s [file-path]\n", argv[0]);
    return 0;
  }
  char path[1024];
  strcpy(path, argv[1]);
  read_file(path);

  long i = 0;
  int last_row = 0;
  int last_diff = 0;
  for (int row_offset : csr_indptr) {
    if (i != 0) {
      std::cout << row_offset - last_row << std::endl;
    }
    last_diff = row_offset - last_row;
    last_row = row_offset;
    i++;
  }
  return 0;
}
