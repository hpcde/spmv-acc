#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <type_traits>
#include <vector>

#include "clipp.h"
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

bool read_file(const char *path) {
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

// cli sub commands
enum class mode { nnz, dist, version, help };

void csr_analyzing(const mode selected, const std::string &csr_path, const std::string &csr_parts);

int main(int argc, char **argv) {
  mode selected = mode::help;

  std::string csr_path;
  std::string csr_parts;
  auto mode_nnz = (clipp::command("nnz").set(selected, mode::nnz),
                   clipp::required("-i", "--input").label("--input") & clipp::value("csr file", csr_path),
                   clipp::option("-p", "--parts") & clipp::value("dividing parts", csr_parts))
                      .doc("nnz of rows.");

  auto mode_dist = (clipp::command("dist").set(selected, mode::dist),
                    clipp::required("-i", "--input").label("--input") & clipp::value("csr file", csr_path))
                       .doc("nnz distribution.");

  std::vector<std::string> wrong_args;
  auto cli =
      ((mode_nnz | mode_dist | clipp::command("-h", "--help").set(selected, mode::help).doc("Show this help message.") |
        clipp::command("-v", "--version").set(selected, mode::version).doc("Display version.")),
       clipp::any_other(wrong_args));

  if (argc > 1) {
    // parse command line
    clipp::parsing_result result = parse(argc, argv, cli);
    // if parsing error
    if (!wrong_args.empty()) {
      for (const auto &arg : wrong_args) {
        std::cerr << "'" << arg << "' is not a valid argument.\n";
        return 1;
      }
    }
    if (result.any_error()) {
      for (const auto &m : result.missing()) {
        std::cerr << "Error: missing " << m.param()->label() << " after index " << m.after_index() << ".\n";
        return 1;
      }
      // per-argument mapping
      for (const auto &m : result) {
        std::cerr << "Error: bad argument at " << m.index() << ": " << m.arg() << " -> " << m.param()->label();
        std::cerr << '\n';
        return 1;
      }
    }
  }

  switch (selected) {
  case mode::nnz:
    csr_analyzing(selected, csr_path, csr_parts);
    return 0;
  case mode::dist:
    csr_analyzing(selected, csr_path, csr_parts);
    return 0;
  case mode::help:
    std::cout << make_man_page(cli, "csr-tool").prepend_section("DESCRIPTION", "csr analyzing tool.");
    return 0;
  case mode::version:
    std::cout << "version 0.1.0" << std::endl
              << "compiled at " << __TIME__ << ", " << __DATE__ << "." << std::endl
              << "Copyright (C) 2021 USTB." << std::endl;
    return 0;
  }
  return 0;
}

void csr_analyzing(const mode sub_cmd, const std::string &csr_path, const std::string &csr_parts) {
  if (!read_file(csr_path.c_str())) {
    std::cerr << "reading file error." << std::endl;
    return;
  }

  int m = csr_indptr.size() - 1;
  int n = dense_vector.size();

  if (sub_cmd == mode::nnz) {
    bool is_first = true;
    int last_row = 0;
    for (int row_offset : csr_indptr) {
      if (!is_first) {
        std::cout << row_offset - last_row << std::endl;
      }
      last_row = row_offset;
      is_first = false;
    }
  }

  if (sub_cmd == mode::dist) {
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
}
