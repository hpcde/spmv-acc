//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_UTILS_HPP
#define SPMV_ACC_UTILS_HPP

#include <iomanip>

#include "building_config.h"
#include "sparse_format.h"

#define HIP_CHECK(stat)                                                                                                \
  {                                                                                                                    \
    if ((stat) != hipSuccess) {                                                                                        \
      std::cerr << "Error: hip error in line " << __LINE__ << std::endl;                                               \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

#define ROCSPARSE_CHECK(stat)                                                                                          \
  {                                                                                                                    \
    if ((stat) != rocsparse_status_success) {                                                                          \
      std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl;                                         \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

template <typename T> void print_vector(std::vector<T> &vecTest) {
  for (auto it : vecTest) {
    std::cout << std::setprecision(10) << it << " ";
  }
  std::cout << std::endl;
}

template <typename T> void print_vector(int n, T *x) {
  for (int i = 0; i < n; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  return;
}

inline double rand_double(double min, double max) {
  double temp = min + (max - min) * double(rand() % 100) / double((101));
  return temp;
}

template <typename T> void generate_vector(int n, T *&x) {
  x = new T[n];
  for (int i = 0; i < n; i++) {
    x[i] = static_cast<T>(rand_double(-1.0, 1.0));
  }
}

template <typename T> struct host_vectors {
  T *hX;
  T *temphY; // the origin y vector
  T *hY;     // store data copied from device side for later verification.
  T *hhY;    // store the correct results produced by host or device verification.
};

void create_host_data(type_csr _csr, host_vectors<dtype> &h_vecs, bool overwrite_hx = false) {
  dtype *temp_csr_dense_vec;
  if (!overwrite_hx) {
    temp_csr_dense_vec = new dtype[_csr.cols];
    memcpy(temp_csr_dense_vec, h_vecs.hX, _csr.cols * sizeof(double)); // backup hX
  }

  generate_vector(_csr.cols, h_vecs.hX);
  generate_vector(_csr.rows, h_vecs.temphY);
  generate_vector(_csr.rows, h_vecs.hY); //实际数据依然来自随机生成
  generate_vector(_csr.rows, h_vecs.hhY);

  //统一主机端和设备端的向量值
  memcpy(h_vecs.hY, h_vecs.temphY, _csr.rows * sizeof(double));
  memcpy(h_vecs.hhY, h_vecs.temphY, _csr.rows * sizeof(double));

  if (!overwrite_hx) {
    memcpy(h_vecs.hX, temp_csr_dense_vec, _csr.cols * sizeof(double)); //外部读取数据恢复 hX
    delete[] temp_csr_dense_vec;
  }
}

void destroy_host_data(host_vectors<dtype> &h_vecs) {
  delete[] h_vecs.hhY;
  delete[] h_vecs.hY;
  delete[] h_vecs.temphY;
  delete[] h_vecs.hX;
}

type_csr create_device_data(type_csr h_csr, dtype *hX, dtype *hY, dtype *&dX, dtype *&dY) {
  const int A_num_rows = h_csr.rows;
  const int A_num_cols = h_csr.cols;
  const int A_nnz = h_csr.nnz;

  type_csr dev_csr;
  dev_csr.rows = A_num_rows;
  dev_csr.cols = A_num_cols;
  dev_csr.nnz = A_nnz;

  HIP_CHECK(hipMalloc((void **)&dev_csr.row_ptr, (A_num_rows + 1) * sizeof(int)))
  HIP_CHECK(hipMalloc((void **)&dev_csr.col_index, A_nnz * sizeof(int)))
  HIP_CHECK(hipMalloc((void **)&dev_csr.values, A_nnz * sizeof(dtype)))
  HIP_CHECK(hipMalloc((void **)&dX, A_num_cols * sizeof(dtype)))
  HIP_CHECK(hipMalloc((void **)&dY, A_num_rows * sizeof(dtype)))

  HIP_CHECK(hipMemcpy(dev_csr.row_ptr, h_csr.row_ptr, (A_num_rows + 1) * sizeof(int), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dev_csr.col_index, h_csr.col_index, A_nnz * sizeof(int), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dev_csr.values, h_csr.values, A_nnz * sizeof(dtype), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dX, hX, A_num_cols * sizeof(dtype), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dY, hY, A_num_rows * sizeof(dtype), hipMemcpyHostToDevice))
  return dev_csr;
}

void destroy_device_data(type_csr d_csr, dtype *&dX, dtype *&dY) {
  HIP_CHECK(hipFree((void *)dY))
  HIP_CHECK(hipFree((void *)dX))
  HIP_CHECK(hipFree((void *)d_csr.values))
  HIP_CHECK(hipFree((void *)d_csr.col_index))
  HIP_CHECK(hipFree((void *)d_csr.row_ptr))
}

#endif // SPMV_ACC_UTILS_HPP
