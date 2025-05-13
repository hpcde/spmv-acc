#define ACSR_STDERR

#include "../utils/benchmark_time.h"
#include "api/types.h"
#include "timer.h"
#include "ACSR.h"
#include <vector>

#define BIN_MAX 30
#define ROW_MAX 1024
#define THREAD_LOAD 5

#define __shfl_down(X, Y) __shfl_down_sync(0xFFFFFFFFu, X, Y, 32)

template<class T>
__inline__ __device__ T warpReduceSum(T val)
{
	val += __shfl_down(val, 16);
	val += __shfl_down(val, 8);
	val += __shfl_down(val, 4);
	val += __shfl_down(val, 2);
	val += __shfl_down(val, 1);
	return val;
}

template<class T>
__inline__ __device__ T blockReduceSum(T val)
{

	static __shared__ T shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val); // Each warp performs partial reduction

	if (lane == 0)
		shared[wid] = val; // Write reduced value to shared memory

	__syncthreads(); // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / 32.0) ? shared[lane] : 0;

	if (wid == 0)
		val = warpReduceSum(val); //Final reduce within first warp

	return val;
}

template<class T>
__global__ void spmv(int alpha, T *__restrict__ values, const int *__restrict__ col_idx, const int *__restrict__ row_off, T *__restrict__ vect,
					 T *__restrict__ res, int m, int n, const int *__restrict__ bin, int bin_size, int N, int nnz)
{
	int tid = threadIdx.x;
	T sum = 0;
	int row = bin[blockIdx.x];
	int row_idx = row_off[row];
	int next_row_idx;

	next_row_idx = row_off[row + 1];

	for (int i = row_idx + tid; i < next_row_idx; i += blockDim.x)
	{
		// sum += values[i] * tex1Dfetch<float>(vect, col_idx[i]); //vect[col_idx[i]];
		sum += values[i] * vect[col_idx[i]];
	}

	sum = blockReduceSum(sum);

	if (tid == 0)
		res[row] = sum * alpha;
}
////////////////////////////////////////////////////////////////////////////////////
// Kernel for dynamic parallelism
// flag -rdc=true should be set if this kernel is called
////////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void row_specific_spmv(int alpha, T *__restrict__ values, int *__restrict__ col_idx, int *__restrict__ row_off,
								  T *__restrict__ x, T *__restrict__ res, int m, int n, int nnz, int row, int noOfThreads)
{
	int tid = threadIdx.x;
	int lid = tid % 32;
	int vid = tid / 32;

	T sum = 0;

	int row_idx = row_off[row];
	int next_row_idx;
	if (row < (m - 1))
		next_row_idx = row_off[row + 1];
	else
		next_row_idx = nnz;
	for (int i = row_idx + tid; i < next_row_idx; i += noOfThreads)
	{
		sum += values[i] * x[col_idx[i]];
	}

	__syncthreads();

	sum = blockReduceSum(sum);

	if (lid == 0 && vid == 0)
		res[row] = sum * alpha;
}
template<class T>
__global__ void dynamicParallelParent(int alpha, T *__restrict__ values, int *__restrict__ col_idx, int *__restrict__ row_off, T *__restrict__ x,
									  T *__restrict__ res, int m, int n, int nnz, int *__restrict__ G1, int G1_size)
{
	int tid = threadIdx.x;
	// printf("threadIdx = %d\n",tid);
	int row = G1[tid];
	int row_idx = row_off[row];
	int next_row_idx;
	if (row == m - 1)
	{
		next_row_idx = nnz;
	}
	else
		next_row_idx = row_off[row + 1];

	int NNZ = next_row_idx - row_idx;
	int bsize = (NNZ - 1) / THREAD_LOAD + 1;

	row_specific_spmv<<<1, bsize>>>(alpha, values, col_idx, row_off, x, res, m, n, nnz, row, bsize);
}

////////////////////////////////////////////////////////////////////////////////////

int calc_bin_index(int nnz)
{
	if (nnz == 0 | nnz == 1)
		return nnz;

	int cnt = 0, orig_nnz = nnz;
	while (nnz > 0)
	{
		nnz >>= 1;
		cnt++;
	}
	if (!(orig_nnz & (orig_nnz - 1)))
		return cnt - 1;
	else
		return cnt;
}

// Matrix : m x n
// Vector : n x 1
template<class T>
void acsr_driver(int alpha, T *values, int *row_off, int* d_col_idx, int * d_row_off, T *x, T *y, int m, int n, int nnz,BenchmarkTime *bmt)
{
	hip::timer::event_timer pre_timer, calc_timer, destroy_timer;

	pre_timer.start();
	cudaMemset(y, 0, n * sizeof(T));
	int max_nnz = INT_MIN;
	for (int i = 1; i < m; i++){
		max_nnz = max(max_nnz, row_off[i] - row_off[i - 1]);
	}
	max_nnz = max(max_nnz, m - row_off[m - 1]);
	int max_bins = calc_bin_index(max_nnz);
	std::vector<int> bins[max_bins + 1];

	for (int i = 1; i < m; i++)
	{
		int nnz = row_off[i] - row_off[i - 1];
		int bin_index = calc_bin_index(nnz);
		bins[bin_index].push_back(i - 1);
	}

	int last_nnz = nnz - row_off[m - 1];
	bins[calc_bin_index(last_nnz)].push_back(m - 1);
	pre_timer.stop();
	double pre_time_use = 0,calc_time_use = 0;
	pre_time_use += pre_timer.time_use;
	//Calculate G2
	for (int i = 1; i <= min(max_bins, BIN_MAX); i++)
	{
		if (bins[i].size() > 0)
		{
			pre_timer.start();
			int *dbin;
			cudaMalloc((void **)&dbin, bins[i].size() * sizeof(int));

			int arr[bins[i].size()]; //Temporary array to store a single bin
			for (int j = 0; j < bins[i].size(); j++)
				arr[j] = bins[i][j];

			cudaMemcpy(dbin, arr, (bins[i].size()) * sizeof(int), cudaMemcpyHostToDevice);

			int dimBlock = (1 << (i - 1));
			if (dimBlock > 1024)
				dimBlock = 1024;
			dim3 dimGrid(bins[i].size());
			pre_timer.stop();
			pre_time_use += pre_timer.time_use;
			calc_timer.start();
			cudaDeviceSynchronize();
			spmv<<<dimGrid, dimBlock>>>(alpha, values, d_col_idx, d_row_off, x, y, m, n, dbin, bins[i].size(), i, nnz);
			cudaDeviceSynchronize();
			calc_timer.stop();

			cudaFree(dbin);

			calc_time_use += calc_timer.time_use;
		}
	}
	pre_timer.start();
	int *G1, *dG1;
	G1 = (int *)malloc(sizeof(int) * (m));
	int no_of_bigrows = 0;
	for (int i = BIN_MAX + 1; i <= max_bins; i++)
	{
		for (int j = 0; j < bins[i].size(); j++)
		{
			G1[no_of_bigrows++] = bins[i][j];
		}
	}

	cudaMalloc((void **)&dG1, (no_of_bigrows) * sizeof(int));
	cudaMemcpy(dG1, G1, no_of_bigrows * sizeof(int), cudaMemcpyHostToDevice);
	pre_timer.stop();
	pre_time_use+=pre_timer.time_use;
	calc_timer.start();
	cudaDeviceSynchronize();
	dynamicParallelParent<<<1,no_of_bigrows>>>(alpha, values, d_col_idx, d_row_off, x, y, m, n, nnz, dG1, no_of_bigrows);
	cudaDeviceSynchronize();
	calc_timer.stop();
	calc_time_use += calc_timer.time_use;

	destroy_timer.start();
	free(G1);
	cudaFree(dG1);
	destroy_timer.stop();
	if (bmt != nullptr) {
    bmt->set_time(pre_time_use, calc_time_use, destroy_timer.time_use);
  }
}


void acsr(int trans, const int alpha, const csr_desc<int, double> h_csr_desc, const csr_desc<int, double> d_csr_desc, const double *x,
          double *y, BenchmarkTime *bmt) {
    double *d_values = const_cast<double *>(d_csr_desc.values);
    double *d_x = const_cast<double *>(x);
    double *d_y = const_cast<double *>(y);
    int *h_row_ptr = const_cast<int *>(h_csr_desc.row_ptr);
    int *d_row_ptr = const_cast<int *>(d_csr_desc.row_ptr);
    int *d_col_index = const_cast<int *>(d_csr_desc.col_index);
    int rows = h_csr_desc.rows;
    int cols = h_csr_desc.cols;
    int nnz = h_csr_desc.nnz;
    acsr_driver(alpha, d_values, h_row_ptr, d_col_index, d_row_ptr, d_x, d_y, rows, cols, nnz, bmt);
}

