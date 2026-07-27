#include <iostream>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <vector>
#include "../utils/matrix_dim.h"
#include "../utils/datatype.h"
#include "../utils/file_parser.h"
#include "./../utils/vector_gen.h"
#include "../utils/create_file.h"
#include "../utils/format/csr_format.h"
#include "../utils/format/ellpack_format.h"
#include "./../CPU/hybrid_ell-csr.h"

#include "./utils/err.hpp"
using namespace std;

template <typename T>
struct Matrix
{
    T *data;
    int32_t rows;
    int32_t cols;

    __host__ __device__ T &operator()(int32_t row, int32_t col)
    {
        return data[(col * rows) + row];
    }

    __host__ __device__ const T &operator()(int32_t row, int32_t col) const
    {
        return data[(col * rows) + row];
    }
};

__global__ void spmv_kernel_C(
    const int32_t *__restrict__ d_rptr,
    const int32_t *__restrict__ d_ind,
    const double *__restrict__ d_vals,
    const int32_t nrows,
    const double *__restrict__ x,
    double *y)
{
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < nrows; i += stride)
    {
        double sum = 0.0;

        int32_t r_start = d_rptr[i];
        int32_t r_end = d_rptr[i + 1];
        for (int32_t j = r_start; j < r_end; ++j)
            sum += d_vals[j] * x[d_ind[j]];

        atomicAdd(&y[i], sum);
    }
}

__global__ void spmv_kernel_E(
    const Matrix<double> A,
    const Matrix<int32_t> J,
    const double *__restrict__ x,
    double *y)
{
    // avoid shared memory and tiling: no reuse of x values
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < J.rows; i += stride)
    {
        double sum = 0.0;
        for (int32_t j = 0; j < J.cols; ++j)
        {
            // coalesced access : columns major form
            int32_t col = J(i, j);
            if (col < 0)
                continue;
            double val = A(i, j);
            sum += val * x[col];
        }

        atomicAdd(&y[i], sum);
    }
}

void Hybrid(const int32_t *__restrict__ d_rptr,
            const int32_t *__restrict__ d_ind,
            const double *__restrict__ d_vals,
            int32_t nrows_csr,
            double *x,
            Matrix<double> A,
            Matrix<int32_t> J,
            int32_t nrows_ell,
            double *y,
            int32_t threads,
            cudaStream_t stream1,
            cudaStream_t stream2)
{
    int32_t blocks_csr = (nrows_csr + threads - 1) / threads;
    int32_t blocks_ell = (nrows_ell + threads - 1) / threads;
    spmv_kernel_C<<<blocks_ell, threads, 0, stream1>>>(d_rptr, d_ind, d_vals, nrows_csr, x, y);
    CUDA_CHECK();
    spmv_kernel_E<<<blocks_csr, threads, 0, stream2>>>(A, J, x, y);
    CUDA_CHECK();
}

double spmv_csr(
    const int32_t threads,
    const int32_t blocks,
    const int32_t *d_rptr,
    const int32_t *d_ind,
    const double *d_vals,
    const int32_t numrows,
    const double *x,
    double *y)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    spmv_kernel_C<<<blocks, threads>>>(d_rptr, d_ind, d_vals, numrows, x, y);
    CUDA_CHECK();

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float nanoseconds = 0;
    cudaEventElapsedTime(&nanoseconds, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return (double)nanoseconds;
}

double spmv_ell(
    const int32_t threads,
    const int32_t blocks,
    const Matrix<double> A,
    const Matrix<int32_t> J,
    const double *d_x,
    double *d_y)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    spmv_kernel_E<<<blocks, threads>>>(A, J, d_x, d_y);
    CUDA_CHECK();

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float nanoseconds = 0;
    cudaEventElapsedTime(&nanoseconds, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return (double)nanoseconds;
}

double spmv_hybd(
    const int32_t threads,
    const int32_t blocks,
    const int32_t *d_rptr,
    const int32_t *d_ind,
    const double *d_vals,
    Matrix<double> A,
    Matrix<int32_t> J,
    int32_t ell_rows,
    int32_t csr_rows,
    const double *x,
    double *y,
    cudaStream_t stream1,
    cudaStream_t stream2)
{

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int32_t blocks_csr = (csr_rows + threads - 1) / threads;
    int32_t blocks_ell = (ell_rows + threads - 1) / threads;
    spmv_kernel_C<<<blocks_csr, threads, 0, stream1>>>(d_rptr, d_ind, d_vals, csr_rows, x, y);
    CUDA_CHECK();
    spmv_kernel_E<<<blocks_ell, threads, 0, stream2>>>(A, J, x, y);
    CUDA_CHECK();
    cudaStreamSynchronize(stream1);
    CUDA_CHECK();

    cudaStreamSynchronize(stream2);
    CUDA_CHECK();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float nanoseconds = 0;
    cudaEventElapsedTime(&nanoseconds, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return (double)nanoseconds;
}